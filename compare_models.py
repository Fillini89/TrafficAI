import glob
import os
import re
import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment

from chaos_wrapper import ChaosMonkeyWrapper, PhaseSafetyWrapper, RewardInfoWrapper, ServiceDebtGuardrailWrapper
from config import DEFAULT_STRESS_ROUTE_FILE, EVAL_SETTINGS, OUTPUT_DIRS, SIM_SETTINGS, TRAIN_SETTINGS, ensure_output_dirs
from custom_obs import LegacyRadarObservation, RadarObservation
from sb3_compat import load_ppo_compat


AGENT_COLORS = {
    "Baseline": "#6B7280",
    "Gen 8": "#A0CBE8",
    "Gen 9": "#4E79A7",
    "Gen 10": "#F28E2B",
    "Gen 11": "#59A14F",
    "Gen 12": "#B07AA1",
    "Gen 13": "#E15759",
    "Gen 14": "#76B7B2",
    "Gen 15": "#EDC948",
    "Gen 16": "#FF9DA7",
}

METRIC_SPECS = {
    "final_total_wait": ("Final accumulated waiting", "Lower means less leftover congestion", "lower", "{:,.0f}"),
    "p95_total_wait": ("95th percentile waiting", "Lower means fewer severe delays", "lower", "{:,.0f}"),
    "stopped_auc": ("Stopped-vehicle burden", "Lower means less stop-and-go over time", "lower", "{:,.0f}"),
    "mean_speed": ("Average network speed", "Higher means traffic keeps moving", "higher", "{:,.2f}"),
    "phase_switch_count": ("Actual signal changes", "Lower is smoother, but too low can starve lanes", "lower", "{:,.0f}"),
}

FAIRNESS_METRIC_SPECS = {
    "max_raw_worst_lane_wait": ("Worst lane wait", "Lower means no hidden lane starvation", "lower", "{:,.0f}"),
    "p95_raw_worst_lane_wait": ("P95 worst-lane wait", "Lower means fewer severe lane delays", "lower", "{:,.0f}"),
    "max_raw_starved_lanes": ("Max starved lanes", "Lower means fewer lanes above starvation threshold", "lower", "{:,.0f}"),
    "mean_raw_starved_lanes": ("Mean starved lanes", "Lower means starvation is not persistent", "lower", "{:,.2f}"),
    "max_raw_starvation_excess": ("Max starvation excess", "Lower means less accumulated fairness debt", "lower", "{:,.0f}"),
    "max_raw_tail_wait_mean": ("Max tail wait", "Lower means top waiting lanes stay bounded", "lower", "{:,.0f}"),
    "fairness_violation_steps": ("Fairness violation steps", "Lower means fewer hard fairness breaches", "lower", "{:,.0f}"),
}

SMOOTHING_METRIC_SPECS = {
    "avg_seconds_between_switches": ("Seconds between actual switches", "Higher means less signal jitter", "higher", "{:,.1f}"),
    "switches_per_hour": ("Actual switches per hour", "Lower means smoother signal service", "lower", "{:,.1f}"),
    "policy_action_switch_count": ("Policy action changes", "Lower means the learned policy is less twitchy", "lower", "{:,.0f}"),
    "policy_switches_per_hour": ("Policy switches per hour", "Lower means less policy-level jitter", "lower", "{:,.1f}"),
    "actual_phase_switch_count": ("Actual signal changes", "Lower means smoother executed service", "lower", "{:,.0f}"),
    "fairness_forced_switch_count": ("Fairness-forced switches", "Lower means fewer emergency guardrail interventions", "lower", "{:,.0f}"),
    "fairness_guardrail_suppressed_switch_count": ("Guardrail suppressions", "Lower means fewer prevented twitch switches", "lower", "{:,.0f}"),
    "cadence_suppressed_switch_count": ("Cadence suppressions", "Lower means the policy respects minimum hold", "lower", "{:,.0f}"),
    "fairness_guardrail_hold_active_steps": ("Guardrail hold steps", "Lower means fewer emergency protected-hold steps", "lower", "{:,.0f}"),
    "avg_phase_hold_seconds": ("Average phase hold", "Higher means steadier green service", "higher", "{:,.1f}"),
    "p95_phase_hold_seconds": ("P95 phase hold", "Lower helps catch overlong holds", "lower", "{:,.1f}"),
    "max_service_age": ("Max service age", "Lower means waiting approaches get served sooner", "lower", "{:,.1f}"),
    "service_age_over_150_steps": ("Service age >150s steps", "Lower means fewer warning-zone service delays", "lower", "{:,.0f}"),
    "service_age_over_210_steps": ("Service age >210s steps", "Lower means fewer critical service delays", "lower", "{:,.0f}"),
    "service_age_over_150_auc": ("Service-age warning burden", "Lower means less duration and depth above 150s", "lower", "{:,.0f}"),
}

ALL_METRIC_SPECS = {**METRIC_SPECS, **FAIRNESS_METRIC_SPECS, **SMOOTHING_METRIC_SPECS}
FAIRNESS_VIOLATION_WAIT_SECONDS = 120.0


TAIL_REGRESSION_SCENARIOS = {
    "daily_seed30000_demand075",
    "stress_seed40346_demand110",
    "daily_seed11096_demand075",
    "stress_seed40173_demand100",
    "daily_seed30411_demand100",
    "stress_seed21211_demand190",
    "daily_seed11507_demand100",
    "stress_seed20865_demand155",
}


class MetricsSnapshotWrapper(gym.Wrapper):
    """Keep final SUMO metrics before DummyVecEnv auto-resets the env."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            info["terminal_metrics"] = list(getattr(self.env.unwrapped, "metrics", []))
        return obs, reward, terminated, truncated, info


def get_top_models(model_dir, base_name, count=2):
    pattern = re.compile(rf"{base_name}_Gen(\d+)\.zip")
    models = []

    if not os.path.exists(model_dir):
        return []

    for file_name in os.listdir(model_dir):
        match = pattern.search(file_name)
        if match:
            gen_num = int(match.group(1))
            full_path = os.path.join(model_dir, file_name.replace(".zip", ""))
            models.append((gen_num, full_path))

    models.sort(key=lambda item: item[0], reverse=True)
    return models[:count]


def get_observation_class(gen_num):
    return RadarObservation if gen_num >= 10 else LegacyRadarObservation


def safe_ppo_load(model_path, env):
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return load_ppo_compat(model_path, env=env, device="cpu", custom_objects=custom_objects)


def make_eval_settings(route_file, seed, num_seconds, observation_class):
    settings = SIM_SETTINGS.copy()
    settings.pop("max_green", None)
    settings.pop("enforce_max_green", None)
    settings.pop("service_debt_metric", None)
    settings.pop("service_debt_soft_threshold", None)
    settings.pop("service_debt_threshold", None)
    settings.pop("service_debt_min_hold", None)
    settings.pop("service_debt_target_hold", None)
    settings.pop("service_debt_protected_hold", None)
    settings.pop("service_debt_adaptive_threshold", None)
    settings.pop("service_debt_queue_imbalance_threshold", None)
    settings.pop("service_debt_worse_multiplier", None)
    settings.pop("service_debt_worse_threshold", None)
    settings.pop("cadence_suppression_penalty", None)
    settings.pop("service_age_warning_threshold", None)
    settings.pop("service_age_critical_threshold", None)
    settings.pop("service_age_warning_norm", None)
    settings.pop("service_age_critical_norm", None)
    settings.pop("service_age_warning_penalty_weight", None)
    settings.pop("service_age_critical_penalty_weight", None)
    settings.pop("service_age_budget_penalty_clip", None)
    settings.pop("service_age_critical_override", None)
    settings.pop("enforce_service_debt", None)
    settings["route_file"] = route_file
    settings["num_seconds"] = num_seconds
    settings["use_gui"] = False
    settings["observation_class"] = observation_class
    settings.pop("out_csv_name", None)
    command = settings.get("additional_sumo_cmd", "")
    if "--seed" not in command and "--random" not in command:
        settings["additional_sumo_cmd"] = f"{command} --seed {seed}".strip()
    return settings


def service_control_config(gen_num, control_profile):
    if gen_num < 12:
        return None

    delta_time = max(SIM_SETTINGS.get("delta_time", 4), 1)

    if control_profile == "current" or (control_profile == "generation" and gen_num >= 16):
        return {
            "service_threshold_seconds": SIM_SETTINGS.get("service_debt_threshold", 120.0),
            "soft_service_threshold_seconds": SIM_SETTINGS.get("service_debt_soft_threshold", 90.0),
            "min_green_steps": max(int(SIM_SETTINGS.get("min_green", 10) / delta_time), 1),
            "min_hold_steps": max(int(float(SIM_SETTINGS.get("service_debt_min_hold", 24.0)) / delta_time), 0),
            "target_hold_steps": max(int(float(SIM_SETTINGS.get("service_debt_target_hold", 32.0)) / delta_time), 0),
            "protected_hold_steps": max(int(float(SIM_SETTINGS.get("service_debt_protected_hold", 24.0)) / delta_time), 0),
            "adaptive_service_threshold_seconds": SIM_SETTINGS.get("service_debt_adaptive_threshold", 75.0),
            "queue_imbalance_threshold": SIM_SETTINGS.get("service_debt_queue_imbalance_threshold", 6.0),
            "worse_debt_multiplier": SIM_SETTINGS.get("service_debt_worse_multiplier", 1.0),
            "worse_debt_seconds": SIM_SETTINGS.get("service_debt_worse_threshold", 150.0),
            "use_service_age": str(SIM_SETTINGS.get("service_debt_metric", "service_age")).lower() == "service_age",
            "cadence_suppression_penalty": SIM_SETTINGS.get("cadence_suppression_penalty", -0.05),
            "service_age_warning_seconds": SIM_SETTINGS.get("service_age_warning_threshold", 150.0),
            "service_age_critical_seconds": SIM_SETTINGS.get("service_age_critical_threshold", 210.0),
            "service_age_warning_norm": SIM_SETTINGS.get("service_age_warning_norm", 60.0),
            "service_age_critical_norm": SIM_SETTINGS.get("service_age_critical_norm", 60.0),
            "service_age_warning_penalty_weight": SIM_SETTINGS.get("service_age_warning_penalty_weight", 0.0),
            "service_age_critical_penalty_weight": SIM_SETTINGS.get("service_age_critical_penalty_weight", 0.0),
            "service_age_budget_penalty_clip": SIM_SETTINGS.get("service_age_budget_penalty_clip", 0.0),
            "service_age_critical_override": bool(SIM_SETTINGS.get("service_age_critical_override", True)),
            "enforce": bool(SIM_SETTINGS.get("enforce_service_debt", True)),
        }

    if control_profile == "generation" and gen_num == 15:
        return {
            "service_threshold_seconds": SIM_SETTINGS.get("service_debt_threshold", 120.0),
            "soft_service_threshold_seconds": SIM_SETTINGS.get("service_debt_soft_threshold", 90.0),
            "min_green_steps": max(int(SIM_SETTINGS.get("min_green", 10) / delta_time), 1),
            "min_hold_steps": max(int(16.0 / delta_time), 0),
            "target_hold_steps": max(int(24.0 / delta_time), 0),
            "protected_hold_steps": max(int(12.0 / delta_time), 0),
            "adaptive_service_threshold_seconds": 70.0,
            "queue_imbalance_threshold": 5.0,
            "worse_debt_multiplier": 1.0,
            "worse_debt_seconds": 130.0,
            "use_service_age": True,
            "cadence_suppression_penalty": -0.075,
            "service_age_warning_seconds": 150.0,
            "service_age_critical_seconds": 210.0,
            "service_age_warning_norm": 60.0,
            "service_age_critical_norm": 60.0,
            "service_age_warning_penalty_weight": 0.0,
            "service_age_critical_penalty_weight": 0.0,
            "service_age_budget_penalty_clip": 0.0,
            "service_age_critical_override": False,
            "enforce": bool(SIM_SETTINGS.get("enforce_service_debt", True)),
        }

    if control_profile == "generation" and gen_num == 14:
        return {
            "service_threshold_seconds": SIM_SETTINGS.get("service_debt_threshold", 120.0),
            "soft_service_threshold_seconds": SIM_SETTINGS.get("service_debt_soft_threshold", 90.0),
            "min_green_steps": max(int(SIM_SETTINGS.get("min_green", 10) / delta_time), 1),
            "min_hold_steps": max(int(16.0 / delta_time), 0),
            "target_hold_steps": max(int(24.0 / delta_time), 0),
            "protected_hold_steps": max(int(12.0 / delta_time), 0),
            "adaptive_service_threshold_seconds": 75.0,
            "queue_imbalance_threshold": 6.0,
            "worse_debt_multiplier": 1.0,
            "worse_debt_seconds": 135.0,
            "use_service_age": True,
            "cadence_suppression_penalty": -0.05,
            "service_age_warning_seconds": 150.0,
            "service_age_critical_seconds": 210.0,
            "service_age_warning_norm": 60.0,
            "service_age_critical_norm": 60.0,
            "service_age_warning_penalty_weight": 0.0,
            "service_age_critical_penalty_weight": 0.0,
            "service_age_budget_penalty_clip": 0.0,
            "service_age_critical_override": False,
            "enforce": bool(SIM_SETTINGS.get("enforce_service_debt", True)),
        }

    if control_profile == "generation" and gen_num == 13:
        return {
            "service_threshold_seconds": SIM_SETTINGS.get("service_debt_threshold", 120.0),
            "soft_service_threshold_seconds": SIM_SETTINGS.get("service_debt_soft_threshold", 90.0),
            "min_green_steps": max(int(SIM_SETTINGS.get("min_green", 10) / delta_time), 1),
            "min_hold_steps": max(int(24.0 / delta_time), 0),
            "target_hold_steps": max(int(32.0 / delta_time), 0),
            "protected_hold_steps": max(int(24.0 / delta_time), 0),
            "adaptive_service_threshold_seconds": 999999.0,
            "queue_imbalance_threshold": 999999.0,
            "worse_debt_multiplier": 1.0,
            "worse_debt_seconds": 150.0,
            "use_service_age": True,
            "cadence_suppression_penalty": SIM_SETTINGS.get("cadence_suppression_penalty", -0.05),
            "service_age_warning_seconds": 150.0,
            "service_age_critical_seconds": 210.0,
            "service_age_warning_norm": 60.0,
            "service_age_critical_norm": 60.0,
            "service_age_warning_penalty_weight": 0.0,
            "service_age_critical_penalty_weight": 0.0,
            "service_age_budget_penalty_clip": 0.0,
            "service_age_critical_override": False,
            "enforce": bool(SIM_SETTINGS.get("enforce_service_debt", True)),
        }

    return {
        "service_threshold_seconds": SIM_SETTINGS.get("service_debt_threshold", 120.0),
        "soft_service_threshold_seconds": SIM_SETTINGS.get("service_debt_soft_threshold", 90.0),
        "min_green_steps": max(int(SIM_SETTINGS.get("min_green", 10) / delta_time), 1),
        "min_hold_steps": 0,
        "target_hold_steps": 0,
        "protected_hold_steps": 0,
        "adaptive_service_threshold_seconds": SIM_SETTINGS.get("service_debt_adaptive_threshold", 75.0),
        "queue_imbalance_threshold": SIM_SETTINGS.get("service_debt_queue_imbalance_threshold", 6.0),
        "worse_debt_multiplier": 1.25,
        "worse_debt_seconds": None,
        "use_service_age": False,
        "cadence_suppression_penalty": 0.0,
        "service_age_warning_seconds": 150.0,
        "service_age_critical_seconds": 210.0,
        "service_age_warning_norm": 60.0,
        "service_age_critical_norm": 60.0,
        "service_age_warning_penalty_weight": 0.0,
        "service_age_critical_penalty_weight": 0.0,
        "service_age_budget_penalty_clip": 0.0,
        "service_age_critical_override": False,
        "enforce": bool(SIM_SETTINGS.get("enforce_service_debt", True)),
    }


def maybe_wrap_service_debt(env, gen_num, control_profile):
    if gen_num < 12:
        return env
    return ServiceDebtGuardrailWrapper(env, **service_control_config(gen_num, control_profile))


def maybe_wrap_phase_safety(env, gen_num):
    if gen_num < 11:
        return env
    max_green_steps = max(int(SIM_SETTINGS.get("max_green", 96) / max(SIM_SETTINGS.get("delta_time", 4), 1)), 1)
    return PhaseSafetyWrapper(
        env,
        max_green_steps=max_green_steps,
        enforce=bool(SIM_SETTINGS.get("enforce_max_green", True)),
    )


def wrap_eval_env(env, gen_num, control_profile="generation"):
    env = RewardInfoWrapper(env)
    env = maybe_wrap_service_debt(env, gen_num, control_profile)
    return maybe_wrap_phase_safety(env, gen_num)


def collect_reward_components(info, reward_rows):
    if isinstance(info, list):
        for item in info:
            collect_reward_components(item, reward_rows)
        return
    if not isinstance(info, dict):
        return
    components = info.get("reward_components")
    if components:
        reward_rows.append(dict(components))


def collect_step_info(info, info_rows):
    if isinstance(info, list):
        for item in info:
            collect_step_info(item, info_rows)
        return
    if not isinstance(info, dict):
        return

    keys = [
        "phase_hold_steps",
        "phase_forced_switch",
        "requested_action",
        "executed_action",
        "actual_phase_changed",
        "actual_phase_switch_count",
        "fairness_forced_switch",
        "fairness_forced_switch_count",
        "fairness_guardrail_suppressed_switch",
        "fairness_guardrail_suppressed_switch_count",
        "cadence_suppressed_switch",
        "cadence_suppressed_switch_count",
        "cadence_suppression_penalty",
        "adaptive_cadence_release",
        "requested_service_debt",
        "requested_queue_advantage",
        "fairness_guardrail_hold_active",
        "fairness_guardrail_hold_steps_remaining",
        "max_service_debt",
        "max_service_age",
        "hard_service_debt",
        "critical_service_debt",
        "service_age_warning_excess",
        "service_age_critical_excess",
        "service_age_budget_penalty",
    ]
    row = {key: info[key] for key in keys if key in info}
    if row:
        info_rows.append(row)


def evaluate_baseline(scenario, control_profile="generation"):
    env = None
    try:
        env = wrap_eval_env(SumoEnvironment(
            **make_eval_settings(
                scenario["route_file"],
                scenario["seed"],
                scenario["num_seconds"],
                LegacyRadarObservation,
            )
        ), gen_num=0, control_profile=control_profile)
        if scenario["chaos_prob"] > 0:
            env = ChaosMonkeyWrapper(env, chaos_prob=scenario["chaos_prob"], seed=scenario["seed"])

        obs, info = env.reset()
        done = False
        step_counter = 0
        policy_action_switch_count = 0
        previous_action = None
        reward_rows = []
        info_rows = []
        while not done:
            action = (step_counter // 8) % env.action_space.n
            if previous_action is not None and action != previous_action:
                policy_action_switch_count += 1
            previous_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            collect_reward_components(info, reward_rows)
            collect_step_info(info, info_rows)
            done = terminated or truncated
            step_counter += 1

        df = pd.DataFrame(env.unwrapped.metrics)
        df.attrs["policy_action_switch_count"] = policy_action_switch_count
        df.attrs["phase_switch_count"] = policy_action_switch_count
        df.attrs["reward_components"] = reward_rows
        df.attrs["step_info"] = info_rows
        print_eval_status("Baseline", df)
        return df
    finally:
        if env is not None:
            env.close()


def evaluate_model(model_path, gen_num, scenario, control_profile="generation"):
    observation_class = get_observation_class(gen_num)
    settings = make_eval_settings(
        scenario["route_file"],
        scenario["seed"],
        scenario["num_seconds"],
        observation_class,
    )
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"

    if gen_num >= 10 and os.path.exists(vecnormalize_path):
        env = None
        try:
            raw_env = DummyVecEnv(
                [
                    lambda: MetricsSnapshotWrapper(
                        ChaosMonkeyWrapper(
                            wrap_eval_env(SumoEnvironment(**settings), gen_num, control_profile=control_profile),
                            chaos_prob=scenario["chaos_prob"],
                            seed=scenario["seed"],
                        )
                    )
                ]
            )
            env = VecNormalize.load(vecnormalize_path, raw_env)
            env.training = False
            env.norm_reward = False
            model = safe_ppo_load(model_path, env)

            obs = env.reset()
            done = [False]
            policy_action_switch_count = 0
            previous_action = None
            terminal_metrics = None
            reward_rows = []
            info_rows = []
            while not done[0]:
                action, _states = model.predict(obs, deterministic=True)
                action_value = int(action[0]) if hasattr(action, "__len__") else int(action)
                if previous_action is not None and action_value != previous_action:
                    policy_action_switch_count += 1
                previous_action = action_value
                obs, reward, done, info = env.step(action)
                collect_reward_components(info, reward_rows)
                collect_step_info(info, info_rows)
                if done[0] and info and "terminal_metrics" in info[0]:
                    terminal_metrics = info[0]["terminal_metrics"]

            df = pd.DataFrame(terminal_metrics if terminal_metrics is not None else env.venv.envs[0].unwrapped.metrics)
            df.attrs["policy_action_switch_count"] = policy_action_switch_count
            df.attrs["phase_switch_count"] = policy_action_switch_count
            df.attrs["reward_components"] = reward_rows
            df.attrs["step_info"] = info_rows
            print_eval_status(f"Gen {gen_num}", df)
            return df
        finally:
            if env is not None:
                env.close()

    env = None
    try:
        env = wrap_eval_env(SumoEnvironment(**settings), gen_num, control_profile=control_profile)
        if scenario["chaos_prob"] > 0:
            env = ChaosMonkeyWrapper(env, chaos_prob=scenario["chaos_prob"], seed=scenario["seed"])

        model = safe_ppo_load(model_path, env)
        obs, info = env.reset()
        done = False
        policy_action_switch_count = 0
        previous_action = None
        reward_rows = []
        info_rows = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action_value = int(action)
            if previous_action is not None and action_value != previous_action:
                policy_action_switch_count += 1
            previous_action = action_value
            obs, reward, terminated, truncated, info = env.step(action)
            collect_reward_components(info, reward_rows)
            collect_step_info(info, info_rows)
            done = terminated or truncated

        df = pd.DataFrame(env.unwrapped.metrics)
        df.attrs["policy_action_switch_count"] = policy_action_switch_count
        df.attrs["phase_switch_count"] = policy_action_switch_count
        df.attrs["reward_components"] = reward_rows
        df.attrs["step_info"] = info_rows
        print_eval_status(f"Gen {gen_num}", df)
        return df
    finally:
        if env is not None:
            env.close()


def print_eval_status(label, df):
    if df is None or df.empty:
        print(f"WARNING: {label} produced no metrics.")
        return
    last_step = df["step"].iloc[-1] if "step" in df else "unknown"
    stopped = df["system_total_stopped"].iloc[-1] if "system_total_stopped" in df else "unknown"
    wait = df["system_total_waiting_time"].iloc[-1] if "system_total_waiting_time" in df else "unknown"
    speed = df["system_mean_speed"].iloc[-1] if "system_mean_speed" in df else "unknown"
    print(f"{label} metrics: rows={len(df)}, last_step={last_step}, stopped={stopped}, wait={wait}, speed={speed}")


def summarize_metrics(label, scenario_name, df):
    if df is None or df.empty:
        return {}

    reward_df = pd.DataFrame(df.attrs.get("reward_components", []))
    step_info_df = pd.DataFrame(df.attrs.get("step_info", []))

    def reward_max(column):
        if reward_df.empty or column not in reward_df:
            return 0.0
        return reward_df[column].max()

    def reward_mean(column):
        if reward_df.empty or column not in reward_df:
            return 0.0
        return reward_df[column].mean()

    def reward_p95(column):
        if reward_df.empty or column not in reward_df:
            return 0.0
        return reward_df[column].quantile(0.95)

    if reward_df.empty:
        fairness_violation_steps = 0
    else:
        worst_wait = reward_df.get("raw_worst_lane_wait", pd.Series(0.0, index=reward_df.index))
        starved = reward_df.get("raw_starved_lanes", pd.Series(0.0, index=reward_df.index))
        fairness_violation_steps = int(((worst_wait >= FAIRNESS_VIOLATION_WAIT_SECONDS) | (starved >= 2.0)).sum())

    policy_action_switch_count = float(df.attrs.get("policy_action_switch_count", df.attrs.get("phase_switch_count", 0)))
    duration_seconds = 0.0
    if "step" in df and not df["step"].empty:
        duration_seconds = float(df["step"].iloc[-1])
    if duration_seconds <= 0.0:
        duration_seconds = float(len(df) * max(SIM_SETTINGS.get("delta_time", 4), 1))

    policy_switches_per_hour = policy_action_switch_count / max(duration_seconds / 3600.0, 1e-9)

    delta_time = float(max(SIM_SETTINGS.get("delta_time", 4), 1))
    if step_info_df.empty:
        actual_phase_switch_count = policy_action_switch_count
        fairness_forced_switch_count = 0.0
        suppressed_switch_count = 0.0
        cadence_suppressed_switch_count = 0.0
        hold_active_steps = 0.0
        max_service_age = 0.0
        service_age_over_150_steps = 0.0
        service_age_over_210_steps = 0.0
        service_age_over_150_auc = 0.0
        if "raw_phase_hold_seconds" in reward_df:
            phase_hold_seconds = pd.to_numeric(reward_df["raw_phase_hold_seconds"], errors="coerce").dropna()
            avg_phase_hold_seconds = float(phase_hold_seconds.mean()) if not phase_hold_seconds.empty else 0.0
            p95_phase_hold_seconds = float(phase_hold_seconds.quantile(0.95)) if not phase_hold_seconds.empty else 0.0
        else:
            avg_phase_hold_seconds = 0.0
            p95_phase_hold_seconds = 0.0
    else:
        if "actual_phase_switch_count" in step_info_df:
            actual_phase_switch_count = float(step_info_df["actual_phase_switch_count"].max())
        else:
            actual_phase_switch_count = float(
                step_info_df.get("actual_phase_changed", pd.Series(dtype=float)).astype(bool).sum()
            )
        if actual_phase_switch_count <= 0.0:
            actual_phase_switch_count = policy_action_switch_count

        if "fairness_forced_switch_count" in step_info_df:
            fairness_forced_switch_count = float(step_info_df["fairness_forced_switch_count"].max())
        else:
            fairness_forced_switch_count = float(step_info_df.get("fairness_forced_switch", pd.Series(dtype=float)).astype(bool).sum())

        if "fairness_guardrail_suppressed_switch_count" in step_info_df:
            suppressed_switch_count = float(step_info_df["fairness_guardrail_suppressed_switch_count"].max())
        else:
            suppressed_switch_count = float(
                step_info_df.get("fairness_guardrail_suppressed_switch", pd.Series(dtype=float)).astype(bool).sum()
            )

        if "cadence_suppressed_switch_count" in step_info_df:
            cadence_suppressed_switch_count = float(step_info_df["cadence_suppressed_switch_count"].max())
        else:
            cadence_suppressed_switch_count = float(
                step_info_df.get("cadence_suppressed_switch", pd.Series(dtype=float)).astype(bool).sum()
            )

        hold_active_steps = float(step_info_df.get("fairness_guardrail_hold_active", pd.Series(dtype=float)).astype(bool).sum())
        if "max_service_age" in step_info_df:
            service_age_series = pd.to_numeric(step_info_df["max_service_age"], errors="coerce").fillna(0.0)
            max_service_age = float(service_age_series.max())
        else:
            service_age_series = pd.Series(dtype=float)
            max_service_age = 0.0

        if "service_age_warning_excess" in step_info_df:
            warning_excess = pd.to_numeric(step_info_df["service_age_warning_excess"], errors="coerce").fillna(0.0)
        elif not service_age_series.empty:
            warning_excess = (service_age_series - 150.0).clip(lower=0.0)
        else:
            warning_excess = pd.Series(dtype=float)

        if "service_age_critical_excess" in step_info_df:
            critical_excess = pd.to_numeric(step_info_df["service_age_critical_excess"], errors="coerce").fillna(0.0)
        elif not service_age_series.empty:
            critical_excess = (service_age_series - 210.0).clip(lower=0.0)
        else:
            critical_excess = pd.Series(dtype=float)

        service_age_over_150_steps = float((warning_excess > 0.0).sum()) if not warning_excess.empty else 0.0
        service_age_over_210_steps = float((critical_excess > 0.0).sum()) if not critical_excess.empty else 0.0
        service_age_over_150_auc = float(warning_excess.sum() * delta_time) if not warning_excess.empty else 0.0

        if "phase_hold_steps" in step_info_df:
            phase_hold_seconds = pd.to_numeric(step_info_df["phase_hold_steps"], errors="coerce").dropna() * delta_time
            avg_phase_hold_seconds = float(phase_hold_seconds.mean()) if not phase_hold_seconds.empty else 0.0
            p95_phase_hold_seconds = float(phase_hold_seconds.quantile(0.95)) if not phase_hold_seconds.empty else 0.0
        elif "raw_phase_hold_seconds" in reward_df:
            phase_hold_seconds = pd.to_numeric(reward_df["raw_phase_hold_seconds"], errors="coerce").dropna()
            avg_phase_hold_seconds = float(phase_hold_seconds.mean()) if not phase_hold_seconds.empty else 0.0
            p95_phase_hold_seconds = float(phase_hold_seconds.quantile(0.95)) if not phase_hold_seconds.empty else 0.0
        else:
            avg_phase_hold_seconds = 0.0
            p95_phase_hold_seconds = 0.0

    avg_seconds_between_switches = (
        duration_seconds / actual_phase_switch_count if actual_phase_switch_count > 0 else duration_seconds
    )
    switches_per_hour = actual_phase_switch_count / max(duration_seconds / 3600.0, 1e-9)

    return {
        "agent": label,
        "scenario": scenario_name,
        "final_total_wait": df["system_total_waiting_time"].iloc[-1],
        "mean_total_wait": df["system_total_waiting_time"].mean(),
        "p95_total_wait": df["system_total_waiting_time"].quantile(0.95),
        "max_stopped": df["system_total_stopped"].max(),
        "stopped_auc": df["system_total_stopped"].sum(),
        "mean_speed": df["system_mean_speed"].mean(),
        "phase_switch_count": actual_phase_switch_count,
        "max_raw_worst_lane_wait": reward_max("raw_worst_lane_wait"),
        "p95_raw_worst_lane_wait": reward_p95("raw_worst_lane_wait"),
        "max_raw_starved_lanes": reward_max("raw_starved_lanes"),
        "mean_raw_starved_lanes": reward_mean("raw_starved_lanes"),
        "max_raw_starvation_excess": reward_max("raw_starvation_excess"),
        "max_raw_tail_wait_mean": reward_max("raw_tail_wait_mean"),
        "fairness_violation_steps": fairness_violation_steps,
        "avg_seconds_between_switches": avg_seconds_between_switches,
        "switches_per_hour": switches_per_hour,
        "policy_action_switch_count": policy_action_switch_count,
        "policy_switches_per_hour": policy_switches_per_hour,
        "actual_phase_switch_count": actual_phase_switch_count,
        "fairness_forced_switch_count": fairness_forced_switch_count,
        "fairness_guardrail_suppressed_switch_count": suppressed_switch_count,
        "cadence_suppressed_switch_count": cadence_suppressed_switch_count,
        "fairness_guardrail_hold_active_steps": hold_active_steps,
        "avg_phase_hold_seconds": avg_phase_hold_seconds,
        "p95_phase_hold_seconds": p95_phase_hold_seconds,
        "max_service_age": max_service_age,
        "service_age_over_150_steps": service_age_over_150_steps,
        "service_age_over_210_steps": service_age_over_210_steps,
        "service_age_over_150_auc": service_age_over_150_auc,
    }


def configure_worker_threads():
    try:
        import torch
    except ImportError:
        return

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def scenario_name_from_route(route_file):
    file_name = os.path.basename(route_file)
    if file_name.endswith(".rou.xml"):
        return file_name[:-8]
    return os.path.splitext(file_name)[0]


def build_agent_specs(top_models):
    specs = [{"agent_index": 0, "label": "Baseline", "kind": "baseline", "gen_num": None, "model_path": None}]
    for index, (gen_num, model_path) in enumerate(reversed(top_models), start=1):
        specs.append(
            {
                "agent_index": index,
                "label": f"Gen {gen_num}",
                "kind": "model",
                "gen_num": gen_num,
                "model_path": model_path,
            }
        )
    return specs


def build_eval_tasks(scenarios, agent_specs, jobs, worker_stagger_seconds, collect_run_metrics, control_profile):
    tasks = []
    task_index = 0
    for scenario_index, scenario in enumerate(scenarios):
        for agent_spec in agent_specs:
            task = {
                "task_index": task_index,
                "scenario_index": scenario_index,
                "scenario": scenario,
                "agent_spec": agent_spec,
                "jobs": jobs,
                "worker_stagger_seconds": worker_stagger_seconds,
                "collect_run_metrics": collect_run_metrics,
                "control_profile": control_profile,
            }
            tasks.append(task)
            task_index += 1
    return tasks


def run_eval_task(task):
    started = time.perf_counter()
    scenario = task["scenario"]
    agent_spec = task["agent_spec"]
    label = agent_spec["label"]
    metric_path = None

    try:
        stagger = max(float(task.get("worker_stagger_seconds", 0.0)), 0.0)
        jobs = max(int(task.get("jobs", 1)), 1)
        if stagger:
            time.sleep((int(task["task_index"]) % jobs) * stagger)

        configure_worker_threads()
        control_profile = task.get("control_profile", "generation")
        if agent_spec["kind"] == "baseline":
            df = evaluate_baseline(scenario, control_profile=control_profile)
        else:
            df = evaluate_model(agent_spec["model_path"], agent_spec["gen_num"], scenario, control_profile=control_profile)

        summary = summarize_metrics(label, scenario["name"], df)
        if task.get("collect_run_metrics"):
            os.makedirs(OUTPUT_DIRS["sumo_compare"], exist_ok=True)
            metric_path = os.path.join(
                OUTPUT_DIRS["sumo_compare"],
                f"metrics_{sanitize_filename(scenario['name'])}_{sanitize_filename(label)}.csv",
            )
            df.to_csv(metric_path, index=False)

        return {
            "status": "ok",
            "scenario_index": task["scenario_index"],
            "agent_index": agent_spec["agent_index"],
            "scenario_name": scenario["name"],
            "agent": label,
            "summary": summary,
            "metric_path": metric_path,
            "elapsed": time.perf_counter() - started,
        }
    except Exception:
        return {
            "status": "error",
            "scenario_index": task["scenario_index"],
            "agent_index": agent_spec["agent_index"],
            "scenario_name": scenario["name"],
            "agent": label,
            "traceback": traceback.format_exc(),
            "elapsed": time.perf_counter() - started,
        }


def run_parallel_evaluation_ordered(tasks, jobs):
    results = []
    metric_paths = {}
    total = len(tasks)
    started = time.perf_counter()

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_eval_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            scenario_name = result["scenario_name"]
            label = result["agent"]
            elapsed_total = time.perf_counter() - started

            if result["status"] != "ok":
                raise RuntimeError(
                    f"Parallel evaluation failed for {label} on {scenario_name}:\n{result['traceback']}"
                )

            results.append(result)
            if result["metric_path"]:
                metric_paths[(scenario_name, label)] = result["metric_path"]
            print(
                f"[{completed}/{total}] {label} | {scenario_name} "
                f"finished in {result['elapsed']:.1f}s (elapsed {elapsed_total:.1f}s)"
            )

    results.sort(key=lambda item: (item["scenario_index"], item["agent_index"]))
    return [result["summary"] for result in results], metric_paths


def plot_parallel_metrics(scenarios, agent_specs, metric_paths):
    for scenario in scenarios:
        df_dict = {}
        for agent_spec in agent_specs:
            label = agent_spec["label"]
            metric_path = metric_paths.get((scenario["name"], label))
            if metric_path and os.path.exists(metric_path):
                df_dict[label] = pd.read_csv(metric_path)
        if df_dict:
            plot_metrics(df_dict, scenario["name"])


def plot_metrics(df_dict, scenario_name):
    plt.figure(figsize=(17, 14), facecolor="white")
    metrics = {
        "system_total_waiting_time": ("Accumulated waiting", "lower is better"),
        "system_total_stopped": ("Stopped vehicles", "lower is better"),
        "system_mean_speed": ("Average speed", "higher is better"),
    }
    smoothing_window = 225

    for index, (column, (title, subtitle)) in enumerate(metrics.items(), 1):
        ax = plt.subplot(4, 1, index)
        for label, df in df_dict.items():
            if df is not None and not df.empty and column in df:
                hours = df["step"] / 3600
                smoothed_values = df[column].rolling(window=smoothing_window, min_periods=1).mean()
                if smoothed_values.dropna().empty:
                    print(f"WARNING: {label} has no valid values for {column}.")
                    continue
                ax.plot(
                    hours,
                    smoothed_values,
                    label=label,
                    color=agent_color(label),
                    linewidth=2.7,
                    alpha=0.95,
                )

        ax.set_title(f"{title} | {subtitle}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Virtual time (hours)", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=11, loc="upper left")

    ax = plt.subplot(4, 1, 4)
    normalized_columns = ["system_total_waiting_time", "system_total_stopped"]
    for label, df in df_dict.items():
        if df is None or df.empty:
            continue
        normalized_parts = []
        for column in normalized_columns:
            if column not in df:
                continue
            series = df[column].rolling(window=smoothing_window, min_periods=1).mean()
            max_value = series.max()
            if max_value and max_value > 0:
                normalized_parts.append(series / max_value)
        if not normalized_parts:
            print(f"WARNING: {label} has no valid congestion metrics for normalized plot.")
            continue
        normalized = sum(normalized_parts) / len(normalized_parts)
        ax.plot(
            df["step"] / 3600,
            normalized,
            label=label,
            color=agent_color(label),
            linewidth=2.7,
            alpha=0.95,
        )

    ax.set_title("Normalized congestion score | lower is better", fontsize=14, fontweight="bold")
    ax.set_xlabel("Virtual time (hours)", fontsize=12)
    ax.set_ylabel("0..1 per agent", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=11, loc="upper left")

    plt.suptitle(f"TrafficAI holdout scenario: {friendly_scenario_name(scenario_name)}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS["comparison_reports"], exist_ok=True)
    report_path = os.path.join(OUTPUT_DIRS["comparison_reports"], f"comparison_{scenario_name}.png")
    plt.savefig(report_path, dpi=300)
    print(f"Report saved: {report_path}")


def build_scenarios(full=False, max_daily=2, max_stress=2, daily_seconds=14400, stress_seconds=14400, tail_regression=False):
    base_seed = EVAL_SETTINGS["seed"]
    holdout_routes = discover_holdout_routes()
    scenarios = []
    daily_count = 0
    stress_count = 0

    for index, route_file in enumerate(holdout_routes):
        name = scenario_name_from_route(route_file)
        if tail_regression and name not in TAIL_REGRESSION_SCENARIOS:
            continue
        is_stress = "stress" in name
        if not full and not tail_regression:
            if is_stress and stress_count >= max_stress:
                continue
            if not is_stress and daily_count >= max_daily:
                continue

        num_seconds = stress_seconds if is_stress else daily_seconds
        chaos_prob = [0.0, 0.001, 0.002][index % 3] if is_stress else [0.0, 0.0005][index % 2]
        scenarios.append(
            {
                "name": name,
                "route_file": route_file,
                "chaos_prob": chaos_prob,
                "seed": base_seed + index,
                "num_seconds": num_seconds,
            }
        )
        if is_stress:
            stress_count += 1
        else:
            daily_count += 1

    return scenarios


def discover_holdout_routes():
    route_files = []
    for pattern in EVAL_SETTINGS["holdout_route_patterns"]:
        route_files.extend(glob.glob(os.path.join(EVAL_SETTINGS["holdout_route_dir"], pattern)))
    route_files = sorted(set(route_files))
    return route_files or [DEFAULT_STRESS_ROUTE_FILE]


def agent_sort_key(agent):
    if agent == "Baseline":
        return -1
    match = re.search(r"Gen\s+(\d+)", str(agent))
    return int(match.group(1)) if match else 999


def agent_color(agent):
    return AGENT_COLORS.get(agent, "#9C755F")


def friendly_scenario_name(name):
    match = re.match(r"(daily|stress)_seed(\d+)_demand(\d+)", name)
    if not match:
        return name
    kind, seed, demand = match.groups()
    kind_label = "Daily traffic" if kind == "daily" else "Stress traffic"
    return f"{kind_label} | demand {int(demand)}% | seed {seed}"


def value_is_better(value, reference, direction):
    if direction == "higher":
        return value > reference
    return value < reference


def percent_delta(value, reference, direction):
    if reference == 0:
        return ""
    raw_delta = ((value - reference) / abs(reference)) * 100.0
    better_delta = raw_delta if direction == "higher" else -raw_delta
    if abs(better_delta) < 0.05:
        return "0.0%"
    sign = "+" if better_delta >= 0 else ""
    return f"{sign}{better_delta:.1f}%"


def format_metric(value, fmt):
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def build_scorecard(summary):
    metric_specs = {metric: spec for metric, spec in ALL_METRIC_SPECS.items() if metric in summary.columns}
    metric_names = list(metric_specs)
    grouped = (
        summary.groupby("agent", as_index=False)
        .agg(
            scenarios=("scenario", "count"),
            **{metric: (metric, "mean") for metric in metric_names},
        )
        .sort_values("agent", key=lambda series: series.map(agent_sort_key))
        .reset_index(drop=True)
    )

    baseline_rows = grouped[grouped["agent"] == "Baseline"]
    if baseline_rows.empty:
        return grouped

    baseline = baseline_rows.iloc[0]
    for metric, (_title, _subtitle, direction, _fmt) in metric_specs.items():
        grouped[f"{metric}_vs_baseline"] = grouped[metric].apply(lambda value: percent_delta(value, baseline[metric], direction))

    return grouped


def plot_summary_dashboard(summary, suffix):
    if summary.empty:
        return None

    scorecard = build_scorecard(summary)
    agents = list(scorecard["agent"])
    x_positions = range(len(agents))
    dashboard_specs = {
        **METRIC_SPECS,
        "max_raw_worst_lane_wait": FAIRNESS_METRIC_SPECS["max_raw_worst_lane_wait"],
        "max_raw_starved_lanes": FAIRNESS_METRIC_SPECS["max_raw_starved_lanes"],
        "max_raw_starvation_excess": FAIRNESS_METRIC_SPECS["max_raw_starvation_excess"],
        "fairness_violation_steps": FAIRNESS_METRIC_SPECS["fairness_violation_steps"],
        "avg_seconds_between_switches": SMOOTHING_METRIC_SPECS["avg_seconds_between_switches"],
        "switches_per_hour": SMOOTHING_METRIC_SPECS["switches_per_hour"],
        "policy_action_switch_count": SMOOTHING_METRIC_SPECS["policy_action_switch_count"],
        "avg_phase_hold_seconds": SMOOTHING_METRIC_SPECS["avg_phase_hold_seconds"],
        "max_service_age": SMOOTHING_METRIC_SPECS["max_service_age"],
        "service_age_over_210_steps": SMOOTHING_METRIC_SPECS["service_age_over_210_steps"],
        "service_age_over_150_auc": SMOOTHING_METRIC_SPECS["service_age_over_150_auc"],
    }
    dashboard_specs = {metric: spec for metric, spec in dashboard_specs.items() if metric in scorecard.columns}
    column_count = 3
    row_count = max((len(dashboard_specs) + column_count - 1) // column_count, 1)
    fig, axes = plt.subplots(row_count, column_count, figsize=(20, 4.8 * row_count), facecolor="white")
    axes = axes.flatten()

    for index, (metric, (title, subtitle, direction, fmt)) in enumerate(dashboard_specs.items()):
        ax = axes[index]
        values = list(scorecard[metric].astype(float))
        colors = [agent_color(agent) for agent in agents]
        bars = ax.bar(x_positions, values, color=colors, width=0.68)
        upper = max(values) * 1.18 if values else 1.0
        ax.set_ylim(0, upper)

        best_value = max(values) if direction == "higher" else min(values)
        for agent, value, bar in zip(agents, values, bars):
            label = format_metric(value, fmt)
            if value == best_value:
                label = f"{label}  Best"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.015,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if value == best_value else "normal",
            )

        ax.set_xticks(list(x_positions), agents, rotation=0)
        ax.text(0.0, 1.10, title, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9.5, color="#4B5563")
        ax.grid(True, axis="y", linestyle="--", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(dashboard_specs):]:
        ax.axis("off")
    scenario_count = summary["scenario"].nunique()
    if suffix == "full":
        mode_label = "Full daily holdout"
    elif suffix == "tail_full":
        mode_label = "Full tail-regression holdout"
    elif suffix == "tail_quick":
        mode_label = "Quick tail-regression holdout"
    else:
        mode_label = "Quick holdout"
    fig.suptitle(
        f"TrafficAI model comparison - {mode_label}",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        f"Average across {scenario_count} scenarios. Lower waiting and fairness violations are better; speed is better when higher.",
        ha="center",
        fontsize=11,
        color="#374151",
    )

    plt.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.88, hspace=0.72, wspace=0.28)
    output_path = os.path.join(OUTPUT_DIRS["holdout_reports"], f"holdout_eval_dashboard_{suffix}.png")
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return output_path


def write_scorecard_csv(scorecard, suffix):
    output_path = os.path.join(OUTPUT_DIRS["holdout_reports"], f"holdout_eval_scorecard_{suffix}.csv")
    scorecard.to_csv(output_path, index=False)
    return output_path


def write_markdown_report(summary, scorecard, suffix):
    output_path = os.path.join(OUTPUT_DIRS["holdout_reports"], f"holdout_eval_report_{suffix}.md")
    agents = sorted(summary["agent"].unique(), key=agent_sort_key)
    latest_agents = [agent for agent in agents if agent != "Baseline"]
    latest_agent = latest_agents[-1] if latest_agents else agents[-1]
    previous_agent = latest_agents[-2] if len(latest_agents) >= 2 else "Baseline"

    rows_by_agent = {row["agent"]: row for _, row in scorecard.iterrows()}
    latest = rows_by_agent.get(latest_agent)
    previous = rows_by_agent.get(previous_agent)
    baseline = rows_by_agent.get("Baseline")

    lines = [
        f"# TrafficAI Holdout Report ({suffix})",
        "",
        f"Scenarios evaluated: {summary['scenario'].nunique()}",
        f"Models compared: {', '.join(agents)}",
        "",
        "## Executive Takeaway",
        "",
    ]

    if latest is not None and previous is not None:
        final_wait_delta = percent_delta(latest["final_total_wait"], previous["final_total_wait"], "lower")
        p95_delta = percent_delta(latest["p95_total_wait"], previous["p95_total_wait"], "lower")
        speed_delta = percent_delta(latest["mean_speed"], previous["mean_speed"], "higher")
        lines.append(
            f"{latest_agent} vs {previous_agent}: final waiting {final_wait_delta}, "
            f"95th-percentile waiting {p95_delta}, average speed {speed_delta}."
        )
    if suffix in {"quick", "tail_quick"}:
        lines.append("Quick holdout is a regression screen, not final fairness proof. Full daily evaluation remains decisive.")
    elif suffix == "tail_full":
        lines.append("Tail-regression holdout focuses on scenarios where Gen11 had the worst 95th-percentile waiting.")
    else:
        lines.append("Full holdout is the main evidence for long-horizon starvation and fairness.")

    lines.extend(["", "## Scorecard", ""])
    header = "| Agent | Final wait | P95 wait | Stopped burden | Mean speed | Signal changes |"
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in scorecard.iterrows():
        lines.append(
            "| {agent} | {final_wait} | {p95_wait} | {stopped} | {speed} | {switches} |".format(
                agent=row["agent"],
                final_wait=format_metric(row["final_total_wait"], METRIC_SPECS["final_total_wait"][3]),
                p95_wait=format_metric(row["p95_total_wait"], METRIC_SPECS["p95_total_wait"][3]),
                stopped=format_metric(row["stopped_auc"], METRIC_SPECS["stopped_auc"][3]),
                speed=format_metric(row["mean_speed"], METRIC_SPECS["mean_speed"][3]),
                switches=format_metric(row["phase_switch_count"], METRIC_SPECS["phase_switch_count"][3]),
            )
        )

    lines.extend(["", "## Hidden Starvation Check", ""])
    fairness_columns = [metric for metric in FAIRNESS_METRIC_SPECS if metric in scorecard.columns]
    if fairness_columns:
        lines.append("| Agent | Worst lane wait | P95 worst-lane wait | Max starved lanes | Max starvation excess | Fairness violations |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in scorecard.iterrows():
            lines.append(
                "| {agent} | {worst} | {p95_worst} | {starved} | {excess} | {violations} |".format(
                    agent=row["agent"],
                    worst=format_metric(row.get("max_raw_worst_lane_wait", 0), FAIRNESS_METRIC_SPECS["max_raw_worst_lane_wait"][3]),
                    p95_worst=format_metric(row.get("p95_raw_worst_lane_wait", 0), FAIRNESS_METRIC_SPECS["p95_raw_worst_lane_wait"][3]),
                    starved=format_metric(row.get("max_raw_starved_lanes", 0), FAIRNESS_METRIC_SPECS["max_raw_starved_lanes"][3]),
                    excess=format_metric(row.get("max_raw_starvation_excess", 0), FAIRNESS_METRIC_SPECS["max_raw_starvation_excess"][3]),
                    violations=format_metric(row.get("fairness_violation_steps", 0), FAIRNESS_METRIC_SPECS["fairness_violation_steps"][3]),
                )
            )

    lines.extend(["", "## Signal Smoothness Check", ""])
    smoothing_columns = [metric for metric in SMOOTHING_METRIC_SPECS if metric in scorecard.columns]
    if smoothing_columns:
        lines.append("| Agent | Actual switches | Policy switches | Seconds between actual switches | Actual switches/hour | Cadence suppressed | Max service age | Avg hold |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in scorecard.iterrows():
            lines.append(
                "| {agent} | {actual} | {policy} | {between} | {per_hour} | {cadence} | {service_age} | {avg_hold} |".format(
                    agent=row["agent"],
                    actual=format_metric(row.get("actual_phase_switch_count", row.get("phase_switch_count", 0)), "{:,.0f}"),
                    policy=format_metric(row.get("policy_action_switch_count", 0), "{:,.0f}"),
                    between=format_metric(
                        row.get("avg_seconds_between_switches", 0),
                        SMOOTHING_METRIC_SPECS["avg_seconds_between_switches"][3],
                    ),
                    per_hour=format_metric(row.get("switches_per_hour", 0), SMOOTHING_METRIC_SPECS["switches_per_hour"][3]),
                    cadence=format_metric(row.get("cadence_suppressed_switch_count", 0), "{:,.0f}"),
                    service_age=format_metric(row.get("max_service_age", 0), "{:,.1f}"),
                    avg_hold=format_metric(row.get("avg_phase_hold_seconds", 0), SMOOTHING_METRIC_SPECS["avg_phase_hold_seconds"][3]),
                )
            )

    service_budget_columns = [
        metric for metric in ("service_age_over_150_steps", "service_age_over_210_steps", "service_age_over_150_auc")
        if metric in scorecard.columns
    ]
    if service_budget_columns:
        lines.extend(["", "## Service-Age Budget Check", ""])
        lines.append("| Agent | Max service age | >150s steps | >210s steps | >150s burden |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in scorecard.iterrows():
            lines.append(
                "| {agent} | {max_age} | {warning_steps} | {critical_steps} | {warning_auc} |".format(
                    agent=row["agent"],
                    max_age=format_metric(row.get("max_service_age", 0), "{:,.1f}"),
                    warning_steps=format_metric(row.get("service_age_over_150_steps", 0), "{:,.0f}"),
                    critical_steps=format_metric(row.get("service_age_over_210_steps", 0), "{:,.0f}"),
                    warning_auc=format_metric(row.get("service_age_over_150_auc", 0), "{:,.0f}"),
                )
            )

    lines.extend(["", "## Baseline Improvement", ""])
    if baseline is not None:
        lines.append("| Agent | Final wait | P95 wait | Stopped burden | Mean speed |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in scorecard.iterrows():
            lines.append(
                "| {agent} | {final_wait} | {p95_wait} | {stopped} | {speed} |".format(
                    agent=row["agent"],
                    final_wait=row.get("final_total_wait_vs_baseline", ""),
                    p95_wait=row.get("p95_total_wait_vs_baseline", ""),
                    stopped=row.get("stopped_auc_vs_baseline", ""),
                    speed=row.get("mean_speed_vs_baseline", ""),
                )
            )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return output_path


def write_presentation_reports(summary, suffix):
    if summary.empty:
        return {}

    os.makedirs(OUTPUT_DIRS["holdout_reports"], exist_ok=True)
    scorecard = build_scorecard(summary)
    return {
        "dashboard": plot_summary_dashboard(summary, suffix),
        "scorecard": write_scorecard_csv(scorecard, suffix),
        "markdown": write_markdown_report(summary, scorecard, suffix),
    }


def load_existing_summary(suffix):
    summary_path = os.path.join(OUTPUT_DIRS["holdout_reports"], f"holdout_eval_summary_{suffix}.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Holdout summary not found: {summary_path}")
    return pd.read_csv(summary_path), summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and PPO generations on holdout SUMO routes.")
    parser.add_argument("--full", action="store_true", help="Evaluate every holdout route with full scenario durations.")
    parser.add_argument("--models", type=int, default=2, help="How many latest generations to compare. Default: 2.")
    parser.add_argument("--max-daily", type=int, default=2, help="Quick mode daily route count.")
    parser.add_argument("--max-stress", type=int, default=2, help="Quick mode stress route count.")
    parser.add_argument("--daily-seconds", type=int, default=14400, help="Daily scenario duration in quick mode.")
    parser.add_argument("--stress-seconds", type=int, default=14400, help="Stress scenario duration.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-scenario PNG plots and only write CSV summary.")
    parser.add_argument(
        "--report-only",
        choices=["quick", "full", "tail_quick", "tail_full"],
        help="Build presentation reports from an existing summary CSV.",
    )
    parser.add_argument("--tail-regression", action="store_true", help="Evaluate the Gen11 worst-p95 scenario subset.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel SUMO evaluation workers. Default: 1.")
    parser.add_argument("--worker-stagger-seconds", type=float, default=1.0, help="Delay worker startup slots on Windows. Default: 1.0.")
    parser.add_argument("--no-run-metrics", action="store_true", help="Do not persist per-run metrics CSV files for plotting.")
    parser.add_argument(
        "--control-profile",
        choices=["generation", "current"],
        default="generation",
        help="Use generation-matched wrappers, or force current control wrappers for all supported models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ensure_output_dirs()
    args = parse_args()
    if args.report_only:
        summary, summary_path = load_existing_summary(args.report_only)
        report_paths = write_presentation_reports(summary, args.report_only)
        print(f"Loaded holdout summary: {summary_path}")
        for label, path in report_paths.items():
            print(f"{label.title()} report saved: {path}")
        raise SystemExit(0)

    model_dir = "models"
    base_name = TRAIN_SETTINGS["model_name"]
    top_models = get_top_models(model_dir, base_name, count=args.models)
    scenarios = build_scenarios(
        full=args.full,
        max_daily=args.max_daily,
        max_stress=args.max_stress,
        daily_seconds=86400 if args.full else args.daily_seconds,
        stress_seconds=args.stress_seconds,
        tail_regression=args.tail_regression,
    )
    summary_rows = []

    agent_count = 1 + len(top_models)
    total_runs = len(scenarios) * agent_count
    mode = "TAIL FULL" if args.tail_regression and args.full else "TAIL QUICK" if args.tail_regression else "FULL" if args.full else "QUICK"
    print(f"{mode} comparison: {len(scenarios)} scenarios x {agent_count} agents = {total_runs} SUMO runs")
    print(f"Models: {[f'Gen {gen}' for gen, _ in reversed(top_models)]}")
    print(f"Control profile: {args.control_profile}")

    jobs = max(int(args.jobs), 1)
    if jobs > 1:
        collect_run_metrics = not args.no_plots and not args.no_run_metrics
        if not args.no_plots and args.no_run_metrics:
            print("WARNING: --no-run-metrics prevents per-scenario plots in parallel mode.")
        agent_specs = build_agent_specs(top_models)
        tasks = build_eval_tasks(
            scenarios,
            agent_specs,
            jobs=jobs,
            worker_stagger_seconds=args.worker_stagger_seconds,
            collect_run_metrics=collect_run_metrics,
            control_profile=args.control_profile,
        )
        print(
            f"Parallel evaluation enabled: jobs={jobs}, "
            f"worker_stagger_seconds={args.worker_stagger_seconds}, "
            f"collect_run_metrics={collect_run_metrics}"
        )
        summary_rows, metric_paths = run_parallel_evaluation_ordered(tasks, jobs)
        if not args.no_plots and collect_run_metrics:
            plot_parallel_metrics(scenarios, agent_specs, metric_paths)
    else:
        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']} ({scenario['num_seconds']} sim seconds)")
            results = {"Baseline": evaluate_baseline(scenario, control_profile=args.control_profile)}
            summary_rows.append(summarize_metrics("Baseline", scenario["name"], results["Baseline"]))

            for gen_num, model_path in reversed(top_models):
                label = f"Gen {gen_num}"
                print(f"Evaluating {label}: {model_path}")
                results[label] = evaluate_model(model_path, gen_num, scenario, control_profile=args.control_profile)
                summary_rows.append(summarize_metrics(label, scenario["name"], results[label]))

            if not args.no_plots:
                plot_metrics(results, scenario["name"])

    summary = pd.DataFrame([row for row in summary_rows if row])
    suffix = "tail_full" if args.tail_regression and args.full else "tail_quick" if args.tail_regression else "full" if args.full else "quick"
    summary_path = os.path.join(OUTPUT_DIRS["holdout_reports"], f"holdout_eval_summary_{suffix}.csv")
    summary.to_csv(summary_path, index=False)
    report_paths = write_presentation_reports(summary, suffix)
    print(f"\nHoldout summary saved: {summary_path}")
    for label, path in report_paths.items():
        print(f"{label.title()} report saved: {path}")
    print(summary)
