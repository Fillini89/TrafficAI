import glob
import os
import re
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment

from chaos_wrapper import ChaosMonkeyWrapper, PhaseSafetyWrapper
from config import DEFAULT_STRESS_ROUTE_FILE, EVAL_SETTINGS, OUTPUT_DIRS, SIM_SETTINGS, TRAIN_SETTINGS, ensure_output_dirs
from custom_obs import LegacyRadarObservation, RadarObservation
from sb3_compat import install_numpy_pickle_aliases


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
    install_numpy_pickle_aliases()
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return PPO.load(model_path, env=env, custom_objects=custom_objects)


def make_eval_settings(route_file, seed, num_seconds, observation_class):
    settings = SIM_SETTINGS.copy()
    settings.pop("max_green", None)
    settings.pop("enforce_max_green", None)
    settings["route_file"] = route_file
    settings["num_seconds"] = num_seconds
    settings["use_gui"] = False
    settings["observation_class"] = observation_class
    settings.pop("out_csv_name", None)
    command = settings.get("additional_sumo_cmd", "")
    if "--seed" not in command and "--random" not in command:
        settings["additional_sumo_cmd"] = f"{command} --seed {seed}".strip()
    return settings


def maybe_wrap_phase_safety(env, gen_num):
    if gen_num < 11:
        return env
    max_green_steps = max(int(SIM_SETTINGS.get("max_green", 96) / max(SIM_SETTINGS.get("delta_time", 4), 1)), 1)
    return PhaseSafetyWrapper(
        env,
        max_green_steps=max_green_steps,
        enforce=bool(SIM_SETTINGS.get("enforce_max_green", True)),
    )


def evaluate_baseline(scenario):
    env = SumoEnvironment(
        **make_eval_settings(
            scenario["route_file"],
            scenario["seed"],
            scenario["num_seconds"],
            LegacyRadarObservation,
        )
    )
    if scenario["chaos_prob"] > 0:
        env = ChaosMonkeyWrapper(env, chaos_prob=scenario["chaos_prob"], seed=scenario["seed"])

    obs, info = env.reset()
    done = False
    step_counter = 0
    phase_switch_count = 0
    previous_action = None
    while not done:
        action = (step_counter // 8) % env.action_space.n
        if previous_action is not None and action != previous_action:
            phase_switch_count += 1
        previous_action = action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_counter += 1

    df = pd.DataFrame(env.unwrapped.metrics)
    df.attrs["phase_switch_count"] = phase_switch_count
    print_eval_status("Baseline", df)
    env.close()
    return df


def evaluate_model(model_path, gen_num, scenario):
    observation_class = get_observation_class(gen_num)
    settings = make_eval_settings(
        scenario["route_file"],
        scenario["seed"],
        scenario["num_seconds"],
        observation_class,
    )
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"

    if gen_num >= 10 and os.path.exists(vecnormalize_path):
        raw_env = DummyVecEnv(
            [
                lambda: MetricsSnapshotWrapper(
                    ChaosMonkeyWrapper(
                        maybe_wrap_phase_safety(SumoEnvironment(**settings), gen_num),
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
        phase_switch_count = 0
        previous_action = None
        terminal_metrics = None
        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            action_value = int(action[0]) if hasattr(action, "__len__") else int(action)
            if previous_action is not None and action_value != previous_action:
                phase_switch_count += 1
            previous_action = action_value
            obs, reward, done, info = env.step(action)
            if done[0] and info and "terminal_metrics" in info[0]:
                terminal_metrics = info[0]["terminal_metrics"]

        df = pd.DataFrame(terminal_metrics if terminal_metrics is not None else env.venv.envs[0].unwrapped.metrics)
        df.attrs["phase_switch_count"] = phase_switch_count
        print_eval_status(f"Gen {gen_num}", df)
        env.close()
        return df

    env = maybe_wrap_phase_safety(SumoEnvironment(**settings), gen_num)
    if scenario["chaos_prob"] > 0:
        env = ChaosMonkeyWrapper(env, chaos_prob=scenario["chaos_prob"], seed=scenario["seed"])

    model = safe_ppo_load(model_path, env)
    obs, info = env.reset()
    done = False
    phase_switch_count = 0
    previous_action = None
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action_value = int(action)
        if previous_action is not None and action_value != previous_action:
            phase_switch_count += 1
        previous_action = action_value
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    df = pd.DataFrame(env.unwrapped.metrics)
    df.attrs["phase_switch_count"] = phase_switch_count
    print_eval_status(f"Gen {gen_num}", df)
    env.close()
    return df


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

    return {
        "agent": label,
        "scenario": scenario_name,
        "final_total_wait": df["system_total_waiting_time"].iloc[-1],
        "mean_total_wait": df["system_total_waiting_time"].mean(),
        "p95_total_wait": df["system_total_waiting_time"].quantile(0.95),
        "max_stopped": df["system_total_stopped"].max(),
        "stopped_auc": df["system_total_stopped"].sum(),
        "mean_speed": df["system_mean_speed"].mean(),
        "phase_switch_count": df.attrs.get("phase_switch_count", 0),
    }


def plot_metrics(df_dict, scenario_name):
    plt.figure(figsize=(16, 14))
    metrics = {
        "system_total_waiting_time": ("Total waiting time", "lower is better"),
        "system_total_stopped": ("Stopped vehicles", "lower is better"),
        "system_mean_speed": ("Mean speed", "higher is better"),
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
                ax.plot(hours, smoothed_values, label=label, linewidth=2.5, alpha=0.9)

        ax.set_title(f"{title} | {subtitle}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Virtual time (hours)", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
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
        ax.plot(df["step"] / 3600, normalized, label=label, linewidth=2.5, alpha=0.9)

    ax.set_title("Normalized congestion index | lower is better", fontsize=14, fontweight="bold")
    ax.set_xlabel("Virtual time (hours)", fontsize=12)
    ax.set_ylabel("0..1 per agent", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS["comparison_reports"], exist_ok=True)
    report_path = os.path.join(OUTPUT_DIRS["comparison_reports"], f"comparison_{scenario_name}.png")
    plt.savefig(report_path, dpi=300)
    print(f"Report saved: {report_path}")


def build_scenarios(full=False, max_daily=2, max_stress=2, daily_seconds=14400, stress_seconds=14400):
    base_seed = EVAL_SETTINGS["seed"]
    holdout_routes = discover_holdout_routes()
    scenarios = []
    daily_count = 0
    stress_count = 0

    for index, route_file in enumerate(holdout_routes):
        name = os.path.splitext(os.path.basename(route_file))[0]
        is_stress = "stress" in name
        if not full:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and PPO generations on holdout SUMO routes.")
    parser.add_argument("--full", action="store_true", help="Evaluate every holdout route with full scenario durations.")
    parser.add_argument("--models", type=int, default=2, help="How many latest generations to compare. Default: 2.")
    parser.add_argument("--max-daily", type=int, default=2, help="Quick mode daily route count.")
    parser.add_argument("--max-stress", type=int, default=2, help="Quick mode stress route count.")
    parser.add_argument("--daily-seconds", type=int, default=14400, help="Daily scenario duration in quick mode.")
    parser.add_argument("--stress-seconds", type=int, default=14400, help="Stress scenario duration.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-scenario PNG plots and only write CSV summary.")
    return parser.parse_args()


if __name__ == "__main__":
    ensure_output_dirs()
    args = parse_args()
    model_dir = "models"
    base_name = TRAIN_SETTINGS["model_name"]
    top_models = get_top_models(model_dir, base_name, count=args.models)
    scenarios = build_scenarios(
        full=args.full,
        max_daily=args.max_daily,
        max_stress=args.max_stress,
        daily_seconds=86400 if args.full else args.daily_seconds,
        stress_seconds=args.stress_seconds,
    )
    summary_rows = []

    agent_count = 1 + len(top_models)
    total_runs = len(scenarios) * agent_count
    mode = "FULL" if args.full else "QUICK"
    print(f"{mode} comparison: {len(scenarios)} scenarios x {agent_count} agents = {total_runs} SUMO runs")
    print(f"Models: {[f'Gen {gen}' for gen, _ in reversed(top_models)]}")

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']} ({scenario['num_seconds']} sim seconds)")
        results = {"Baseline": evaluate_baseline(scenario)}
        summary_rows.append(summarize_metrics("Baseline", scenario["name"], results["Baseline"]))

        for gen_num, model_path in reversed(top_models):
            label = f"Gen {gen_num}"
            print(f"Evaluating {label}: {model_path}")
            results[label] = evaluate_model(model_path, gen_num, scenario)
            summary_rows.append(summarize_metrics(label, scenario["name"], results[label]))

        if not args.no_plots:
            plot_metrics(results, scenario["name"])

    summary = pd.DataFrame([row for row in summary_rows if row])
    suffix = "full" if args.full else "quick"
    summary_path = os.path.join(OUTPUT_DIRS["holdout_reports"], f"holdout_eval_summary_{suffix}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nHoldout summary saved: {summary_path}")
    print(summary)
