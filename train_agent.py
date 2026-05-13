import glob
import os
import random
import re
import time

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from sumo_rl import SumoEnvironment

from chaos_wrapper import ChaosMonkeyWrapper, PhaseSafetyWrapper, RewardInfoWrapper, ServiceDebtGuardrailWrapper
from sb3_compat import load_ppo_compat
from config import (
    CURRICULUM_SETTINGS,
    EVAL_SETTINGS,
    OUTPUT_DIRS,
    PPO_SETTINGS,
    SIM_SETTINGS,
    TRAIN_SETTINGS,
    VEC_NORMALIZE_SETTINGS,
    ensure_output_dirs,
)
from custom_obs import RadarObservation


class SyncBreakerWrapper(gym.Wrapper):
    """Stagger Windows TCP port requests to reduce SUMO/TraCI collisions."""

    def __init__(self, env, env_id):
        super().__init__(env)
        self.env_id = env_id

    def reset(self, **kwargs):
        time.sleep(self.env_id * 0.3 + random.uniform(0.1, 0.5))
        return self.env.reset(**kwargs)


class DynamicRouteEnv(gym.Env):
    """Recreate SUMO with a new train route file on every episode."""

    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id
        self.episode_index = 0
        self.env = None
        self.current_route_file = None
        self.current_seed = None
        self.current_chaos_prob = None
        time.sleep(env_id * 0.3 + random.uniform(0.1, 0.5))
        self._rebuild_env()

    def reset(self, **kwargs):
        if self.episode_index > 0:
            time.sleep(self.env_id * 0.3 + random.uniform(0.1, 0.5))
            self._rebuild_env()
        self.episode_index += 1
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["route_file"] = self.current_route_file
        info["route_seed"] = self.current_seed
        info["chaos_prob"] = self.current_chaos_prob
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["route_file"] = self.current_route_file
        info["route_seed"] = self.current_seed
        info["chaos_prob"] = self.current_chaos_prob
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:
                print(f"SUMO env close warning ignored: {exc}")
            self.env = None

    def _rebuild_env(self):
        self.close()
        env_settings, chaos_prob, seed = build_training_episode_settings(self.env_id, self.episode_index)
        out_csv_name = os.path.join(OUTPUT_DIRS["sumo_train"], f"train_parallel_{self.env_id}_episode_{self.episode_index}")
        self.env = create_sumo_env(env_settings, chaos_prob, seed, out_csv_name=out_csv_name)
        self.current_route_file = env_settings["route_file"]
        self.current_seed = seed
        self.current_chaos_prob = chaos_prob
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


class RewardComponentCallback(BaseCallback):
    """Log shaped reward components into SB3/TensorBoard."""

    def _on_step(self):
        infos = self.locals.get("infos", [])
        reward_components = [info.get("reward_components") for info in infos if info.get("reward_components")]
        if reward_components:
            keys = reward_components[0].keys()
            for key in keys:
                values = [float(components[key]) for components in reward_components if key in components]
                if values:
                    self.logger.record(f"reward/{key}", sum(values) / len(values))

        chaos_started = sum(1 for info in infos if info.get("chaos_incident_started"))
        if chaos_started:
            self.logger.record("chaos/incidents_started", chaos_started)
        forced_switches = sum(1 for info in infos if info.get("phase_forced_switch"))
        if forced_switches:
            self.logger.record("safety/phase_forced_switches", forced_switches)
        fairness_forced_switches = sum(1 for info in infos if info.get("fairness_forced_switch"))
        if fairness_forced_switches:
            self.logger.record("safety/fairness_forced_switches", fairness_forced_switches)
        actual_switches = sum(1 for info in infos if info.get("actual_phase_changed"))
        if actual_switches:
            self.logger.record("safety/actual_phase_switches", actual_switches)
        cadence_suppressed = sum(1 for info in infos if info.get("cadence_suppressed_switch"))
        if cadence_suppressed:
            self.logger.record("safety/cadence_suppressed_switches", cadence_suppressed)
        suppressed_switches = sum(1 for info in infos if info.get("fairness_guardrail_suppressed_switch"))
        if suppressed_switches:
            self.logger.record("safety/fairness_guardrail_suppressed_switches", suppressed_switches)
        hold_active = sum(1 for info in infos if info.get("fairness_guardrail_hold_active"))
        if hold_active:
            self.logger.record("safety/fairness_guardrail_hold_active", hold_active)
        adaptive_releases = sum(1 for info in infos if info.get("adaptive_cadence_release"))
        if adaptive_releases:
            self.logger.record("safety/adaptive_cadence_releases", adaptive_releases)
        requested_service_debts = [
            float(info.get("requested_service_debt", 0.0)) for info in infos if "requested_service_debt" in info
        ]
        if requested_service_debts:
            self.logger.record(
                "safety/requested_service_debt",
                sum(requested_service_debts) / len(requested_service_debts),
            )
        requested_queue_advantages = [
            float(info.get("requested_queue_advantage", 0.0)) for info in infos if "requested_queue_advantage" in info
        ]
        if requested_queue_advantages:
            self.logger.record(
                "safety/requested_queue_advantage",
                sum(requested_queue_advantages) / len(requested_queue_advantages),
            )
        service_debts = [float(info.get("max_service_debt", 0.0)) for info in infos if "max_service_debt" in info]
        if service_debts:
            self.logger.record("safety/max_service_debt", sum(service_debts) / len(service_debts))
        service_ages = [float(info.get("max_service_age", 0.0)) for info in infos if "max_service_age" in info]
        if service_ages:
            self.logger.record("safety/max_service_age", sum(service_ages) / len(service_ages))
        warning_excess = [
            float(info.get("service_age_warning_excess", 0.0))
            for info in infos
            if "service_age_warning_excess" in info
        ]
        if warning_excess:
            self.logger.record("safety/service_age_warning_excess", sum(warning_excess) / len(warning_excess))
        critical_excess = [
            float(info.get("service_age_critical_excess", 0.0))
            for info in infos
            if "service_age_critical_excess" in info
        ]
        if critical_excess:
            self.logger.record("safety/service_age_critical_excess", sum(critical_excess) / len(critical_excess))
        budget_penalties = [
            float(info.get("service_age_budget_penalty", 0.0))
            for info in infos
            if "service_age_budget_penalty" in info
        ]
        if budget_penalties:
            self.logger.record("safety/service_age_budget_penalty", sum(budget_penalties) / len(budget_penalties))
        return True


class VecNormalizeSaveCallback(BaseCallback):
    """Persist VecNormalize statistics alongside model checkpoints."""

    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = max(int(save_freq), 1)
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self._save_stats()
        return True

    def _on_training_end(self):
        self._save_stats()

    def _save_stats(self):
        env = self.model.get_vec_normalize_env()
        if env is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            env.save(self.save_path)


def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def make_env(env_id, eval_mode=False):
    def _init():
        if CURRICULUM_SETTINGS["enabled"] and CURRICULUM_SETTINGS["route_per_episode"] and not eval_mode:
            env = DynamicRouteEnv(env_id)
        else:
            env_settings, chaos_prob, seed = build_env_settings(env_id, eval_mode=eval_mode)
            env = create_sumo_env(
                env_settings,
                chaos_prob,
                seed,
                out_csv_name=os.path.join(
                    OUTPUT_DIRS["sumo_eval" if eval_mode else "sumo_train"],
                    f"{'eval' if eval_mode else 'train'}_parallel_{env_id}",
                ),
            )
            if not eval_mode:
                env = SyncBreakerWrapper(env, env_id)
        env = Monitor(env)
        return env

    return _init


def create_sumo_env(env_settings, chaos_prob, seed, out_csv_name):
    env_settings = env_settings.copy()
    max_green = env_settings.pop("max_green", 96)
    enforce_max_green = env_settings.pop("enforce_max_green", True)
    service_debt_metric = env_settings.pop("service_debt_metric", "service_age")
    service_debt_soft_threshold = env_settings.pop("service_debt_soft_threshold", 90.0)
    service_debt_threshold = env_settings.pop("service_debt_threshold", 120.0)
    service_debt_min_hold = env_settings.pop("service_debt_min_hold", 24.0)
    service_debt_target_hold = env_settings.pop("service_debt_target_hold", 32.0)
    service_debt_protected_hold = env_settings.pop("service_debt_protected_hold", 24.0)
    service_debt_adaptive_threshold = env_settings.pop("service_debt_adaptive_threshold", 75.0)
    service_debt_queue_imbalance_threshold = env_settings.pop("service_debt_queue_imbalance_threshold", 6.0)
    service_debt_worse_multiplier = env_settings.pop("service_debt_worse_multiplier", 1.25)
    service_debt_worse_threshold = env_settings.pop("service_debt_worse_threshold", None)
    cadence_suppression_penalty = env_settings.pop("cadence_suppression_penalty", 0.0)
    service_age_warning_threshold = env_settings.pop("service_age_warning_threshold", 150.0)
    service_age_critical_threshold = env_settings.pop("service_age_critical_threshold", 210.0)
    service_age_warning_norm = env_settings.pop("service_age_warning_norm", 60.0)
    service_age_critical_norm = env_settings.pop("service_age_critical_norm", 60.0)
    service_age_warning_penalty_weight = env_settings.pop("service_age_warning_penalty_weight", 0.0)
    service_age_critical_penalty_weight = env_settings.pop("service_age_critical_penalty_weight", 0.0)
    service_age_budget_penalty_clip = env_settings.pop("service_age_budget_penalty_clip", 0.0)
    service_age_critical_override = env_settings.pop("service_age_critical_override", False)
    enforce_service_debt = env_settings.pop("enforce_service_debt", True)
    os.makedirs(os.path.dirname(out_csv_name), exist_ok=True)
    env = SumoEnvironment(
        **env_settings,
        observation_class=RadarObservation,
        use_gui=False,
        out_csv_name=out_csv_name,
    )
    env = RewardInfoWrapper(env)
    delta_time = max(env_settings.get("delta_time", 4), 1)
    min_green_steps = max(int(env_settings.get("min_green", 10) / delta_time), 1)
    min_hold_steps = max(int(float(service_debt_min_hold) / delta_time), 0)
    target_hold_steps = max(int(float(service_debt_target_hold) / delta_time), 0)
    protected_hold_steps = max(int(float(service_debt_protected_hold) / delta_time), 0)
    env = ServiceDebtGuardrailWrapper(
        env,
        service_threshold_seconds=service_debt_threshold,
        soft_service_threshold_seconds=service_debt_soft_threshold,
        min_green_steps=min_green_steps,
        min_hold_steps=min_hold_steps,
        target_hold_steps=target_hold_steps,
        protected_hold_steps=protected_hold_steps,
        adaptive_service_threshold_seconds=service_debt_adaptive_threshold,
        queue_imbalance_threshold=service_debt_queue_imbalance_threshold,
        worse_debt_multiplier=service_debt_worse_multiplier,
        worse_debt_seconds=service_debt_worse_threshold,
        use_service_age=str(service_debt_metric).lower() == "service_age",
        cadence_suppression_penalty=cadence_suppression_penalty,
        service_age_warning_seconds=service_age_warning_threshold,
        service_age_critical_seconds=service_age_critical_threshold,
        service_age_warning_norm=service_age_warning_norm,
        service_age_critical_norm=service_age_critical_norm,
        service_age_warning_penalty_weight=service_age_warning_penalty_weight,
        service_age_critical_penalty_weight=service_age_critical_penalty_weight,
        service_age_budget_penalty_clip=service_age_budget_penalty_clip,
        service_age_critical_override=service_age_critical_override,
        enforce=bool(enforce_service_debt),
    )
    max_green_steps = max(int(max_green / delta_time), 1)
    env = PhaseSafetyWrapper(
        env,
        max_green_steps=max_green_steps,
        enforce=bool(enforce_max_green),
    )
    env = ChaosMonkeyWrapper(env, chaos_prob=chaos_prob, seed=seed)
    return env


def build_env_settings(env_id, eval_mode=False):
    settings = SIM_SETTINGS.copy()
    settings.pop("observation_class", None)
    settings.pop("use_gui", None)
    settings.pop("out_csv_name", None)

    if eval_mode:
        seed = EVAL_SETTINGS["seed"] + env_id
        settings["route_file"] = EVAL_SETTINGS["route_file"]
        settings["num_seconds"] = EVAL_SETTINGS["num_seconds"]
        chaos_prob = EVAL_SETTINGS["chaos_prob"]
    elif CURRICULUM_SETTINGS["enabled"]:
        route_files = discover_route_pool()
        chaos_probs = CURRICULUM_SETTINGS["chaos_probabilities"]
        demand_scales = CURRICULUM_SETTINGS["demand_scales"]
        seeds = CURRICULUM_SETTINGS["seeds"]
        seed = seeds[env_id % len(seeds)]
        route_rng = random.Random(seed)
        settings["route_file"] = route_rng.choice(route_files)
        settings["num_seconds"] = CURRICULUM_SETTINGS["episode_seconds"]
        chaos_prob = chaos_probs[env_id % len(chaos_probs)]
        settings["additional_sumo_cmd"] = append_sumo_scale(settings.get("additional_sumo_cmd", ""), demand_scales[env_id % len(demand_scales)])
    else:
        seed = 1000 + env_id
        chaos_prob = 0.001

    if os.getenv("TRAFFICAI_DISABLE_CHAOS") == "1":
        chaos_prob = 0.0

    settings["additional_sumo_cmd"] = append_sumo_seed(settings.get("additional_sumo_cmd", ""), seed)
    return settings, chaos_prob, seed


def build_training_episode_settings(env_id, episode_index):
    settings = SIM_SETTINGS.copy()
    settings.pop("observation_class", None)
    settings.pop("use_gui", None)
    settings.pop("out_csv_name", None)

    route_files = discover_route_pool()
    chaos_probs = CURRICULUM_SETTINGS["chaos_probabilities"]
    demand_scales = CURRICULUM_SETTINGS["demand_scales"]
    seeds = CURRICULUM_SETTINGS["seeds"]
    base_seed = seeds[env_id % len(seeds)]
    seed = base_seed + (episode_index * 10_000)
    rng = random.Random(seed)

    settings["route_file"] = rng.choice(route_files)
    episode_seconds = CURRICULUM_SETTINGS["episode_seconds"]
    route_name = os.path.basename(settings["route_file"])
    if route_name.startswith("daily_") and rng.random() < CURRICULUM_SETTINGS.get("long_episode_probability", 0.0):
        episode_seconds = CURRICULUM_SETTINGS.get("long_episode_seconds", episode_seconds)
    settings["num_seconds"] = episode_seconds
    settings["additional_sumo_cmd"] = append_sumo_scale(
        settings.get("additional_sumo_cmd", ""),
        rng.choice(demand_scales),
    )

    chaos_prob = rng.choice(chaos_probs)
    if os.getenv("TRAFFICAI_DISABLE_CHAOS") == "1":
        chaos_prob = 0.0

    settings["additional_sumo_cmd"] = append_sumo_seed(settings.get("additional_sumo_cmd", ""), seed)
    return settings, chaos_prob, seed


def discover_route_pool():
    route_files = []
    route_dir = CURRICULUM_SETTINGS["route_dir"]
    for pattern in CURRICULUM_SETTINGS["route_patterns"]:
        route_files.extend(glob.glob(os.path.join(route_dir, pattern)))
    route_files = sorted(set(route_files))
    if route_files:
        return route_files
    return CURRICULUM_SETTINGS["fallback_route_files"]


def append_sumo_seed(command, seed):
    command = command or ""
    if "--seed" in command or "--random" in command:
        return command
    return f"{command} --seed {seed}".strip()


def append_sumo_scale(command, scale):
    command = command or ""
    if "--scale" in command:
        return command
    return f"{command} --scale {scale}".strip()


def get_latest_checkpoint(folder, base_name):
    files = glob.glob(os.path.join(folder, f"{base_name}_autosave_Gen*_steps.zip"))
    pattern = re.compile(rf"{re.escape(base_name)}_autosave_Gen(\d+)_(\d+)_steps\.zip$")
    candidates = []

    for file_path in files:
        match = pattern.match(os.path.basename(file_path))
        if match:
            candidates.append(
                (
                    os.path.getctime(file_path),
                    file_path,
                    int(match.group(1)),
                    int(match.group(2)),
                )
            )

    if not candidates:
        return None, None, 0

    _, latest_file, checkpoint_gen, completed_steps = max(candidates, key=lambda item: item[0])
    return latest_file, checkpoint_gen, completed_steps


def get_generation_info(model_dir, base_name):
    os.makedirs(model_dir, exist_ok=True)
    files = glob.glob(os.path.join(model_dir, f"{base_name}_Gen*.zip"))
    highest_gen = 0

    for file_path in files:
        match = re.search(rf"{base_name}_Gen(\d+)\.zip", file_path)
        if match:
            highest_gen = max(highest_gen, int(match.group(1)))
    return highest_gen


def model_artifact_paths(model_dir, base_name, generation):
    model_path = os.path.join(model_dir, f"{base_name}_Gen{generation}")
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"
    return model_path, vecnormalize_path


def checkpoint_continuation_enabled():
    return (
        os.getenv("TRAFFICAI_CONTINUE_CHECKPOINT") == "1"
        or bool(TRAIN_SETTINGS.get("continue_latest_checkpoint", False))
    )


def explicit_warm_start_generation():
    value = os.getenv("TRAFFICAI_WARM_START_GEN")
    if not value:
        return None
    try:
        generation = int(value)
    except ValueError:
        print(f"Ignoring invalid TRAFFICAI_WARM_START_GEN={value!r}.")
        return None
    return generation if generation > 0 else None


def build_ppo_kwargs():
    kwargs = {
        "learning_rate": linear_schedule(PPO_SETTINGS["learning_rate_initial"], PPO_SETTINGS["learning_rate_final"]),
        "n_steps": PPO_SETTINGS["n_steps"],
        "batch_size": PPO_SETTINGS["batch_size"],
        "n_epochs": PPO_SETTINGS["n_epochs"],
        "gamma": PPO_SETTINGS["gamma"],
        "gae_lambda": PPO_SETTINGS["gae_lambda"],
        "clip_range": PPO_SETTINGS["clip_range"],
        "ent_coef": PPO_SETTINGS["ent_coef"],
        "vf_coef": PPO_SETTINGS["vf_coef"],
        "max_grad_norm": PPO_SETTINGS["max_grad_norm"],
        "policy_kwargs": PPO_SETTINGS["policy_kwargs"],
    }
    if tensorboard_available():
        kwargs["tensorboard_log"] = TRAIN_SETTINGS["tensorboard_log"]
    else:
        print("TensorBoard is not installed. Continuing without TensorBoard logging.")
    return kwargs


def tensorboard_available():
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


def wrap_vec_normalize(raw_env, stats_path=None, training=True):
    if stats_path and os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, raw_env)
        env.training = training
        env.norm_reward = training
        return env
    return VecNormalize(raw_env, training=training, **VEC_NORMALIZE_SETTINGS)


def close_vec_env_quietly(vec_env):
    try:
        vec_env.close()
    except (BrokenPipeError, EOFError) as exc:
        print(f"VecEnv close warning ignored after interrupt: {exc}")
    except Exception as exc:
        print(f"VecEnv close warning ignored: {exc}")


def configure_torch_threads():
    thread_count = os.getenv("TRAFFICAI_TORCH_NUM_THREADS")
    if not thread_count:
        return

    try:
        thread_count = max(int(thread_count), 1)
    except ValueError:
        print(f"Ignoring invalid TRAFFICAI_TORCH_NUM_THREADS={thread_count!r}.")
        return

    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError as exc:
        print(f"Torch interop thread setting ignored: {exc}")
    print(f"Torch CPU threads per process set to {thread_count}.")


def build_callbacks(current_gen, num_cpu, base_name, vecnormalize_path, smoke_mode=False):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(OUTPUT_DIRS["eval_callback"], exist_ok=True)

    callback_freq = max(10000 // num_cpu, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=callback_freq,
        save_path="checkpoints",
        name_prefix=f"{base_name}_autosave_Gen{current_gen}",
    )

    callbacks = [
        checkpoint_callback,
        VecNormalizeSaveCallback(callback_freq, vecnormalize_path),
        RewardComponentCallback(),
    ]

    if EVAL_SETTINGS["enabled"] and not smoke_mode:
        eval_raw_env = DummyVecEnv([make_env(0, eval_mode=True)])
        eval_env = wrap_vec_normalize(eval_raw_env, training=False)
        eval_env.norm_reward = False
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=f"models/best_Gen{current_gen}",
                log_path=OUTPUT_DIRS["eval_callback"],
                eval_freq=max(EVAL_SETTINGS["eval_freq"] // num_cpu, 1),
                n_eval_episodes=EVAL_SETTINGS["n_eval_episodes"],
                deterministic=True,
                render=False,
            )
        )
    elif smoke_mode:
        print("Smoke mode: skipping holdout EvalCallback to keep the test run short.")

    return CallbackList(callbacks)


def generate_training_report_if_available(run_name):
    try:
        from training_report import generate_training_report

        result = generate_training_report(run_name=run_name)
    except Exception as exc:
        print(f"Training report was not generated: {exc}")
        return

    if result["report_png"]:
        print(f"Training report saved: {result['report_png']}")
    else:
        print("Training report PNG skipped because matplotlib is not installed.")
    print(f"Training summary saved: {result['summary_csv']}")
    print(f"Training summary text saved: {result['summary_txt']}")


if __name__ == "__main__":
    ensure_output_dirs()
    configure_torch_threads()
    smoke_mode = os.getenv("TRAFFICAI_SMOKE_TEST") == "1"
    num_cpu = TRAIN_SETTINGS["num_cpu"]
    vec_env_class = DummyVecEnv if num_cpu == 1 else SubprocVecEnv
    raw_vec_env = vec_env_class([make_env(i) for i in range(num_cpu)])

    base_name = f"{TRAIN_SETTINGS['model_name']}_smoke" if smoke_mode else TRAIN_SETTINGS["model_name"]
    vecnormalize_path = (
        "checkpoints/vecnormalize_smoke_latest.pkl" if smoke_mode else TRAIN_SETTINGS["vecnormalize_path"]
    )
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    highest_gen = get_generation_info(model_dir, base_name)
    latest_checkpoint, checkpoint_gen, completed_steps = get_latest_checkpoint("checkpoints", base_name)
    use_checkpoint = bool(
        latest_checkpoint
        and (checkpoint_gen > highest_gen or checkpoint_continuation_enabled())
    )
    if latest_checkpoint and not use_checkpoint:
        print(
            f"Stale autosave ignored for Gen {checkpoint_gen}: {latest_checkpoint}. "
            "Set TRAFFICAI_CONTINUE_CHECKPOINT=1 to resume it explicitly."
        )

    current_gen = checkpoint_gen if use_checkpoint else (highest_gen + 1 if highest_gen else 1)
    save_model_path = os.path.join(model_dir, f"{base_name}_Gen{current_gen}")
    warm_start_model = None
    warm_start_vecnormalize = None
    warm_start_gen = None
    warm_start_enabled = (
        TRAIN_SETTINGS.get("warm_start_latest_model")
        and os.getenv("TRAFFICAI_DISABLE_WARM_START") != "1"
    )
    if not use_checkpoint and warm_start_enabled and highest_gen:
        requested_warm_start_gen = explicit_warm_start_generation()
        warm_start_gen = requested_warm_start_gen if requested_warm_start_gen is not None else highest_gen
        warm_start_model, warm_start_vecnormalize = model_artifact_paths(model_dir, base_name, warm_start_gen)
        if not os.path.exists(f"{warm_start_model}.zip"):
            print(f"Warm-start Gen {warm_start_gen} model not found: {warm_start_model}.zip")
            warm_start_model = None
        if not os.path.exists(warm_start_vecnormalize):
            print(f"Warm-start Gen {warm_start_gen} VecNormalize not found: {warm_start_vecnormalize}")
            warm_start_vecnormalize = None
        if warm_start_model is None or warm_start_vecnormalize is None:
            warm_start_model = None
            warm_start_vecnormalize = None

    vec_env = wrap_vec_normalize(
        raw_vec_env,
        vecnormalize_path if use_checkpoint else warm_start_vecnormalize,
        training=True,
    )

    if use_checkpoint:
        print(f"Autosave detected for Gen {current_gen}. Continuing training with: {latest_checkpoint}")
        print(f"Resuming from step {completed_steps}...")
        model = load_ppo_compat(latest_checkpoint, env=vec_env, ppo_kwargs=build_ppo_kwargs(), verbose=1)
    elif warm_start_model:
        print(f"Warm-starting Gen {current_gen} from Gen {warm_start_gen}: {warm_start_model}.zip")
        if warm_start_vecnormalize:
            print(f"Loaded VecNormalize stats from: {warm_start_vecnormalize}")
        model = load_ppo_compat(warm_start_model, env=vec_env, ppo_kwargs=build_ppo_kwargs(), verbose=1)
        completed_steps = 0
    else:
        print(f"Creating Gen {current_gen} from scratch with current observation/reward stack.")
        model = PPO("MlpPolicy", vec_env, verbose=1, **build_ppo_kwargs())
        completed_steps = 0

    total_target_steps = TRAIN_SETTINGS["smoke_timesteps"] if smoke_mode else TRAIN_SETTINGS["total_timesteps"]
    remaining_steps = total_target_steps - completed_steps

    if os.getenv("TRAFFICAI_STARTUP_CHECK") == "1":
        print(
            f"Startup check complete for Gen {current_gen}. "
            f"Completed: {completed_steps}, remaining: {remaining_steps}."
        )
        close_vec_env_quietly(vec_env)
        raise SystemExit(0)

    if remaining_steps <= 0:
        print(f"Model has already reached or exceeded the target of {total_target_steps} steps.")
        model.save(save_model_path)
        vec_env.save(f"{save_model_path}_vecnormalize.pkl")
        vec_env.save(vecnormalize_path)
        close_vec_env_quietly(vec_env)
        raise SystemExit(0)

    callbacks = build_callbacks(current_gen, num_cpu, base_name, vecnormalize_path, smoke_mode=smoke_mode)

    mode_label = "SMOKE" if smoke_mode else "FULL"
    tb_run_name = f"{base_name}_Gen{current_gen}" if tensorboard_available() else "PPO"
    print(f"{mode_label} training Generation {current_gen} started. Target: {total_target_steps} (Remaining: {remaining_steps})")
    model.num_timesteps = completed_steps
    interrupted = False
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            reset_num_timesteps=False,
            tb_log_name=tb_run_name,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted by user. Saving current model and report...")
    finally:
        model.save(save_model_path)
        vec_env.save(f"{save_model_path}_vecnormalize.pkl")
        vec_env.save(vecnormalize_path)
        print(f"Model saved as '{save_model_path}.zip'.")
        print(f"VecNormalize stats saved as '{save_model_path}_vecnormalize.pkl'.")
        generate_training_report_if_available(f"{tb_run_name}_0")
        close_vec_env_quietly(vec_env)

    if interrupted:
        raise SystemExit(130)
