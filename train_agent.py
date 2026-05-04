import glob
import os
import random
import re
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from sumo_rl import SumoEnvironment

from chaos_wrapper import ChaosMonkeyWrapper, PhaseSafetyWrapper, RewardInfoWrapper
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
    os.makedirs(os.path.dirname(out_csv_name), exist_ok=True)
    env = SumoEnvironment(
        **env_settings,
        observation_class=RadarObservation,
        use_gui=False,
        out_csv_name=out_csv_name,
    )
    env = RewardInfoWrapper(env)
    max_green_steps = max(int(max_green / max(env_settings.get("delta_time", 4), 1)), 1)
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


def get_latest_checkpoint(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.zip"))
    if not files:
        return None, 0

    latest_file = max(files, key=os.path.getctime)
    match = re.search(r"_(\d+)_steps\.zip", latest_file)
    completed_steps = int(match.group(1)) if match else 0
    return latest_file, completed_steps


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
    smoke_mode = os.getenv("TRAFFICAI_SMOKE_TEST") == "1"
    num_cpu = TRAIN_SETTINGS["num_cpu"]
    raw_vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    base_name = f"{TRAIN_SETTINGS['model_name']}_smoke" if smoke_mode else TRAIN_SETTINGS["model_name"]
    vecnormalize_path = (
        "checkpoints/vecnormalize_smoke_latest.pkl" if smoke_mode else TRAIN_SETTINGS["vecnormalize_path"]
    )
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    highest_gen = get_generation_info(model_dir, base_name)
    current_gen = highest_gen + 1 if highest_gen else 1
    save_model_path = os.path.join(model_dir, f"{base_name}_Gen{current_gen}")
    checkpoint_prefix = f"{base_name}_autosave_Gen{current_gen}"

    latest_checkpoint, completed_steps = get_latest_checkpoint("checkpoints", checkpoint_prefix)
    warm_start_model = None
    warm_start_vecnormalize = None
    warm_start_enabled = (
        TRAIN_SETTINGS.get("warm_start_latest_model")
        and os.getenv("TRAFFICAI_DISABLE_WARM_START") != "1"
    )
    if not latest_checkpoint and warm_start_enabled and highest_gen:
        warm_start_model, warm_start_vecnormalize = model_artifact_paths(model_dir, base_name, highest_gen)
        if not os.path.exists(f"{warm_start_model}.zip"):
            warm_start_model = None
        if not os.path.exists(warm_start_vecnormalize):
            warm_start_vecnormalize = None

    vec_env = wrap_vec_normalize(
        raw_vec_env,
        vecnormalize_path if latest_checkpoint else warm_start_vecnormalize,
        training=True,
    )

    if latest_checkpoint:
        print(f"Autosave detected for Gen {current_gen}. Continuing training with: {latest_checkpoint}")
        print(f"Resuming from step {completed_steps}...")
        model = PPO.load(latest_checkpoint, env=vec_env)
    elif warm_start_model:
        print(f"Warm-starting Gen {current_gen} from Gen {highest_gen}: {warm_start_model}.zip")
        if warm_start_vecnormalize:
            print(f"Loaded VecNormalize stats from: {warm_start_vecnormalize}")
        model = PPO.load(warm_start_model, env=vec_env)
        completed_steps = 0
    else:
        print(f"Creating Gen {current_gen} from scratch with current observation/reward stack.")
        model = PPO("MlpPolicy", vec_env, verbose=1, **build_ppo_kwargs())
        completed_steps = 0

    total_target_steps = TRAIN_SETTINGS["smoke_timesteps"] if smoke_mode else TRAIN_SETTINGS["total_timesteps"]
    remaining_steps = total_target_steps - completed_steps

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
