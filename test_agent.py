import glob
import os
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment

from chaos_wrapper import ChaosMonkeyWrapper, PhaseSafetyWrapper
from config import SIM_SETTINGS, TRAIN_SETTINGS
from custom_obs import LegacyRadarObservation, RadarObservation
from sb3_compat import install_numpy_pickle_aliases


def get_latest_model(model_dir, base_name):
    files = glob.glob(os.path.join(model_dir, f"{base_name}_Gen*.zip"))
    highest_gen = 0
    latest_model = ""
    for file_path in files:
        match = re.search(rf"{base_name}_Gen(\d+)\.zip", file_path)
        if match:
            gen = int(match.group(1))
            if gen > highest_gen:
                highest_gen = gen
                latest_model = file_path.replace(".zip", "")
    return highest_gen, latest_model


def make_gui_env(gen_num):
    settings = SIM_SETTINGS.copy()
    max_green = settings.pop("max_green", 96)
    enforce_max_green = settings.pop("enforce_max_green", True)
    settings["use_gui"] = True
    settings["observation_class"] = RadarObservation if gen_num >= 10 else LegacyRadarObservation
    env = SumoEnvironment(**settings)
    if gen_num >= 11:
        max_green_steps = max(int(max_green / max(settings.get("delta_time", 4), 1)), 1)
        env = PhaseSafetyWrapper(env, max_green_steps=max_green_steps, enforce=bool(enforce_max_green))
    return ChaosMonkeyWrapper(env, chaos_prob=0.005)


def safe_ppo_load(model_path, env):
    install_numpy_pickle_aliases()
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return PPO.load(model_path, env=env, custom_objects=custom_objects)


def run_visual_test():
    print("Initializing GUI environment...")
    base_name = TRAIN_SETTINGS["model_name"]
    gen_num, latest_model_path = get_latest_model("models", base_name)

    if not latest_model_path:
        print("No Gen models found in 'models/' directory.")
        return

    print(f"Loading latest model: Gen {gen_num} '{latest_model_path}'...")
    vecnormalize_path = f"{latest_model_path}_vecnormalize.pkl"

    if gen_num >= 10 and os.path.exists(vecnormalize_path):
        env = DummyVecEnv([lambda: make_gui_env(gen_num)])
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        model = safe_ppo_load(latest_model_path, env)

        obs = env.reset()
        done = [False]
        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        env.close()
    else:
        env = make_gui_env(gen_num)
        model = safe_ppo_load(latest_model_path, env)

        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()

    print("Simulation done.")


if __name__ == "__main__":
    run_visual_test()
