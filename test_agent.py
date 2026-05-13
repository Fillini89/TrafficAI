import argparse
import glob
import os
import re

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment

from chaos_wrapper import ChaosMonkeyWrapper
from compare_models import make_eval_settings, wrap_eval_env
from config import DEFAULT_ROUTE_FILE, HOLDOUT_ROUTE_DIR, SIM_SETTINGS, TRAIN_SETTINGS
from custom_obs import LegacyRadarObservation, RadarObservation
from sb3_compat import load_ppo_compat


ROUTE_PRESETS = {
    "default": DEFAULT_ROUTE_FILE,
    "light": os.path.join(HOLDOUT_ROUTE_DIR, "daily_seed30000_demand075.rou.xml"),
    "medium": os.path.join(HOLDOUT_ROUTE_DIR, "daily_seed30274_demand095.rou.xml"),
    "heavy": os.path.join(HOLDOUT_ROUTE_DIR, "daily_seed30411_demand100.rou.xml"),
    "stress": os.path.join(HOLDOUT_ROUTE_DIR, "stress_seed20865_demand155.rou.xml"),
    "extreme": os.path.join(HOLDOUT_ROUTE_DIR, "stress_seed21211_demand190.rou.xml"),
}


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


def get_model_by_gen(model_dir, base_name, gen_num):
    model_path = os.path.join(model_dir, f"{base_name}_Gen{gen_num}")
    if not os.path.exists(f"{model_path}.zip"):
        return ""
    return model_path


def resolve_route(route_name_or_path):
    route_file = ROUTE_PRESETS.get(route_name_or_path, route_name_or_path)
    if not os.path.exists(route_file):
        presets = ", ".join(sorted(ROUTE_PRESETS))
        raise FileNotFoundError(f"Route not found: {route_file}. Presets: {presets}")
    return route_file


def make_gui_env(gen_num, route_file, seconds, seed, control_profile, chaos_prob):
    observation_class = RadarObservation if gen_num >= 10 else LegacyRadarObservation
    if gen_num >= 10:
        settings = make_eval_settings(route_file, seed, seconds, observation_class)
    else:
        settings = SIM_SETTINGS.copy()
        settings["route_file"] = route_file
        settings["num_seconds"] = seconds
        settings["observation_class"] = observation_class
        settings.pop("out_csv_name", None)
    settings["use_gui"] = True
    env = SumoEnvironment(**settings)
    if gen_num >= 10:
        env = wrap_eval_env(env, gen_num, control_profile=control_profile)
    if chaos_prob > 0.0:
        env = ChaosMonkeyWrapper(env, chaos_prob=chaos_prob, seed=seed)
    return env


def safe_ppo_load(model_path, env):
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    return load_ppo_compat(model_path, env=env, device="cpu", custom_objects=custom_objects)


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained TrafficAI model in SUMO GUI.")
    parser.add_argument("--gen", type=int, default=None, help="Model generation to load. Defaults to latest.")
    parser.add_argument(
        "--route",
        default="default",
        help="Route preset or .rou.xml path. Presets: default, light, medium, heavy, stress, extreme.",
    )
    parser.add_argument("--seconds", type=int, default=3600, help="Simulation length to watch.")
    parser.add_argument("--seed", type=int, default=4242, help="SUMO random seed.")
    parser.add_argument("--chaos", type=float, default=0.0, help="Optional chaos probability for visual stress tests.")
    parser.add_argument(
        "--control-profile",
        choices=["generation", "current"],
        default="generation",
        help="Use each generation's trained control stack, or force current wrappers.",
    )
    parser.add_argument("--list-routes", action="store_true", help="Print visual route presets and exit.")
    return parser.parse_args()


def run_visual_test(args=None):
    if args is None:
        args = parse_args()

    if args.list_routes:
        for name, route_file in ROUTE_PRESETS.items():
            print(f"{name}: {route_file}")
        return

    print("Initializing GUI environment...")
    base_name = TRAIN_SETTINGS["model_name"]
    if args.gen is None:
        gen_num, model_path = get_latest_model("models", base_name)
    else:
        gen_num = args.gen
        model_path = get_model_by_gen("models", base_name, gen_num)

    if not model_path:
        print(f"No Gen {gen_num if args.gen is not None else ''} model found in 'models/' directory.")
        return

    route_file = resolve_route(args.route)
    print(f"Loading model: Gen {gen_num} '{model_path}'...")
    print(f"Route: {route_file}")
    print(f"Seconds: {args.seconds}, control profile: {args.control_profile}, chaos: {args.chaos}")
    vecnormalize_path = f"{model_path}_vecnormalize.pkl"

    if gen_num >= 10 and os.path.exists(vecnormalize_path):
        env = DummyVecEnv(
            [lambda: make_gui_env(gen_num, route_file, args.seconds, args.seed, args.control_profile, args.chaos)]
        )
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        model = safe_ppo_load(model_path, env)

        obs = env.reset()
        done = [False]
        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        env.close()
    else:
        env = make_gui_env(gen_num, route_file, args.seconds, args.seed, args.control_profile, args.chaos)
        model = safe_ppo_load(model_path, env)

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
