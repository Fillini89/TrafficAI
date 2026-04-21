import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from config import SIM_SETTINGS, TRAIN_SETTINGS
from custom_obs import RadarObservation
from chaos_wrapper import ChaosMonkeyWrapper
import os, glob, re

def get_latest_model(model_dir, base_name):
    files = glob.glob(os.path.join(model_dir, f"{base_name}_Gen*.zip"))
    highest_gen = 0
    latest_model = ""
    for f in files:
        match = re.search(rf"{base_name}_Gen(\d+)\.zip", f)
        if match:
            gen = int(match.group(1))
            if gen > highest_gen:
                highest_gen = gen
                latest_model = f.replace('.zip', '')
    return latest_model

def run_visual_test():
    print("🚦 Initialization of environment with GUI...")
    
    test_settings = SIM_SETTINGS.copy()
    test_settings['use_gui'] = True
    test_settings['observation_class'] = RadarObservation
    
    env = SumoEnvironment(**test_settings)
    env = ChaosMonkeyWrapper(env, chaos_prob=0.005)
    
    base_name = TRAIN_SETTINGS['model_name']
    latest_model_path = get_latest_model("models", base_name)
    
    if not latest_model_path:
        print("❌ No Gen models found in 'models/' directory!")
        return
        
    print(f"🧠 Loading latest model: '{latest_model_path}'...")
    model = PPO.load(latest_model_path, env=env)
    
    obs, info = env.reset()
    done = False
    
    print("✅ Environment ready!")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print("🏁 Simulation done!")

if __name__ == '__main__':
    run_visual_test()