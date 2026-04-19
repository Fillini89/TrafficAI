import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from config import SIM_SETTINGS
from custom_obs import RadarObservation
from chaos_wrapper import ChaosMonkeyWrapper

def run_visual_test():
    print("🚦 Initialization of environment with GUI...")
    
    test_settings = SIM_SETTINGS.copy()
    test_settings['use_gui'] = True
    test_settings['observation_class'] = RadarObservation
    
    env = SumoEnvironment(**test_settings)
    env = ChaosMonkeyWrapper(env, chaos_prob=0.005)
    
    print("🧠 Loading Phase 3 model 'ppo_traffic_model_phase3'...")
    model = PPO.load("models/ppo_traffic_model_phase3", env=env)
    
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