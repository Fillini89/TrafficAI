import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_obs import RadarObservation
from chaos_wrapper import ChaosMonkeyWrapper
import os
import glob
import re
import time
import random
from config import SIM_SETTINGS, TRAIN_SETTINGS

class SyncBreakerWrapper(gym.Wrapper):
    """Artificially staggers requests to Windows TCP ports to avoid collisions (Address already in use)"""
    def __init__(self, env, env_id):
        super().__init__(env)
        self.env_id = env_id

    def reset(self, **kwargs):
        # Каждый из 12 процессов будет ждать свою долю секунды + легкий рандом
        time.sleep(self.env_id * 0.3 + random.uniform(0.1, 0.5))
        return self.env.reset(**kwargs)

def make_env(env_id):
    def _init():
        env = SumoEnvironment(
            **SIM_SETTINGS,
            observation_class=RadarObservation,
            use_gui=False,
            out_csv_name=f'outputs/train_parallel_{env_id}' 
        )
        env = ChaosMonkeyWrapper(env)
        env = SyncBreakerWrapper(env, env_id)
        env = Monitor(env) 
        return env
    return _init

def get_latest_checkpoint(folder='./checkpoints/'):
    """Находит самый свежий архив и извлекает количество пройденных шагов."""
    files = glob.glob(os.path.join(folder, '*.zip'))
    if not files:
        return None, 0
    
    latest_file = max(files, key=os.path.getctime)
    
    # Ищем шаги в названии файла (например, _120000_steps.zip)
    match = re.search(r'_(\d+)_steps\.zip', latest_file)
    completed_steps = int(match.group(1)) if match else 0
    
    return latest_file, completed_steps

def get_generation_info(model_dir, base_name):
    """Looking for the old model generation in the folder."""
    os.makedirs(model_dir, exist_ok=True)
    files = glob.glob(os.path.join(model_dir, f"{base_name}_Gen*.zip"))
    highest_gen = 0
    
    for f in files:
        match = re.search(rf"{base_name}_Gen(\d+)\.zip", f)
        if match:
            gen = int(match.group(1))
            if gen > highest_gen:
                highest_gen = gen
                
    return highest_gen

if __name__ == '__main__':
    num_cpu = TRAIN_SETTINGS['num_cpu']
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    base_name = TRAIN_SETTINGS['model_name']
    model_dir = 'models'
    
    highest_gen = get_generation_info(model_dir, base_name)
    
    if highest_gen == 0:
        current_gen = 1
        base_model_path = None
    else:
        current_gen = highest_gen + 1
        base_model_path = os.path.join(model_dir, f"{base_name}_Gen{highest_gen}")
        
    save_model_path = os.path.join(model_dir, f"{base_name}_Gen{current_gen}")

    # 👈 Умная загрузка модели с учетом пройденных шагов
    latest_checkpoint, completed_steps = get_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"🔄 Autosave detected! Continuing training with: {latest_checkpoint}")
        print(f"📈 Resuming from step {completed_steps}...")
        model = PPO.load(latest_checkpoint, env=vec_env)
    elif base_model_path and os.path.exists(base_model_path + ".zip"):
        print(f"🧠 Transfer Learning: Loading base model '{base_model_path}'...")
        model = PPO.load(base_model_path, env=vec_env)
        completed_steps = 0
    else:
        print(f"✨ Creating a new neural network from scratch (Gen 1)...")
        model = PPO("MlpPolicy", vec_env, verbose=1)
        completed_steps = 0

    # 👈 МАТЕМАТИКА МАРАФОНА
    total_target_steps = TRAIN_SETTINGS['total_timesteps']
    remaining_steps = total_target_steps - completed_steps

    if remaining_steps <= 0:
        print(f"✅ Model has already reached or exceeded the target of {total_target_steps} steps!")
        model.save(save_model_path)
        vec_env.close()
        exit()

    os.makedirs('./checkpoints/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix=f'{base_name}_autosave_Gen{current_gen}' 
    )

    print(f"🚀 Training Generation {current_gen} started! Target: {total_target_steps} (Remaining: {remaining_steps})")
    
    # Синхронизируем внутренние часы модели
    model.num_timesteps = completed_steps

    # reset_num_timesteps=False критически важно, чтобы логгер продолжал счет, а не начинал с 0
    model.learn(
        total_timesteps=remaining_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=False 
    )

    model.save(save_model_path)
    print(f"💾 Model saved as '{save_model_path}.zip'.")

    vec_env.close()