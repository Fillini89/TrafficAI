import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob
from config import SIM_SETTINGS, TRAIN_SETTINGS

def make_env(env_id):
    def _init():
        env = SumoEnvironment(
            **SIM_SETTINGS,
            use_gui=False,
            out_csv_name=f'outputs/train_parallel_{env_id}' 
        )
        env = Monitor(env) 
        return env
    return _init

def get_latest_checkpoint(folder='./checkpoints/'):
    """Находит самый свежий .zip архив в папке автосохранений."""
    files = glob.glob(os.path.join(folder, '*.zip'))
    if not files:
        return None
    return max(files, key=os.path.getctime) # Возвращает файл с последней датой создания

if __name__ == '__main__':
    num_cpu = TRAIN_SETTINGS['num_cpu']
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 👈 Умная загрузка модели
    latest_checkpoint = get_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"🔄 Обнаружено автосохранение! Продолжаем обучение с: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=vec_env)
    else:
        print(f"Загрузка базовых знаний из модели '{TRAIN_SETTINGS['base_model']}'...")
        model = PPO.load(TRAIN_SETTINGS['base_model'], env=vec_env)

    os.makedirs('./checkpoints/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='ppo_autosave'
    )

    print("🧠 Тренировка запущена...")
    model.learn(
        total_timesteps=TRAIN_SETTINGS['total_timesteps'],
        callback=checkpoint_callback 
    )

    save_path = TRAIN_SETTINGS['save_model']
    model.save(save_path)
    print(f"💾 Модель сохранена как '{save_path}.zip'.")

    vec_env.close()