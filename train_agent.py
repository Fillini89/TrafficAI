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
from config import SIM_SETTINGS, TRAIN_SETTINGS

def make_env(env_id):
    def _init():
        env = SumoEnvironment(
            **SIM_SETTINGS,
            observation_class=RadarObservation,
            use_gui=False,
            out_csv_name=f'outputs/train_parallel_{env_id}' 
        )
        env = ChaosMonkeyWrapper(env)
        env = Monitor(env) 
        return env
    return _init

def get_latest_checkpoint(folder='./checkpoints/'):
    """Находит самый свежий .zip архив в папке автосохранений."""
    files = glob.glob(os.path.join(folder, '*.zip'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

if __name__ == '__main__':
    num_cpu = TRAIN_SETTINGS['num_cpu']
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 👈 Умная загрузка модели
    latest_checkpoint = get_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"🔄 Autosave detected! Continuing training with: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=vec_env)
    else:
        # 👈 КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Создаем новую нейросеть под новые фазы светофора
        print(f"✨ Создание новой нейросети с расширенным Action Space (Защищенные повороты)...")
        model = PPO("MlpPolicy", vec_env, verbose=1)

    os.makedirs('./checkpoints/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='ppo_autosave_phase4' # 👈 Обновляем префикс
    )

    print("🧠 Training started!")
    model.learn(
        total_timesteps=TRAIN_SETTINGS['total_timesteps'],
        callback=checkpoint_callback 
    )

    save_path = TRAIN_SETTINGS['save_model']
    model.save(save_path)
    print(f"💾 Model saved as '{save_path}.zip'.")

    vec_env.close()