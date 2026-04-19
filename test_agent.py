import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from config import SIM_SETTINGS

def run_visual_test():
    print("🚦 Инициализация среды с включенным GUI...")
    
    # Копируем настройки и принудительно включаем графический интерфейс
    test_settings = SIM_SETTINGS.copy()
    test_settings['use_gui'] = True
    
    env = SumoEnvironment(**test_settings)
    
    print("🧠 Загрузка финальной модели 'ppo_traffic_model_prod'...")
    # Загружаем нашу готовую модель
    model = PPO.load("ppo_traffic_model_prod", env=env)
    
    obs, info = env.reset()
    done = False
    
    print("✅ Среда готова. Откройте окно SUMO GUI и нажмите кнопку Play (Треугольник)!")
    
    while not done:
        # deterministic=True заставляет ИИ выбирать строго лучшее действие, без "исследовательского" рандома
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print("🏁 Симуляция завершена.")

if __name__ == '__main__':
    run_visual_test()