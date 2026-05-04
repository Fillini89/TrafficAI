import gymnasium as gym
from sumo_rl import SumoEnvironment
from config import DEFAULT_ROUTE_FILE, OUTPUT_DIRS, ensure_output_dirs

ensure_output_dirs()

# 1. Инициализируем среду
env = SumoEnvironment(
    net_file='SumoNetwork01.net.xml',
    route_file=DEFAULT_ROUTE_FILE,
    out_csv_name=f'{OUTPUT_DIRS["sumo_misc"]}/random_agent',
    use_gui=True,
    num_seconds=3600,
    min_green=5,
    single_agent=True  # 👈 КРИТИЧЕСКИ ВАЖНЫЙ ПАРАМЕТР ДЛЯ ОДНОГО ПЕРЕКРЕСТКА
)

# В режиме single_agent=True среда строго следует стандарту Gymnasium
obs, info = env.reset()
done = False

print("✅ Симуляция успешно запущена! Python управляет светофором '-12408'.")

# 2. Главный цикл
while not done:
    # Выбираем случайную фазу
    action = env.action_space.sample()
    
    # Делаем шаг. Теперь метод step() корректно возвращает 5 значений
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Симуляция заканчивается, если достигнут лимит времени (truncated) или иное условие (terminated)
    done = terminated or truncated

env.close()
print("🛑 Симуляция завершена.")
