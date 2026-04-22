import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from config import SIM_SETTINGS
from custom_obs import RadarObservation

def evaluate_model(model_path, agent_name, is_baseline=False):
    print(f"\n🚦 Запуск симуляции: {agent_name}...")
    
    # Настраиваем среду (без GUI, чтобы прогнать 24 часа за пару минут)
    eval_settings = SIM_SETTINGS.copy()
    eval_settings['use_gui'] = False 
    eval_settings['observation_class'] = RadarObservation
    
    # Задаем имя файла, куда SUMO будет писать метрики
    csv_prefix = f'outputs/eval_{agent_name}'
    eval_settings['out_csv_name'] = csv_prefix
    
    env = SumoEnvironment(**eval_settings)
    obs, info = env.reset()
    done = False
    
    # Загружаем нейросеть, если это не базовый алгоритм
    if not is_baseline:
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Ошибка: Модель {model_path}.zip не найдена!")
            env.close()
            return None
        model = PPO.load(model_path, env=env)
    
    step_counter = 0
    while not done:
        if not is_baseline:
            # Умный ИИ принимает решение
            action, _ = model.predict(obs, deterministic=True)
        else:
            # Базовый алгоритм: переключаем зеленую фазу каждые 8 шагов (32 секунды)
            action = (step_counter // 8) % env.action_space.n
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_counter += 1

    env.close()
    
    # Находим сгенерированный CSV файл
    csv_files = glob.glob(f"{csv_prefix}*.csv")
    if not csv_files:
        return None
        
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"✅ Данные {agent_name} сохранены в: {latest_csv}")
    return latest_csv

def plot_metrics(csv_dict):
    print("\n📊 Отрисовка графиков...")
    plt.figure(figsize=(16, 12))
    
    # Метрики, которые мы хотим сравнить (названия колонок из CSV от sumo-rl)
    metrics = {
        'system_total_waiting_time': ('Общее время ожидания (секунды)', 'Меньше = Лучше'),
        'system_total_stopped': ('Количество стоящих машин (Очередь)', 'Меньше = Лучше'),
        'system_mean_speed': ('Средняя скорость потока (м/с)', 'Больше = Лучше')
    }
    
    colors = {'Baseline (Таймер)': '#e74c3c', 'Gen 7 (Справедливость)': '#3498db', 'Gen 8 (Скорость + CO2)': '#2ecc71'}
    
    for i, (col, (title, subtitle)) in enumerate(metrics.items(), 1):
        ax = plt.subplot(3, 1, i)
        
        for label, csv_file in csv_dict.items():
            if csv_file:
                df = pd.read_csv(csv_file)
                # Переводим шаги в виртуальные часы для красоты оси X
                hours = df['step'] * SIM_SETTINGS['delta_time'] / 3600
                ax.plot(hours, df[col], label=label, color=colors.get(label, '#333333'), linewidth=2)
        
        ax.set_title(f"{title} | {subtitle}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Виртуальное время (Часы)', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_xlim(0, 24)
        
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    report_path = 'outputs/models_comparison_report.png'
    plt.savefig(report_path, dpi=300)
    print(f"🎉 Отчет успешно сохранен в '{report_path}'")
    plt.show()

if __name__ == '__main__':
    # 1. Запускаем тесты
    csv_baseline = evaluate_model(None, "Baseline (Таймер)", is_baseline=True)
    csv_gen7 = evaluate_model("models/ppo_traffic_model_Gen7", "Gen 7 (Справедливость)", is_baseline=False)
    csv_gen8 = evaluate_model("models/ppo_traffic_model_Gen8", "Gen 8 (Скорость + CO2)", is_baseline=False)
    
    # 2. Собираем пути к файлам и рисуем графики
    results = {
        "Baseline (Таймер)": csv_baseline,
        "Gen 7 (Справедливость)": csv_gen7,
        "Gen 8 (Скорость + CO2)": csv_gen8
    }
    
    plot_metrics(results)