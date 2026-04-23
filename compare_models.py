import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from config import SIM_SETTINGS, TRAIN_SETTINGS
from custom_obs import RadarObservation

def get_top_models(model_dir, base_name, count=2):
    """Автоматически находит последние N поколений моделей."""
    pattern = re.compile(rf"{base_name}_Gen(\d+)\.zip")
    models = []
    
    if not os.path.exists(model_dir):
        return []

    for f in os.listdir(model_dir):
        match = pattern.search(f)
        if match:
            gen_num = int(match.group(1))
            full_path = os.path.join(model_dir, f.replace('.zip', ''))
            models.append((gen_num, full_path))
    
    # Сортируем по номеру поколения (от новых к старым)
    models.sort(key=lambda x: x[0], reverse=True)
    return models[:count]

def evaluate_model(model_path, agent_name, is_baseline=False):
    print(f"\n🚦 Запуск симуляции: {agent_name}...")
    
    eval_settings = SIM_SETTINGS.copy()
    eval_settings['use_gui'] = False 
    eval_settings['observation_class'] = RadarObservation
    
    if 'out_csv_name' in eval_settings:
        del eval_settings['out_csv_name']
    
    env = SumoEnvironment(**eval_settings)
    obs, info = env.reset()
    
    if not is_baseline:
        model = PPO.load(model_path, env=env)
    
    step_counter = 0
    while True:
        if not is_baseline:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = (step_counter // 8) % env.action_space.n
            
        obs, reward, terminated, truncated, info = env.step(action)
        step_counter += 1
        
        if truncated:
            break

    df = pd.DataFrame(env.unwrapped.metrics)
    env.close()
    
    print(f"✅ Данные {agent_name} успешно собраны.")
    return df

def plot_metrics(df_dict):
    print("\n📊 Отрисовка графиков (с фильтрацией шума)...")
    plt.figure(figsize=(16, 12))
    
    metrics = {
        'system_total_waiting_time': ('Общее время ожидания (сек)', 'Меньше = Лучше'),
        'system_total_stopped': ('Длина очереди (машин)', 'Меньше = Лучше'),
        'system_mean_speed': ('Средняя скорость (м/с)', 'Больше = Лучше')
    }
    
    # Настройка сглаживания: окно в 15 минут (15 * 60 / 4 сек = 225 шагов)
    # Это уберет "пики" и оставит чистые линии тренда
    smoothing_window = 225 

    for i, (col, (title, subtitle)) in enumerate(metrics.items(), 1):
        ax = plt.subplot(3, 1, i)
        
        for label, df in df_dict.items():
            if df is not None and not df.empty:
                hours = df['step'] / 3600
                # Применяем скользящее среднее для подавления шума
                smoothed_values = df[col].rolling(window=smoothing_window, min_periods=1).mean()
                
                ax.plot(hours, smoothed_values, label=label, linewidth=2.5, alpha=0.9)
        
        ax.set_title(f"{title} | {subtitle}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Виртуальное время (Часы)', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_xlim(0, 24)
        
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/final_comparison_report.png', dpi=300)
    print(f"🎉 Отчет сохранен: outputs/final_comparison_report.png")
    plt.show()

if __name__ == '__main__':
    results = {}

    # 1. Сначала считаем Baseline
    results["Baseline (Таймер)"] = evaluate_model(None, "Baseline", is_baseline=True)

    # 2. Автоматически находим две последние модели
    model_dir = 'models'
    base_name = TRAIN_SETTINGS['model_name']
    top_models = get_top_models(model_dir, base_name, count=2)

    for gen_num, model_path in reversed(top_models):
        label = f"Gen {gen_num} (ИИ)"
        results[label] = evaluate_model(model_path, label)
    
    # 3. Визуализируем
    plot_metrics(results)