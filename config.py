# config.py

def balanced_reward(traffic_signal):
    """
    Сбалансированная система KPI: длинные пробки теперь причиняют сильную "боль",
    а штрафы за безопасность снижены до адекватного уровня.
    """
    # 1. Усиливаем штраф за пробки (умножаем на 3)
    # Теперь ИИ не сможет их игнорировать ради безопасности
    queue = traffic_signal.get_total_queued()
    reward = -(queue * 3)

    try:
        collisions = traffic_signal.sumo.simulation.getCollidingVehiclesNumber()
        emergency_stops = traffic_signal.sumo.simulation.getEmergencyStoppingVehiclesNumber()

        # 2. Снижаем уровень "террора" для нейросети
        if emergency_stops > 0:
            reward -= 15 * emergency_stops  # Было 50. Теперь это просто строгий выговор, а не катастрофа.
            
        if collisions > 0:
            reward -= 200 * collisions      # Было 500. Всё ещё больно, но сопоставимо с большой пробкой.
            
    except Exception:
        pass

    return reward


# --- НАСТРОЙКИ ЖЕЛЕЗА И ОБУЧЕНИЯ ---
TRAIN_SETTINGS = {
    'num_cpu': 12,                
    'total_timesteps': 3000000,   # 🚀 Целевой марафон: 3 миллиона шагов
    'base_model': 'ppo_traffic_model_safe', # Берем перепуганную модель
    'save_model': 'ppo_traffic_model_prod'  # Сохраняем как релизный кандидат (Production)
}

# --- НАСТРОЙКИ СИМУЛЯТОРА SUMO ---
SIM_SETTINGS = {
    'net_file': 'SumoNetwork01.net.xml',
    'route_file': 'routes.rou.xml',
    'num_seconds': 14400,
    'min_green': 15,
    'yellow_time': 4,
    'delta_time': 5,
    'single_agent': True,
    'additional_sumo_cmd': '--no-step-log',
    'reward_fn': balanced_reward            # 👈 Подключаем новую функцию
}

# --- ДИНАМИЧЕСКИЕ ФАЗЫ ТРАФИКА (СНИЖЕННАЯ НАГРУЗКА) ---
TRAFFIC_PHASES = [
    {
        "name": "Утро (Час пик на f_4)", 
        "begin": 0, "end": 3600, 
        # f_4 снижен с 1260 до ~900 машин/час. Фоновый трафик снижен до ~108 машин.
        "probs": {"f_4": 0.25, "f_12": 0.03, "f_11": 0.01, "default": 0.03}
    },
    {
        "name": "День (Средний трафик)", 
        "begin": 3600, "end": 7200, 
        # Фоновый трафик снижен с 288 до ~180 машин/час на каждый маршрут.
        "probs": {"default": 0.05}
    },
    {
        "name": "Вечер (Пробка в обратную сторону f_11 и f_12)", 
        "begin": 7200, "end": 10800, 
        # Главные вечерние потоки снижены с 1260/1080 до ~900/720 машин/час.
        "probs": {"f_4": 0.03, "f_12": 0.25, "f_11": 0.20, "default": 0.03}
    },
    {
        "name": "Ночь (Пустота)", 
        "begin": 10800, "end": 14400, 
        # Ночь стала еще тише: ~18 машин/час на маршрут.
        "probs": {"default": 0.005}
    }
]