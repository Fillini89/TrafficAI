# config.py

def balanced_reward(traffic_signal):
    # --- 1. СБОР СЫРЫХ ДАННЫХ (R&D ФАЗА) ---
    
    # Справедливость (Accumulated Waiting Time)
    # Получаем время ожидания по всем входящим полосам и суммируем
    wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    total_wait_time = sum(wait_times) if isinstance(wait_times, (list, tuple)) else sum(wait_times.values())
    
    # Стратегия (Pressure)
    # Встроенная метрика: количество машин въезжающих МИНУС выезжающих
    pressure = traffic_signal.get_pressure()
    
    # Статус-кво (Длина очереди)
    queue = traffic_signal.get_total_queued()
    
    # --- 2. ВРЕМЕННЫЙ ЛОГГЕР ---
    # Чтобы не засорять терминал (у нас 12 потоков и шаг в 4 секунды), 
    # выводим статистику только 1 раз в виртуальный час.
    sim_time = traffic_signal.sumo.simulation.getTime()
    if int(sim_time) % 3600 == 0 and int(sim_time) > 0:
        hour = int(sim_time // 3600)
        print(f"📊 [Hour {hour:02d}] R&D Metrics | Queue: {queue:02d} | Wait Time: {total_wait_time:06.1f}s | Pressure: {pressure:05.2f}")

    # --- 3. ФИЗИКА ШТРАФОВ (ПОКА ОСТАВЛЯЕМ СТАРУЮ) ---
    reward = -(queue * 3)

    try:
        collisions = traffic_signal.sumo.simulation.getCollidingVehiclesNumber()
        emergency_stops = traffic_signal.sumo.simulation.getEmergencyStoppingVehiclesNumber()

        if emergency_stops > 0:
            reward -= 15 * emergency_stops 
            
        if collisions > 0:
            reward -= 200 * collisions      
            
    except Exception:
        pass

    return reward

# --- НАСТРОЙКИ ЖЕЛЕЗА И ОБУЧЕНИЯ ---
TRAIN_SETTINGS = {
    'num_cpu': 12,                
    'total_timesteps': 3000000,
    # Указываем только базовое имя. Скрипт сам добавит _GenX
    'model_name': 'ppo_traffic_model'  
}

# --- НАСТРОЙКИ СИМУЛЯТОРА SUMO ---
SIM_SETTINGS = {
    'net_file': 'SumoNetwork01.net.xml',
    'route_file': 'routes.rou.xml',
    'num_seconds': 86400,
    'min_green': 10,
    'yellow_time': 3,
    'delta_time': 4,
    'single_agent': True,
    'time_to_teleport': -1,
    'additional_sumo_cmd': '--no-step-log --device.rerouting.probability 1.0',
    'reward_fn': balanced_reward            
}

# --- ИЗМЕНЕНИЕ 4: СЛОВАРЬ МАРШРУТОВ ДЛЯ НОВОГО ГЕНЕРАТОРА ---
TRAFFIC_ROUTES = {
    "f_0": {"from": "-31272#6", "to": "-31272#7"},
    "f_1": {"from": "-30892#16", "to": "-31272#7"},
    "f_2": {"from": "-31272#6", "to": "--30892#16"},
    "f_3": {"from": "--31272#7", "to": "--30892#16"},
    "f_4": {"from": "--31272#7", "to": "--31272#6"},
    "f_5": {"from": "-30892#16", "to": "-30892#17"},
    "f_6": {"from": "--31272#7", "to": "-30892#17"},
    "f_7": {"from": "--30892#17", "to": "--31272#6"},
    "f_8": {"from": "-31272#6", "to": "--30892#16"},
    "f_9": {"from": "-30892#16", "to": "-31272#7"},
    "f_10": {"from": "--31272#7", "to": "-30892#17"},
    "f_11": {"from": "-30892#16", "to": "--31272#6"},
    "f_12": {"from": "-31272#6", "to": "-30892#17"},
    "f_13": {"from": "--30892#17", "to": "-31272#7"}
}