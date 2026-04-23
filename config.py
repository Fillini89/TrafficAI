# config.py

def balanced_reward(traffic_signal):
    # --- 1. СБОР ДАННЫХ (КНУТЫ) ---
    queue = traffic_signal.get_total_queued()
    
    wait_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    # Считаем сумму для общего фона
    total_wait_time = sum(wait_times) if isinstance(wait_times, (list, tuple)) else sum(wait_times.values())
    
    # NEW: Находим сумму ожиданий по полосам
    max_wait_time = max(wait_times) if wait_times else 0 
    
    pressure = traffic_signal.get_pressure()

    # --- 2. СБОР ДАННЫХ (ПРЯНИК И ЭКОЛОГИЯ) ---
    vehicles = []
    for lane in traffic_signal.lanes:
        vehicles.extend(traffic_signal.sumo.lane.getLastStepVehicleIDs(lane))
    
    if vehicles:
        speeds = [traffic_signal.sumo.vehicle.getSpeed(v) for v in vehicles]
        avg_speed = sum(speeds) / len(speeds)
        co2_emissions = sum([traffic_signal.sumo.vehicle.getCO2Emission(v) for v in vehicles])
    else:
        avg_speed = 0.0
        co2_emissions = 0.0

    # --- 3. НОРМАЛИЗАЦИЯ И КАЛИБРОВКА ---
    penalty_queue = queue * 0.5 
    penalty_pressure = abs(pressure) * 0.5 
    penalty_wait = total_wait_time / 100.0 
    penalty_co2 = co2_emissions / 10000.0

    # ИСПРАВЛЕНИЕ: Даем дополнительный "линейный", а не "экспоненциальный" 
    # вес самой загруженной полосе. Это сфокусирует ИИ на разгребании худшего затора, 
    # но не сведет его с ума гигантскими числами.
    penalty_worst_lane = max_wait_time / 50.0 

    # ПРЯНИКИ
    bonus_speed = avg_speed * 2.0 
    arrived_cars = traffic_signal.sumo.simulation.getArrivedNumber()
    bonus_throughput = arrived_cars * 10.0 

    # --- 4. ФИНАЛЬНЫЙ БАЛАНС ---
    reward = (bonus_speed + bonus_throughput) - (penalty_queue + penalty_pressure + penalty_wait + penalty_worst_lane + penalty_co2)

    # --- 5. КРИТИЧЕСКИЕ ШТРАФЫ ---
    try:
        collisions = traffic_signal.sumo.simulation.getCollidingVehiclesNumber()
        emergency_stops = traffic_signal.sumo.simulation.getEmergencyStoppingVehiclesNumber()
        # NEW: Считываем количество телепортированных из-за затора машин
        teleports = traffic_signal.sumo.simulation.getStartingTeleportNumber()

        if emergency_stops > 0:
            reward -= 15 * emergency_stops 
        if collisions > 0:
            reward -= 200 * collisions      
        if teleports > 0:
            reward -= 50 * teleports # Жесткий штраф за мертвые пробки
            
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