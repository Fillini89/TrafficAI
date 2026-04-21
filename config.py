# config.py

def balanced_reward(traffic_signal):
    queue = traffic_signal.get_total_queued()
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