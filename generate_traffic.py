import math
import random
from config import TRAFFIC_ROUTES

# Настройки генерации
SIMULATION_STEPS = 86400  # 24 часа
INTERVAL = 900            # 15 минут (каждые 15 минут меняется плотность потока)

# Типы транспорта (Смешанный трафик)
V_TYPES = """
    <vType id="car" length="5.0" maxSpeed="15.0" accel="2.6" decel="4.5" sigma="0.5" probability="0.8"/>
    <vType id="truck" length="12.0" maxSpeed="10.0" accel="1.2" decel="2.5" sigma="0.7" probability="0.1"/>
    <vType id="bus" length="15.0" maxSpeed="12.0" accel="1.2" decel="3.0" sigma="0.6" probability="0.08"/>
    <vType id="moto" length="2.0" maxSpeed="20.0" accel="4.0" decel="6.0" sigma="0.5" probability="0.02"/>
"""

def calculate_wave_probability(time_sec, is_morning_route, is_evening_route):
    base_prob = 0.01  # Ночной фоновый трафик
    hour = time_sec / 3600.0

    # Утренний пик (около 08:30)
    morning_peak = math.sin((hour - 6) / 5 * math.pi) * 0.25 if 6 <= hour <= 11 and is_morning_route else 0
    # Вечерний пик (около 18:30)
    evening_peak = math.sin((hour - 16) / 5 * math.pi) * 0.30 if 16 <= hour <= 21 and is_evening_route else 0
    # Дневной средний фон (с 11 до 16)
    day_bg = 0.06 if 11 < hour < 16 else 0

    # Шум (±10%), имитируя неравномерность
    noise = random.uniform(0.9, 1.1)
    
    final_prob = (base_prob + morning_peak + evening_peak + day_bg) * noise
    return max(0.005, min(final_prob, 0.4))

def generate_routes():
    with open("routes.rou.xml", "w", encoding="utf-8") as routes:
        routes.write('<?xml version="1.0" ?>\n')
        routes.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        routes.write(V_TYPES)

        for step in range(0, SIMULATION_STEPS, INTERVAL):
            begin = step
            end = step + INTERVAL
            hour_display = f"{int(begin//3600):02d}:{(int(begin%3600)//60):02d}"
            
            routes.write(f'    \n')

            for route_id, data in TRAFFIC_ROUTES.items():
                is_morning = route_id == "f_4"
                is_evening = route_id in ["f_11", "f_12"]
                prob = calculate_wave_probability(begin, is_morning, is_evening)
                
                routes.write(
                    f'    <flow id="{route_id}_{begin}" begin="{begin}" end="{end}" '
                    f'probability="{prob:.4f}" from="{data["from"]}" to="{data["to"]}"/>\n'
                )

        routes.write("</routes>\n")
    print("✅ Файл routes.rou.xml успешно сгенерирован (Суточные волны + Смешанный трафик)!")

if __name__ == "__main__":
    generate_routes()