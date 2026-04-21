import math
import random
from config import TRAFFIC_ROUTES

def generate_stress_routefile():
    total_seconds = 14400  # 4 часа (4 * 3600)
    filename = "routes_stress.rou.xml"
    
    # --- НАСТРОЙКИ СИЛЫ СТРЕССА ---
    old_min_prob = 0.01
    new_min_prob = old_min_prob * 2  # 0.02 (Стартовая и конечная нагрузка)
    max_prob = 0.15                  # Пиковая нагрузка 
    
    amplitude = max_prob - new_min_prob

    # Здесь дистрибуция не нужна, так как мы явно назначаем type каждому <vehicle>
    v_types_xml = """
    <vType id="car" length="5.0" maxSpeed="15.0" accel="2.6" decel="4.5" sigma="0.5" guiShape="passenger"/>
    <vType id="truck" length="12.0" maxSpeed="10.0" accel="1.2" decel="2.5" sigma="0.7" guiShape="truck"/>
    <vType id="bus" length="15.0" maxSpeed="12.0" accel="1.2" decel="3.0" sigma="0.6" guiShape="bus"/>
    <vType id="moto" length="2.0" maxSpeed="20.0" accel="4.0" decel="6.0" sigma="0.5" guiShape="motorcycle"/>
"""

    with open(filename, "w") as routes:
        routes.write('<?xml version="1.0"?>\n')
        routes.write('<routes>\n')
        
        # Записываем XML-блок с физикой и графикой
        routes.write(v_types_xml)

        # Подтягиваем словарь маршрутов напрямую из config.py
        for route_id, data in TRAFFIC_ROUTES.items():
            routes.write(f'    <route id="{route_id}" edges="{data["from"]} {data["to"]}"/>\n')

        routes.write('\n')
        veh_nr = 0

        # Вероятности спавна для стресс-теста (75% легковушки, 15% фуры, 8% автобусы, 2% мото)
        vehicle_types = ["car", "truck", "bus", "moto"]
        vehicle_weights = [0.75, 0.15, 0.08, 0.02]

        # Генерация 4-часового цикла
        for step in range(total_seconds):
            # Математика волны
            wave = math.sin(math.pi * (step / total_seconds))
            current_prob = new_min_prob + (amplitude * wave)

            # Прогоняем вероятность по каждому маршруту
            for route_id in TRAFFIC_ROUTES.keys():
                if random.uniform(0, 1) < current_prob:
                    # Выбираем тип транспорта средствами Python
                    vtype = random.choices(vehicle_types, weights=vehicle_weights, k=1)[0]
                    # Явно передаем type="{vtype}"
                    routes.write(f'    <vehicle id="{route_id}_{veh_nr}" type="{vtype}" route="{route_id}" depart="{step}"/>\n')
                    veh_nr += 1

        routes.write('</routes>\n')
        print(f"🔥 Stress test route file '{filename}' generated with full vehicle GUI shapes!")
        print(f"🚗 Total vehicles: {veh_nr} over 4 hours.")

if __name__ == "__main__":
    generate_stress_routefile()