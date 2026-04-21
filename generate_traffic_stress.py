import math
import random
from config import TRAFFIC_ROUTES

def generate_stress_routefile():
    total_seconds = 14400  # 4 часа (4 * 3600)
    filename = "routes_stress.rou.xml"
    
    # --- НАСТРОЙКИ СИЛЫ СТРЕССА ---
    # По вашим условиям: минимум увеличен в 2 раза, пик оставлен прежним
    old_min_prob = 0.01
    new_min_prob = old_min_prob * 2  # 0.02 (Стартовая и конечная нагрузка)
    max_prob = 0.15                  # Пиковая нагрузка (оставляем как было)
    
    amplitude = max_prob - new_min_prob

    with open(filename, "w") as routes:
        routes.write('<?xml version="1.0"?>\n')
        routes.write('<routes>\n')
        
        # Физика автомобилей (оставляем тяжелые фуры для создания заторов)
        routes.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
        routes.write('    <vType id="truck" accel="1.2" decel="2.5" sigma="0.5" length="12.0" minGap="3.0" maxSpeed="11.11" guiShape="truck"/>\n\n')

        # Подтягиваем словарь маршрутов напрямую из config.py
        for route_id, data in TRAFFIC_ROUTES.items():
            routes.write(f'    <route id="{route_id}" edges="{data["from"]} {data["to"]}"/>\n')

        routes.write('\n')
        veh_nr = 0

        # Генерация 4-часового цикла
        for step in range(total_seconds):
            # Математика волны: sin(pi * x) дает идеальный холм от 0 до 1, с пиком ровно на 2-м часе
            wave = math.sin(math.pi * (step / total_seconds))
            
            # Текущая вероятность генерации машины в эту конкретную секунду
            current_prob = new_min_prob + (amplitude * wave)

            # Прогоняем вероятность по каждому маршруту
            for route_id in TRAFFIC_ROUTES.keys():
                if random.uniform(0, 1) < current_prob:
                    # 15% шанс заспавнить фуру вместо легковушки для дополнительного физического стресса
                    vtype = "truck" if random.uniform(0, 1) < 0.15 else "car"
                    routes.write(f'    <vehicle id="{route_id}_{veh_nr}" type="{vtype}" route="{route_id}" depart="{step}"/>\n')
                    veh_nr += 1

        routes.write('</routes>\n')
        print(f"🔥 Stress test route file '{filename}' generated!")
        print(f"🚗 Total vehicles: {veh_nr} over 4 hours.")

if __name__ == "__main__":
    generate_stress_routefile()