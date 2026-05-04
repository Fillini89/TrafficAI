import math
import os
import random

from config import HOLDOUT_ROUTE_DIR, TRAIN_ROUTE_DIR, TRAFFIC_ROUTES


STRESS_PROFILE_COUNT = 8

V_TYPES_XML = """
    <vType id="car" length="5.0" maxSpeed="15.0" accel="2.6" decel="4.5" sigma="0.5" guiShape="passenger"/>
    <vType id="truck" length="12.0" maxSpeed="10.0" accel="1.2" decel="2.5" sigma="0.7" guiShape="truck"/>
    <vType id="bus" length="15.0" maxSpeed="12.0" accel="1.2" decel="3.0" sigma="0.6" guiShape="bus"/>
    <vType id="moto" length="2.0" maxSpeed="20.0" accel="4.0" decel="6.0" sigma="0.5" guiShape="motorcycle"/>
"""


def generate_stress_routefile(filename, seed, demand_scale=1.0, total_seconds=14400):
    rng = random.Random(seed)
    min_prob = rng.uniform(0.014, 0.028) * demand_scale
    max_prob = rng.uniform(0.11, 0.19) * demand_scale
    pulse_count = rng.choice([1, 2, 3])
    route_bias = {route_id: rng.uniform(0.75, 1.35) for route_id in TRAFFIC_ROUTES}
    vehicle_types = ["car", "truck", "bus", "moto"]
    vehicle_weights = [0.74, 0.15, 0.09, 0.02]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as routes:
        routes.write('<?xml version="1.0"?>\n')
        routes.write("<routes>\n")
        routes.write(V_TYPES_XML)

        for route_id, data in TRAFFIC_ROUTES.items():
            routes.write(f'    <route id="{route_id}" edges="{data["from"]} {data["to"]}"/>\n')

        routes.write("\n")
        veh_nr = 0

        for step in range(total_seconds):
            wave = math.sin(math.pi * (step / total_seconds))
            pulse = 0.0
            for pulse_index in range(pulse_count):
                center = (pulse_index + 1) * total_seconds / (pulse_count + 1)
                width = total_seconds / rng.uniform(8.0, 14.0)
                pulse += math.exp(-((step - center) ** 2) / (2 * width**2)) * rng.uniform(0.02, 0.06)

            base_prob = min_prob + ((max_prob - min_prob) * wave) + pulse

            for route_id in TRAFFIC_ROUTES:
                route_prob = max(0.001, min(base_prob * route_bias[route_id] * rng.uniform(0.85, 1.15), 0.5))
                if rng.random() < route_prob:
                    vtype = rng.choices(vehicle_types, weights=vehicle_weights, k=1)[0]
                    routes.write(f'    <vehicle id="{route_id}_{veh_nr}" type="{vtype}" route="{route_id}" depart="{step}"/>\n')
                    veh_nr += 1

        routes.write("</routes>\n")

    return veh_nr


def generate_stress_pool(count=STRESS_PROFILE_COUNT, output_dir=TRAIN_ROUTE_DIR, seed_start=20_000):
    demand_scales = [0.85, 1.0, 1.1, 1.25, 1.4, 1.55, 1.7, 1.9]
    generated = []

    for index in range(count):
        seed = seed_start + index * 173
        demand_scale = demand_scales[index % len(demand_scales)]
        filename = os.path.join(output_dir, f"stress_seed{seed}_demand{int(demand_scale * 100):03d}.rou.xml")
        vehicle_count = generate_stress_routefile(filename, seed=seed, demand_scale=demand_scale)
        generated.append((filename, vehicle_count))

    print(f"Generated {len(generated)} stress route files in '{output_dir}'.")
    for filename, vehicle_count in generated:
        print(f"{filename}: {vehicle_count} vehicles")
    return generated


if __name__ == "__main__":
    generate_stress_pool(count=5, output_dir=TRAIN_ROUTE_DIR, seed_start=20_000)
    generate_stress_pool(count=3, output_dir=HOLDOUT_ROUTE_DIR, seed_start=40_000)
