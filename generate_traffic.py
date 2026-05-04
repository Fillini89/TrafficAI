import math
import os
import random

from config import HOLDOUT_ROUTE_DIR, TRAIN_ROUTE_DIR, TRAFFIC_ROUTES


SIMULATION_STEPS = 86400
INTERVAL = 900
DAILY_PROFILE_COUNT = 12

V_TYPES = """
    <vTypeDistribution id="mixed_traffic">
        <vType id="car" length="5.0" maxSpeed="15.0" accel="2.6" decel="4.5" sigma="0.5" probability="0.8" guiShape="passenger"/>
        <vType id="truck" length="12.0" maxSpeed="10.0" accel="1.2" decel="2.5" sigma="0.7" probability="0.1" guiShape="truck"/>
        <vType id="bus" length="15.0" maxSpeed="12.0" accel="1.2" decel="3.0" sigma="0.6" probability="0.08" guiShape="bus"/>
        <vType id="moto" length="2.0" maxSpeed="20.0" accel="4.0" decel="6.0" sigma="0.5" probability="0.02" guiShape="motorcycle"/>
    </vTypeDistribution>
"""


def calculate_wave_probability(time_sec, route_id, rng, demand_scale, morning_shift, evening_shift):
    hour = time_sec / 3600.0
    base_prob = rng.uniform(0.006, 0.016)
    day_bg = rng.uniform(0.035, 0.085) if 11 < hour < 16 else 0.0

    morning_routes = {"f_4", "f_7", "f_13"}
    evening_routes = {"f_10", "f_11", "f_12"}
    morning_peak = 0.0
    evening_peak = 0.0

    morning_start = 6.0 + morning_shift
    morning_end = 11.0 + morning_shift
    if morning_start <= hour <= morning_end and route_id in morning_routes:
        morning_peak = math.sin((hour - morning_start) / 5.0 * math.pi) * rng.uniform(0.16, 0.31)

    evening_start = 16.0 + evening_shift
    evening_end = 21.0 + evening_shift
    if evening_start <= hour <= evening_end and route_id in evening_routes:
        evening_peak = math.sin((hour - evening_start) / 5.0 * math.pi) * rng.uniform(0.18, 0.35)

    local_noise = rng.uniform(0.85, 1.2)
    final_prob = (base_prob + day_bg + morning_peak + evening_peak) * local_noise * demand_scale
    return max(0.002, min(final_prob, 0.45))


def generate_daily_routefile(filename, seed, demand_scale=1.0):
    rng = random.Random(seed)
    morning_shift = rng.uniform(-0.75, 0.75)
    evening_shift = rng.uniform(-0.75, 0.75)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as routes:
        routes.write('<?xml version="1.0" ?>\n')
        routes.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        routes.write(V_TYPES)

        for step in range(0, SIMULATION_STEPS, INTERVAL):
            begin = step
            end = step + INTERVAL
            routes.write("\n")

            for route_id, data in TRAFFIC_ROUTES.items():
                prob = calculate_wave_probability(begin, route_id, rng, demand_scale, morning_shift, evening_shift)
                routes.write(
                    f'    <flow id="{route_id}_{begin}" begin="{begin}" end="{end}" '
                    f'probability="{prob:.4f}" type="mixed_traffic" from="{data["from"]}" to="{data["to"]}"/>\n'
                )

        routes.write("</routes>\n")


def generate_route_pool(count=DAILY_PROFILE_COUNT, output_dir=TRAIN_ROUTE_DIR, seed_start=10_000):
    demand_scales = [0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35, 1.5]
    generated = []

    for index in range(count):
        seed = seed_start + index * 137
        demand_scale = demand_scales[index % len(demand_scales)]
        filename = os.path.join(output_dir, f"daily_seed{seed}_demand{int(demand_scale * 100):03d}.rou.xml")
        generate_daily_routefile(filename, seed=seed, demand_scale=demand_scale)
        generated.append(filename)

    print(f"Generated {len(generated)} daily route files in '{output_dir}'.")
    return generated


def generate_routes():
    train_routes = generate_route_pool(count=8, output_dir=TRAIN_ROUTE_DIR, seed_start=10_000)
    holdout_routes = generate_route_pool(count=4, output_dir=HOLDOUT_ROUTE_DIR, seed_start=30_000)
    return train_routes + holdout_routes


if __name__ == "__main__":
    generate_routes()
