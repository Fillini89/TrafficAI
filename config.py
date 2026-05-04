import os


ROUTE_DIR = "routes"
TRAIN_ROUTE_DIR = os.path.join(ROUTE_DIR, "train")
HOLDOUT_ROUTE_DIR = os.path.join(ROUTE_DIR, "holdout")
LEGACY_ROUTE_DIR = os.path.join(ROUTE_DIR, "legacy")
DEFAULT_ROUTE_FILE = os.path.join(TRAIN_ROUTE_DIR, "daily_seed10000_demand075.rou.xml")
DEFAULT_STRESS_ROUTE_FILE = os.path.join(HOLDOUT_ROUTE_DIR, "stress_seed40000_demand085.rou.xml")

OUTPUT_DIR = "outputs"
OUTPUT_DIRS = {
    "tensorboard": os.path.join(OUTPUT_DIR, "tensorboard"),
    "training_reports": os.path.join(OUTPUT_DIR, "reports", "training"),
    "comparison_reports": os.path.join(OUTPUT_DIR, "reports", "comparisons"),
    "holdout_reports": os.path.join(OUTPUT_DIR, "reports", "holdout"),
    "sumo_train": os.path.join(OUTPUT_DIR, "runtime", "sumo", "train"),
    "sumo_eval": os.path.join(OUTPUT_DIR, "runtime", "sumo", "eval"),
    "sumo_compare": os.path.join(OUTPUT_DIR, "runtime", "sumo", "compare"),
    "sumo_misc": os.path.join(OUTPUT_DIR, "runtime", "sumo", "misc"),
    "eval_callback": os.path.join(OUTPUT_DIR, "runtime", "eval_callback"),
}


def ensure_output_dirs():
    for path in OUTPUT_DIRS.values():
        os.makedirs(path, exist_ok=True)


def _clip(value, low, high):
    return max(low, min(float(value), high))


def _safe_call(func, default=0.0):
    try:
        return func()
    except Exception:
        return default


def _wait_values(wait_times):
    if isinstance(wait_times, dict):
        return list(wait_times.values())
    if wait_times is None:
        return []
    return list(wait_times)


def balanced_reward(traffic_signal):
    """Gen 11-compatible reward focused on flow plus long-horizon fairness."""

    weights = REWARD_WEIGHTS
    queue = float(_safe_call(traffic_signal.get_total_queued))
    wait_values = _wait_values(_safe_call(traffic_signal.get_accumulated_waiting_time_per_lane, []))
    total_wait_time = float(sum(wait_values))
    max_wait_time = float(max(wait_values) if wait_values else 0.0)
    pressure = float(_safe_call(traffic_signal.get_pressure))

    vehicles = []
    for lane_id in traffic_signal.lanes:
        vehicles.extend(_safe_call(lambda lane_id=lane_id: traffic_signal.sumo.lane.getLastStepVehicleIDs(lane_id), []))

    if vehicles:
        speeds = [_safe_call(lambda vehicle_id=vehicle_id: traffic_signal.sumo.vehicle.getSpeed(vehicle_id)) for vehicle_id in vehicles]
        co2_emissions = sum(
            _safe_call(lambda vehicle_id=vehicle_id: traffic_signal.sumo.vehicle.getCO2Emission(vehicle_id))
            for vehicle_id in vehicles
        )
        avg_speed = sum(speeds) / len(speeds)
        stopped_ratio = queue / max(len(vehicles), 1)
    else:
        avg_speed = 0.0
        co2_emissions = 0.0
        stopped_ratio = 0.0

    sim_time = float(_safe_call(traffic_signal.sumo.simulation.getTime))
    state = getattr(traffic_signal, "_gen10_reward_state", None)
    phase_id = _safe_call(lambda: _traffic_light_phase(traffic_signal), None)
    if state is None or sim_time < state.get("sim_time", 0.0):
        state = {
            "queue": queue,
            "wait": total_wait_time,
            "gridlock_seconds": 0.0,
            "phase_id": phase_id,
            "phase_since": sim_time,
            "sim_time": sim_time,
        }

    dt = max(sim_time - state.get("sim_time", sim_time), 0.0)
    delta_queue = state.get("queue", queue) - queue
    delta_wait = state.get("wait", total_wait_time) - total_wait_time

    gridlock_active = (
        queue >= REWARD_LIMITS["gridlock_queue_threshold"]
        and stopped_ratio >= REWARD_LIMITS["gridlock_stopped_ratio"]
    )
    gridlock_seconds = state.get("gridlock_seconds", 0.0) + dt if gridlock_active else 0.0
    if phase_id != state.get("phase_id"):
        phase_since = sim_time
    else:
        phase_since = state.get("phase_since", sim_time)
    phase_hold_seconds = max(0.0, sim_time - phase_since)

    starvation_threshold = REWARD_LIMITS["starvation_wait_threshold"]
    starved_lanes = sum(1 for value in wait_values if value >= starvation_threshold)
    starvation_excess = sum(max(value - starvation_threshold, 0.0) for value in wait_values)
    long_green_active = (
        phase_hold_seconds >= REWARD_LIMITS["max_green_soft"]
        and (starved_lanes > 0 or queue >= REWARD_LIMITS["long_green_queue_threshold"])
    )
    long_green_excess = max(phase_hold_seconds - REWARD_LIMITS["max_green_soft"], 0.0) if long_green_active else 0.0

    arrived_cars = float(_safe_call(traffic_signal.sumo.simulation.getArrivedNumber))
    collisions = float(_safe_call(traffic_signal.sumo.simulation.getCollidingVehiclesNumber))
    emergency_stops = float(_safe_call(traffic_signal.sumo.simulation.getEmergencyStoppingVehiclesNumber))

    components = {
        "speed": weights["speed"] * _clip(avg_speed / REWARD_LIMITS["speed_norm"], 0.0, 1.0),
        "throughput": weights["throughput"] * _clip(arrived_cars / REWARD_LIMITS["throughput_norm"], 0.0, 2.0),
        "delta_queue": weights["delta_queue"] * _clip(delta_queue / REWARD_LIMITS["delta_queue_norm"], -1.0, 1.0),
        "delta_wait": weights["delta_wait"] * _clip(delta_wait / REWARD_LIMITS["delta_wait_norm"], -1.0, 1.0),
        "queue": -weights["queue"] * _clip(queue / REWARD_LIMITS["queue_norm"], 0.0, 2.0),
        "pressure": -weights["pressure"] * _clip(abs(pressure) / REWARD_LIMITS["pressure_norm"], 0.0, 2.0),
        "wait": -weights["wait"] * _clip(total_wait_time / REWARD_LIMITS["wait_norm"], 0.0, 2.0),
        "worst_lane": -weights["worst_lane"] * _clip(max_wait_time / REWARD_LIMITS["worst_lane_norm"], 0.0, 2.0),
        "starvation": -weights["starvation"] * _clip(starvation_excess / REWARD_LIMITS["starvation_excess_norm"], 0.0, 2.0),
        "starved_lanes": -weights["starved_lanes"] * _clip(starved_lanes / REWARD_LIMITS["starved_lanes_norm"], 0.0, 2.0),
        "long_green": -weights["long_green"] * _clip(long_green_excess / REWARD_LIMITS["long_green_excess_norm"], 0.0, 2.0),
        "co2": -weights["co2"] * _clip(co2_emissions / REWARD_LIMITS["co2_norm"], 0.0, 2.0),
        "gridlock": -weights["gridlock"] * _clip(gridlock_seconds / REWARD_LIMITS["gridlock_seconds_norm"], 0.0, 2.0),
        "collisions": -weights["collision"] * collisions,
        "emergency_stops": -weights["emergency_stop"] * emergency_stops,
    }
    for component_name in REWARD_ABLATIONS.get(os.getenv("TRAFFICAI_REWARD_ABLATION", ""), []):
        if component_name in components:
            components[component_name] = 0.0

    reward = sum(components.values())

    metrics = {
        **components,
        "reward_total": reward,
        "raw_queue": queue,
        "raw_total_wait": total_wait_time,
        "raw_worst_lane_wait": max_wait_time,
        "raw_pressure": pressure,
        "raw_avg_speed": avg_speed,
        "raw_arrived": arrived_cars,
        "raw_co2": co2_emissions,
        "raw_stopped_ratio": stopped_ratio,
        "raw_gridlock_seconds": gridlock_seconds,
        "raw_phase_hold_seconds": phase_hold_seconds,
        "raw_starved_lanes": starved_lanes,
        "raw_starvation_excess": starvation_excess,
        "raw_collisions": collisions,
        "raw_emergency_stops": emergency_stops,
    }

    traffic_signal._gen10_reward_state = {
        "queue": queue,
        "wait": total_wait_time,
        "gridlock_seconds": gridlock_seconds,
        "phase_id": phase_id,
        "phase_since": phase_since,
        "sim_time": sim_time,
    }
    traffic_signal.last_reward_components = metrics

    env = getattr(traffic_signal, "env", None)
    if env is not None:
        env.last_reward_components = metrics

    return reward


def _traffic_light_phase(traffic_signal):
    tls_id = getattr(traffic_signal, "id", getattr(traffic_signal, "ts_id", None))
    if tls_id is not None:
        return traffic_signal.sumo.trafficlight.getPhase(tls_id)
    return getattr(traffic_signal, "green_phase", None)


REWARD_WEIGHTS = {
    "speed": 2.0,
    "throughput": 5.0,
    "delta_queue": 2.0,
    "delta_wait": 2.0,
    "queue": 2.0,
    "pressure": 0.8,
    "wait": 2.5,
    "worst_lane": 3.5,
    "starvation": 4.0,
    "starved_lanes": 1.5,
    "long_green": 1.2,
    "co2": 0.8,
    "gridlock": 6.0,
    "collision": 200.0,
    "emergency_stop": 15.0,
}

REWARD_LIMITS = {
    "speed_norm": 15.0,
    "throughput_norm": 10.0,
    "delta_queue_norm": 10.0,
    "delta_wait_norm": 300.0,
    "queue_norm": 80.0,
    "pressure_norm": 80.0,
    "wait_norm": 5000.0,
    "worst_lane_norm": 450.0,
    "starvation_wait_threshold": 420.0,
    "starvation_excess_norm": 1200.0,
    "starved_lanes_norm": 4.0,
    "max_green_soft": 90.0,
    "long_green_queue_threshold": 12.0,
    "long_green_excess_norm": 120.0,
    "co2_norm": 100000.0,
    "gridlock_queue_threshold": 25.0,
    "gridlock_stopped_ratio": 0.85,
    "gridlock_seconds_norm": 120.0,
}

REWARD_ABLATIONS = {
    "no_worst_lane": ["worst_lane"],
    "no_starvation": ["starvation", "starved_lanes"],
    "no_long_green": ["long_green"],
    "no_co2": ["co2"],
    "no_delta": ["delta_queue", "delta_wait"],
}


TRAIN_SETTINGS = {
    "num_cpu": 12,
    "total_timesteps": 3000000,
    "model_name": "ppo_traffic_model",
    "tensorboard_log": OUTPUT_DIRS["tensorboard"],
    "vecnormalize_path": "checkpoints/vecnormalize_latest.pkl",
    "smoke_timesteps": 49152,
    "warm_start_latest_model": True,
}


PPO_SETTINGS = {
    "learning_rate_initial": 3e-4,
    "learning_rate_final": 5e-5,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
    },
}


VEC_NORMALIZE_SETTINGS = {
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "gamma": PPO_SETTINGS["gamma"],
}


CURRICULUM_SETTINGS = {
    "enabled": True,
    "episode_seconds": 14400,
    "long_episode_seconds": 86400,
    "long_episode_probability": 0.25,
    "route_dir": TRAIN_ROUTE_DIR,
    "route_patterns": ["daily_*.rou.xml", "stress_*.rou.xml"],
    "fallback_route_files": [DEFAULT_ROUTE_FILE],
    "chaos_probabilities": [0.0, 0.0005, 0.001, 0.002],
    "demand_scales": [0.9, 1.0, 1.1],
    "route_per_episode": True,
    "seeds": [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203],
}


EVAL_SETTINGS = {
    "enabled": True,
    "eval_freq": 50000,
    "n_eval_episodes": 3,
    "route_file": DEFAULT_STRESS_ROUTE_FILE,
    "holdout_route_dir": HOLDOUT_ROUTE_DIR,
    "holdout_route_patterns": ["daily_*.rou.xml", "stress_*.rou.xml"],
    "chaos_prob": 0.001,
    "seed": 4242,
    "num_seconds": 14400,
}


SIM_SETTINGS = {
    "net_file": "SumoNetwork01.net.xml",
    "route_file": DEFAULT_ROUTE_FILE,
    "num_seconds": 86400,
    "min_green": 10,
    "max_green": 96,
    "enforce_max_green": True,
    "yellow_time": 3,
    "delta_time": 4,
    "single_agent": True,
    "time_to_teleport": -1,
    "additional_sumo_cmd": "--no-step-log --device.rerouting.probability 1.0",
    "reward_fn": balanced_reward,
}


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
    "f_13": {"from": "--30892#17", "to": "-31272#7"},
}
