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


def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _wait_values(wait_times):
    if isinstance(wait_times, dict):
        return list(wait_times.values())
    if wait_times is None:
        return []
    return list(wait_times)


def _convex_wait_penalty(values, threshold, norm, high=3.0):
    penalty = 0.0
    for value in values:
        excess = max(float(value) - threshold, 0.0) / max(norm, 1.0)
        penalty += excess + (excess * excess)
    return _clip(penalty, 0.0, high)


def balanced_reward(traffic_signal):
    """Current fairness reward: preserve Gen12 tail discipline; cadence lives in wrappers."""

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
    previous_phase_id = state.get("phase_id")
    previous_phase_since = state.get("phase_since", sim_time)
    phase_changed = previous_phase_id is not None and phase_id != previous_phase_id
    previous_phase_hold_seconds = max(0.0, sim_time - previous_phase_since)
    if phase_changed:
        phase_since = sim_time
    else:
        phase_since = previous_phase_since
    phase_hold_seconds = max(0.0, sim_time - phase_since)

    starvation_threshold = REWARD_LIMITS["starvation_wait_threshold"]
    starved_lanes = sum(1 for value in wait_values if value >= starvation_threshold)
    starvation_excess = sum(max(value - starvation_threshold, 0.0) for value in wait_values)
    tail_wait_values = sorted(wait_values, reverse=True)[: REWARD_LIMITS["tail_wait_top_k"]]
    tail_wait_mean = sum(tail_wait_values) / len(tail_wait_values) if tail_wait_values else 0.0
    tail_wait_excess = sum(max(value - REWARD_LIMITS["tail_wait_threshold"], 0.0) for value in tail_wait_values)
    tail_wait_penalty = _convex_wait_penalty(
        tail_wait_values,
        REWARD_LIMITS["tail_wait_threshold"],
        REWARD_LIMITS["tail_wait_norm"],
        high=REWARD_LIMITS["tail_wait_penalty_clip"],
    )
    starvation_penalty = _convex_wait_penalty(
        wait_values,
        starvation_threshold,
        REWARD_LIMITS["starvation_wait_norm"],
        high=REWARD_LIMITS["starvation_penalty_clip"],
    )
    fairness_debt = _clip(
        max_wait_time / REWARD_LIMITS["fairness_gate_hard_wait"]
        + tail_wait_excess / REWARD_LIMITS["fairness_gate_tail_norm"]
        + starved_lanes / REWARD_LIMITS["fairness_gate_starved_norm"]
        + starvation_excess / REWARD_LIMITS["fairness_gate_starvation_norm"],
        0.0,
        4.0,
    )
    hard_fairness_debt = (
        max_wait_time >= REWARD_LIMITS["fairness_gate_hard_wait"]
        or starved_lanes >= REWARD_LIMITS["fairness_gate_hard_starved_lanes"]
    )
    if hard_fairness_debt:
        flow_gate = 0.0
    elif max_wait_time >= REWARD_LIMITS["fairness_gate_soft_wait"] or starved_lanes > 0:
        flow_gate = _clip(1.0 - (fairness_debt * REWARD_LIMITS["fairness_gate_slope"]), 0.15, 1.0)
    else:
        flow_gate = 1.0

    short_phase_excess = 0.0
    if phase_changed and not hard_fairness_debt:
        short_phase_excess = max(REWARD_LIMITS["short_phase_target"] - previous_phase_hold_seconds, 0.0)
    phase_change_penalty = 1.0 if phase_changed and not hard_fairness_debt else 0.0
    long_green_active = (
        phase_hold_seconds >= REWARD_LIMITS["max_green_soft"]
        and (starved_lanes > 0 or queue >= REWARD_LIMITS["long_green_queue_threshold"])
    )
    long_green_excess = max(phase_hold_seconds - REWARD_LIMITS["max_green_soft"], 0.0) if long_green_active else 0.0

    arrived_cars = float(_safe_call(traffic_signal.sumo.simulation.getArrivedNumber))
    collisions = float(_safe_call(traffic_signal.sumo.simulation.getCollidingVehiclesNumber))
    emergency_stops = float(_safe_call(traffic_signal.sumo.simulation.getEmergencyStoppingVehiclesNumber))

    components = {
        "speed": flow_gate * weights["speed"] * _clip(avg_speed / REWARD_LIMITS["speed_norm"], 0.0, 1.0),
        "throughput": flow_gate * weights["throughput"] * _clip(arrived_cars / REWARD_LIMITS["throughput_norm"], 0.0, 2.0),
        "delta_queue": weights["delta_queue"] * _clip(delta_queue / REWARD_LIMITS["delta_queue_norm"], -1.0, 1.0),
        "delta_wait": weights["delta_wait"] * _clip(delta_wait / REWARD_LIMITS["delta_wait_norm"], -1.0, 1.0),
        "queue": -weights["queue"] * _clip(queue / REWARD_LIMITS["queue_norm"], 0.0, 2.0),
        "pressure": -weights["pressure"] * _clip(abs(pressure) / REWARD_LIMITS["pressure_norm"], 0.0, 2.0),
        "wait": -weights["wait"] * _clip(total_wait_time / REWARD_LIMITS["wait_norm"], 0.0, 2.0),
        "worst_lane": -weights["worst_lane"] * _clip(max_wait_time / REWARD_LIMITS["worst_lane_norm"], 0.0, 2.0),
        "tail_wait": -weights["tail_wait"] * tail_wait_penalty,
        "starvation": -weights["starvation"] * starvation_penalty,
        "starved_lanes": -weights["starved_lanes"] * _clip(starved_lanes / REWARD_LIMITS["starved_lanes_norm"], 0.0, 2.0),
        "short_phase": -weights["short_phase"] * _clip(short_phase_excess / REWARD_LIMITS["short_phase_norm"], 0.0, 1.0),
        "phase_change": -weights["phase_change"] * phase_change_penalty,
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
        "raw_previous_phase_hold_seconds": previous_phase_hold_seconds,
        "raw_phase_changed": 1.0 if phase_changed else 0.0,
        "raw_short_phase_excess": short_phase_excess,
        "raw_phase_change_penalty": phase_change_penalty,
        "raw_tail_wait_mean": tail_wait_mean,
        "raw_tail_wait_excess": tail_wait_excess,
        "raw_tail_wait_penalty": tail_wait_penalty,
        "raw_starved_lanes": starved_lanes,
        "raw_starvation_excess": starvation_excess,
        "raw_starvation_penalty": starvation_penalty,
        "raw_fairness_debt": fairness_debt,
        "raw_flow_gate": flow_gate,
        "raw_hard_fairness_debt": 1.0 if hard_fairness_debt else 0.0,
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
    green_phase = getattr(traffic_signal, "green_phase", None)
    if green_phase is not None:
        return green_phase

    tls_id = getattr(traffic_signal, "id", getattr(traffic_signal, "ts_id", None))
    if tls_id is not None:
        return traffic_signal.sumo.trafficlight.getPhase(tls_id)
    return None


REWARD_WEIGHTS = {
    "speed": 1.5,
    "throughput": 3.25,
    "delta_queue": 2.0,
    "delta_wait": 2.0,
    "queue": 2.0,
    "pressure": 0.8,
    "wait": 2.5,
    "worst_lane": 5.0,
    "tail_wait": 5.0,
    "starvation": 7.0,
    "starved_lanes": 3.0,
    "short_phase": 1.4,
    "phase_change": 0.05,
    "long_green": 1.0,
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
    "worst_lane_norm": 300.0,
    "tail_wait_top_k": 3,
    "tail_wait_threshold": 90.0,
    "tail_wait_norm": 360.0,
    "tail_wait_penalty_clip": 3.0,
    "starvation_wait_threshold": 120.0,
    "starvation_excess_norm": 1200.0,
    "starvation_wait_norm": 360.0,
    "starvation_penalty_clip": 3.0,
    "starved_lanes_norm": 4.0,
    "short_phase_target": 28.0,
    "short_phase_norm": 28.0,
    "fairness_gate_soft_wait": 90.0,
    "fairness_gate_hard_wait": 120.0,
    "fairness_gate_hard_starved_lanes": 2.0,
    "fairness_gate_tail_norm": 1800.0,
    "fairness_gate_starved_norm": 4.0,
    "fairness_gate_starvation_norm": 2400.0,
    "fairness_gate_slope": 0.45,
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
    "no_tail_wait": ["tail_wait"],
    "no_starvation": ["starvation", "starved_lanes", "tail_wait"],
    "no_short_phase": ["short_phase", "phase_change"],
    "no_long_green": ["long_green"],
    "no_co2": ["co2"],
    "no_delta": ["delta_queue", "delta_wait"],
}


TRAIN_SETTINGS = {
    "num_cpu": _env_int("TRAFFICAI_NUM_CPU", 12, minimum=1),
    "total_timesteps": _env_int("TRAFFICAI_TOTAL_TIMESTEPS", 3000000, minimum=1),
    "model_name": "ppo_traffic_model",
    "tensorboard_log": OUTPUT_DIRS["tensorboard"],
    "vecnormalize_path": "checkpoints/vecnormalize_latest.pkl",
    "smoke_timesteps": 49152,
    "warm_start_latest_model": True,
    "continue_latest_checkpoint": False,
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
    "long_episode_probability": 0.50,
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
    "service_debt_metric": "service_age",
    "service_debt_soft_threshold": 90.0,
    "service_debt_threshold": 120.0,
    "service_debt_min_hold": 16.0,
    "service_debt_target_hold": 24.0,
    "service_debt_protected_hold": 12.0,
    "service_debt_adaptive_threshold": 70.0,
    "service_debt_queue_imbalance_threshold": 5.0,
    "service_debt_worse_multiplier": 1.0,
    "service_debt_worse_threshold": 130.0,
    "cadence_suppression_penalty": -0.075,
    "service_age_warning_threshold": 150.0,
    "service_age_critical_threshold": 210.0,
    "service_age_warning_norm": 60.0,
    "service_age_critical_norm": 60.0,
    "service_age_warning_penalty_weight": 0.06,
    "service_age_critical_penalty_weight": 0.12,
    "service_age_budget_penalty_clip": 0.25,
    "service_age_critical_override": True,
    "enforce_service_debt": True,
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
