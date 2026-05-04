from gymnasium import spaces
from sumo_rl import ObservationFunction
import numpy as np
import warnings


class LegacyRadarObservation(ObservationFunction):
    """Original Gen 8/9 observation shape kept for backward-compatible eval."""

    def __init__(self, ts):
        super().__init__(ts)

    def __call__(self):
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        speeds = []
        for lane in self.ts.lanes:
            vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            if len(vehicles) == 0:
                speeds.append(1.0)
            else:
                avg_speed = sum(self.ts.sumo.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicles) / len(vehicles)
                max_speed = self.ts.sumo.lane.getMaxSpeed(lane)
                speeds.append(avg_speed / max(max_speed, 1e-6))

        return np.array(density + queue + speeds, dtype=np.float32)

    def observation_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(3 * len(self.ts.lanes),), dtype=np.float32)


class RadarObservation(ObservationFunction):
    """Camera-compatible aggregated intersection state for Gen 10.

    Per lane we expose normalized aggregates that can later be estimated from
    video tracking: occupancy, queue, speed, waiting/stopped time proxies,
    halting count, vehicle count, and an explicit empty-lane flag.
    """

    LANE_FEATURES = 8
    GLOBAL_FEATURES = 4
    DAY_SECONDS = 24 * 60 * 60
    WAIT_NORMALIZER = 300.0
    SWITCH_NORMALIZER = 300.0
    DEFAULT_VEHICLE_LENGTH = 7.5
    DEFAULT_MAX_PHASES = 16.0

    def __init__(self, ts):
        super().__init__(ts)
        self._last_phase = None
        self._last_phase_change_time = None
        self._phase_count = None

    def __call__(self):
        lanes = list(self.ts.lanes)
        density = self._as_lane_list(self.ts.get_lanes_density(), lanes)
        queue = self._as_lane_list(self.ts.get_lanes_queue(), lanes)
        waiting_time = self._as_lane_list(self.ts.get_accumulated_waiting_time_per_lane(), lanes)

        lane_features = []
        for idx, lane_id in enumerate(lanes):
            vehicle_count = self._safe_call(self.ts.sumo.lane.getLastStepVehicleNumber, lane_id, default=0)
            halting_count = self._safe_call(self.ts.sumo.lane.getLastStepHaltingNumber, lane_id, default=0)
            capacity = self._lane_capacity(lane_id)

            if vehicle_count <= 0:
                speed_norm = 0.0
                empty_flag = 1.0
            else:
                speed = self._mean_lane_speed(lane_id)
                max_speed = self._safe_call(self.ts.sumo.lane.getMaxSpeed, lane_id, default=1.0)
                speed_norm = self._clip01(speed / max(max_speed, 1e-6))
                empty_flag = 0.0

            lane_features.extend(
                [
                    self._clip01(density[idx]),
                    self._clip01(queue[idx]),
                    speed_norm,
                    self._clip01(waiting_time[idx] / self.WAIT_NORMALIZER),
                    self._clip01(halting_count / capacity),
                    self._clip01(vehicle_count / capacity),
                    empty_flag,
                    self._clip01(1.0 - speed_norm if vehicle_count > 0 else 0.0),
                ]
            )

        phase_norm, time_since_switch_norm, can_switch, time_of_day = self._global_features()
        obs = lane_features + [phase_norm, time_since_switch_norm, can_switch, time_of_day]
        return np.array(obs, dtype=np.float32)

    def observation_space(self):
        shape = (self.LANE_FEATURES * len(self.ts.lanes) + self.GLOBAL_FEATURES,)
        return spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

    def _global_features(self):
        sim_time = float(self._safe_call(self.ts.sumo.simulation.getTime, default=0.0))
        phase = int(self._get_current_phase())

        if self._last_phase is None:
            self._last_phase = phase
            self._last_phase_change_time = sim_time
        elif phase != self._last_phase:
            self._last_phase = phase
            self._last_phase_change_time = sim_time

        time_since_switch = max(0.0, sim_time - float(self._last_phase_change_time or sim_time))
        min_green = float(getattr(self.ts, "min_green", 0.0) or 0.0)

        if self._phase_count is None:
            self._phase_count = self._get_phase_count()
        phase_count = max(float(self._phase_count), 1.0)
        phase_norm = self._clip01(phase / max(phase_count - 1.0, 1.0))
        time_since_switch_norm = self._clip01(time_since_switch / self.SWITCH_NORMALIZER)
        can_switch = 1.0 if time_since_switch >= min_green else 0.0
        time_of_day = (sim_time % self.DAY_SECONDS) / self.DAY_SECONDS
        return phase_norm, time_since_switch_norm, can_switch, time_of_day

    def _get_current_phase(self):
        tls_id = getattr(self.ts, "id", getattr(self.ts, "ts_id", None))
        if tls_id is not None:
            return self._safe_call(self.ts.sumo.trafficlight.getPhase, tls_id, default=0)
        return getattr(self.ts, "green_phase", 0)

    def _get_phase_count(self):
        tls_id = getattr(self.ts, "id", getattr(self.ts, "ts_id", None))
        if tls_id is None:
            return self.DEFAULT_MAX_PHASES

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*getAllProgramLogics.*",
                category=UserWarning,
            )
            definitions = self._safe_call(
                self.ts.sumo.trafficlight.getCompleteRedYellowGreenDefinition,
                tls_id,
                default=[],
            )
        if definitions:
            phases = getattr(definitions[0], "phases", None)
            if phases:
                return len(phases)
        return self.DEFAULT_MAX_PHASES

    def _mean_lane_speed(self, lane_id):
        vehicles = self._safe_call(self.ts.sumo.lane.getLastStepVehicleIDs, lane_id, default=[])
        if not vehicles:
            return 0.0
        speeds = [self._safe_call(self.ts.sumo.vehicle.getSpeed, vehicle_id, default=0.0) for vehicle_id in vehicles]
        return sum(speeds) / max(len(speeds), 1)

    def _lane_capacity(self, lane_id):
        length = self._safe_call(self.ts.sumo.lane.getLength, lane_id, default=50.0)
        return max(length / self.DEFAULT_VEHICLE_LENGTH, 1.0)

    @staticmethod
    def _clip01(value):
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _safe_call(func, *args, default=None):
        try:
            return func(*args)
        except Exception:
            return default

    @staticmethod
    def _as_lane_list(values, lanes):
        if isinstance(values, dict):
            return [float(values.get(lane_id, 0.0)) for lane_id in lanes]
        if isinstance(values, np.ndarray):
            values = values.tolist()
        values = list(values or [])
        if len(values) < len(lanes):
            values.extend([0.0] * (len(lanes) - len(values)))
        return [float(value) for value in values[: len(lanes)]]
