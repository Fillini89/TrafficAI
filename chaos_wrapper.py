import gymnasium as gym
import random


class PhaseSafetyWrapper(gym.Wrapper):
    """Apply conservative max-green guardrails without changing observation shape."""

    def __init__(self, env, max_green_steps=24, enforce=True):
        super().__init__(env)
        self.max_green_steps = max(1, int(max_green_steps))
        self.enforce = enforce
        self.current_phase = None
        self.phase_steps = 0
        self.forced_switches = 0

    def reset(self, **kwargs):
        self.current_phase = None
        self.phase_steps = 0
        self.forced_switches = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        phase_before = self._green_phase()
        if phase_before == self.current_phase:
            self.phase_steps += 1
        else:
            self.current_phase = phase_before
            self.phase_steps = 0

        forced = False
        if self.enforce and self.phase_steps >= self.max_green_steps:
            replacement = self._next_action(action)
            if replacement is not None:
                action = replacement
                forced = True
                self.forced_switches += 1
                self.phase_steps = 0

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["phase_hold_steps"] = self.phase_steps
        info["phase_forced_switch"] = forced
        info["phase_forced_switch_count"] = self.forced_switches
        return obs, reward, terminated, truncated, info

    def _green_phase(self):
        unwrapped = self.env.unwrapped
        traffic_signals = getattr(unwrapped, "traffic_signals", {})
        if isinstance(traffic_signals, dict) and traffic_signals:
            traffic_signal = next(iter(traffic_signals.values()))
            return getattr(traffic_signal, "green_phase", None)
        return None

    def _next_action(self, action):
        if not hasattr(self.action_space, "n"):
            return None

        current = self._green_phase()
        try:
            current = int(current)
        except (TypeError, ValueError):
            try:
                if hasattr(action, "item"):
                    action = action.item()
                current = int(action)
            except (TypeError, ValueError):
                return None
        return (current + 1) % int(self.action_space.n)


class ServiceDebtGuardrailWrapper(gym.Wrapper):
    """Force service for severely waiting lanes without changing observations."""

    def __init__(self, env, service_threshold_seconds=120.0, min_green_steps=3, enforce=True):
        super().__init__(env)
        self.service_threshold_seconds = float(service_threshold_seconds)
        self.min_green_steps = max(1, int(min_green_steps))
        self.enforce = enforce
        self.current_phase = None
        self.phase_steps = 0
        self.forced_switches = 0
        self.last_selected_action = None

    def reset(self, **kwargs):
        self.current_phase = None
        self.phase_steps = 0
        self.forced_switches = 0
        self.last_selected_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        phase_before = self._green_phase()
        if phase_before == self.current_phase:
            self.phase_steps += 1
        else:
            self.current_phase = phase_before
            self.phase_steps = 0

        waits = self._lane_waits()
        max_service_debt = max(waits.values(), default=0.0)
        replacement = self._best_service_action(waits) if self.enforce else None
        forced = False
        selected_action = None

        if replacement is not None and self.phase_steps >= self.min_green_steps:
            current_action = self._action_value(action)
            if current_action is None or int(replacement) != int(current_action):
                action = int(replacement)
                forced = True
                selected_action = int(replacement)
                self.forced_switches += 1
                self.phase_steps = 0

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["fairness_forced_switch"] = forced
        info["fairness_forced_switch_count"] = self.forced_switches
        info["max_service_debt"] = max_service_debt
        info["selected_debt_phase"] = selected_action
        self.last_selected_action = selected_action
        return obs, reward, terminated, truncated, info

    def _traffic_signal(self):
        traffic_signals = getattr(self.env.unwrapped, "traffic_signals", {})
        if isinstance(traffic_signals, dict) and traffic_signals:
            return next(iter(traffic_signals.values()))
        return None

    def _green_phase(self):
        traffic_signal = self._traffic_signal()
        if traffic_signal is None:
            return None
        return getattr(traffic_signal, "green_phase", None)

    def _lane_waits(self):
        traffic_signal = self._traffic_signal()
        if traffic_signal is None:
            return {}
        try:
            waits = traffic_signal.get_accumulated_waiting_time_per_lane()
        except Exception:
            return {}
        if isinstance(waits, dict):
            return {lane: float(value) for lane, value in waits.items()}
        lanes = list(getattr(traffic_signal, "lanes", []))
        return {lane: float(value) for lane, value in zip(lanes, waits)}

    def _best_service_action(self, waits):
        if not waits or max(waits.values()) < self.service_threshold_seconds:
            return None
        if not hasattr(self.action_space, "n"):
            return None

        best_action = None
        best_score = 0.0
        for action in range(int(self.action_space.n)):
            served_lanes = self._served_lanes_for_action(action)
            if not served_lanes:
                continue
            score = sum(max(waits.get(lane, 0.0) - self.service_threshold_seconds, 0.0) for lane in served_lanes)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _served_lanes_for_action(self, action):
        traffic_signal = self._traffic_signal()
        if traffic_signal is None:
            return set()

        state = self._phase_state(traffic_signal, action)
        if not state:
            return set()

        tls_id = getattr(traffic_signal, "id", getattr(traffic_signal, "ts_id", None))
        if tls_id is None:
            return set()

        try:
            controlled_links = traffic_signal.sumo.trafficlight.getControlledLinks(tls_id)
        except Exception:
            return set()

        served = set()
        for index, signal_state in enumerate(str(state)):
            if signal_state not in {"G", "g"} or index >= len(controlled_links):
                continue
            for link in controlled_links[index]:
                if link and link[0]:
                    served.add(link[0])
        return served

    def _phase_state(self, traffic_signal, action):
        green_phases = getattr(traffic_signal, "green_phases", None)
        if green_phases and 0 <= int(action) < len(green_phases):
            phase = green_phases[int(action)]
            return getattr(phase, "state", phase)

        tls_id = getattr(traffic_signal, "id", getattr(traffic_signal, "ts_id", None))
        if tls_id is None:
            return None
        try:
            definitions = traffic_signal.sumo.trafficlight.getAllProgramLogics(tls_id)
            phases = getattr(definitions[0], "phases", None) if definitions else None
        except Exception:
            return None
        if phases and 0 <= int(action) < len(phases):
            return getattr(phases[int(action)], "state", None)
        return None

    def _action_value(self, action):
        try:
            if hasattr(action, "item"):
                action = action.item()
            return int(action)
        except (TypeError, ValueError):
            return None


class ChaosMonkeyWrapper(gym.Wrapper):
    """Inject temporary lane-blocking breakdowns to train robust policies."""

    def __init__(self, env, chaos_prob=0.001, block_steps=12, seed=None):
        super().__init__(env)
        self.chaos_prob = chaos_prob
        self.block_steps = block_steps
        self.rng = random.Random(seed)
        self.blocked_vehicle = None
        self.block_timer = 0
        self.incident_count = 0

    def reset(self, **kwargs):
        self.blocked_vehicle = None
        self.block_timer = 0
        self.incident_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        traci = self.env.unwrapped.sumo
        incident_started = False

        if self.block_timer > 0:
            self.block_timer -= 1
            if self.block_timer == 0 and self.blocked_vehicle in traci.vehicle.getIDList():
                traci.vehicle.setColor(self.blocked_vehicle, (255, 255, 0, 255))
                traci.vehicle.setSpeed(self.blocked_vehicle, -1)
                self.blocked_vehicle = None
        elif self.chaos_prob > 0 and self.rng.random() < self.chaos_prob:
            vehicles = list(traci.vehicle.getIDList())
            if vehicles:
                self.blocked_vehicle = self.rng.choice(vehicles)
                self.block_timer = self.block_steps
                self.incident_count += 1
                incident_started = True
                traci.vehicle.setColor(self.blocked_vehicle, (255, 0, 0, 255))
                traci.vehicle.setSpeed(self.blocked_vehicle, 0.0)

        info = dict(info)
        info["chaos_active"] = self.block_timer > 0
        info["chaos_incident_started"] = incident_started
        info["chaos_incident_count"] = self.incident_count
        return obs, reward, terminated, truncated, info


class RewardInfoWrapper(gym.Wrapper):
    """Expose reward components from config.balanced_reward through Gym info."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        components = self._collect_reward_components()
        if components:
            info["reward_components"] = components
        return obs, reward, terminated, truncated, info

    def _collect_reward_components(self):
        unwrapped = self.env.unwrapped
        direct = getattr(unwrapped, "last_reward_components", None)
        if direct:
            return direct

        traffic_signals = getattr(unwrapped, "traffic_signals", {})
        if isinstance(traffic_signals, dict):
            for traffic_signal in traffic_signals.values():
                components = getattr(traffic_signal, "last_reward_components", None)
                if components:
                    return components
        return None
