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
