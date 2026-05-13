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

    def __init__(
        self,
        env,
        service_threshold_seconds=120.0,
        soft_service_threshold_seconds=90.0,
        min_green_steps=3,
        min_hold_steps=0,
        target_hold_steps=8,
        protected_hold_steps=6,
        adaptive_service_threshold_seconds=75.0,
        queue_imbalance_threshold=6.0,
        worse_debt_multiplier=1.25,
        worse_debt_seconds=None,
        use_service_age=False,
        cadence_suppression_penalty=0.0,
        service_age_warning_seconds=150.0,
        service_age_critical_seconds=210.0,
        service_age_warning_norm=60.0,
        service_age_critical_norm=60.0,
        service_age_warning_penalty_weight=0.0,
        service_age_critical_penalty_weight=0.0,
        service_age_budget_penalty_clip=0.0,
        service_age_critical_override=False,
        enforce=True,
    ):
        super().__init__(env)
        self.service_threshold_seconds = float(service_threshold_seconds)
        self.soft_service_threshold_seconds = float(soft_service_threshold_seconds)
        self.min_green_steps = max(1, int(min_green_steps))
        self.min_hold_steps = max(0, int(min_hold_steps))
        self.target_hold_steps = max(0, int(target_hold_steps))
        self.protected_hold_steps = max(0, int(protected_hold_steps))
        self.adaptive_service_threshold_seconds = float(adaptive_service_threshold_seconds)
        self.queue_imbalance_threshold = max(0.0, float(queue_imbalance_threshold))
        self.worse_debt_multiplier = max(1.0, float(worse_debt_multiplier))
        self.worse_debt_seconds = None if worse_debt_seconds is None else float(worse_debt_seconds)
        self.use_service_age = bool(use_service_age)
        self.cadence_suppression_penalty = float(cadence_suppression_penalty)
        self.service_age_warning_seconds = float(service_age_warning_seconds)
        self.service_age_critical_seconds = float(service_age_critical_seconds)
        self.service_age_warning_norm = max(float(service_age_warning_norm), 1.0)
        self.service_age_critical_norm = max(float(service_age_critical_norm), 1.0)
        self.service_age_warning_penalty_weight = max(0.0, float(service_age_warning_penalty_weight))
        self.service_age_critical_penalty_weight = max(0.0, float(service_age_critical_penalty_weight))
        self.service_age_budget_penalty_clip = max(0.0, float(service_age_budget_penalty_clip))
        self.service_age_critical_override = bool(service_age_critical_override)
        self.enforce = enforce
        self.current_phase = None
        self.phase_steps = 0
        self.forced_switches = 0
        self.suppressed_switches = 0
        self.cadence_suppressed_switches = 0
        self.actual_switches = 0
        self.last_selected_action = None
        self.protected_action = None
        self.protected_steps_remaining = 0
        self.last_forced_debt = 0.0
        self.service_age_seconds = {}
        self.last_sim_time = None
        self.last_executed_action = None

    def reset(self, **kwargs):
        self.current_phase = None
        self.phase_steps = 0
        self.forced_switches = 0
        self.suppressed_switches = 0
        self.cadence_suppressed_switches = 0
        self.actual_switches = 0
        self.last_selected_action = None
        self.protected_action = None
        self.protected_steps_remaining = 0
        self.last_forced_debt = 0.0
        self.service_age_seconds = {}
        self.last_sim_time = None
        self.last_executed_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        phase_before = self._green_phase()
        if phase_before == self.current_phase:
            self.phase_steps += 1
        else:
            self.current_phase = phase_before
            self.phase_steps = 0

        self._update_service_age()
        debts = self._service_debts()
        queues = self._lane_queues()
        max_service_debt = max(debts.values(), default=0.0)
        replacement = self._best_service_action(debts) if self.enforce else None
        forced = False
        suppressed = False
        cadence_suppressed = False
        hold_active = False
        selected_action = None
        requested_action = self._action_value(action)
        final_action = requested_action
        current_phase_action = self._action_value(phase_before)
        hard_service_debt = max_service_debt >= self.service_threshold_seconds
        critical_service_debt = max_service_debt >= self.service_age_critical_seconds
        critical_service_override_debt = critical_service_debt and self.service_age_critical_override
        service_age_warning_excess = max(max_service_debt - self.service_age_warning_seconds, 0.0)
        service_age_critical_excess = max(max_service_debt - self.service_age_critical_seconds, 0.0)
        service_age_budget_penalty = self._service_age_budget_penalty(
            service_age_warning_excess,
            service_age_critical_excess,
        )
        requested_service_debt = self._service_score_above(
            requested_action,
            debts,
            self.adaptive_service_threshold_seconds,
        )
        requested_queue_advantage = self._queue_advantage(requested_action, current_phase_action, queues)
        adaptive_release = (
            self.phase_steps >= self.min_green_steps
            and (
                requested_service_debt > 0.0
                or requested_queue_advantage >= self.queue_imbalance_threshold
                or self.phase_steps >= self.target_hold_steps
            )
        )

        if self.enforce:
            protected_active = self.protected_steps_remaining > 0 and self.protected_action is not None
            worse_debt_threshold = max(self.service_threshold_seconds, self.last_forced_debt * self.worse_debt_multiplier)
            if self.worse_debt_seconds is not None:
                worse_debt_threshold = max(worse_debt_threshold, self.worse_debt_seconds)
            worse_debt = max_service_debt >= worse_debt_threshold

            if protected_active and not worse_debt and not critical_service_override_debt:
                hold_active = True
                if requested_action is None or int(requested_action) != int(self.protected_action):
                    final_action = int(self.protected_action)
                    action = final_action
                if replacement is not None and int(replacement) != int(self.protected_action):
                    suppressed = True
                    self.suppressed_switches += 1
            else:
                if (
                    current_phase_action is not None
                    and requested_action is not None
                    and int(requested_action) != int(current_phase_action)
                    and self.phase_steps < self.min_hold_steps
                    and not hard_service_debt
                    and not adaptive_release
                ):
                    final_action = int(current_phase_action)
                    action = final_action
                    cadence_suppressed = True
                    self.cadence_suppressed_switches += 1

                if replacement is not None and self.phase_steps >= self.min_green_steps:
                    current_after_cadence = self._action_value(action)
                    if current_after_cadence is None or int(replacement) != int(current_after_cadence):
                        final_action = int(replacement)
                        action = final_action
                        forced = True
                        selected_action = int(replacement)
                        self.forced_switches += 1
                        self.phase_steps = 0
                        self.protected_action = int(replacement)
                        self.protected_steps_remaining = self.protected_hold_steps
                        self.last_forced_debt = max_service_debt

            if self.protected_steps_remaining > 0:
                self.protected_steps_remaining -= 1
            if self.protected_steps_remaining <= 0 and not forced:
                self.protected_action = None

        obs, reward, terminated, truncated, info = self.env.step(action)
        if cadence_suppressed and self.cadence_suppression_penalty:
            reward += self.cadence_suppression_penalty
        if service_age_budget_penalty:
            reward += service_age_budget_penalty

        executed_action = self._action_value(action)
        actual_action_changed = (
            executed_action is not None
            and self.last_executed_action is not None
            and int(executed_action) != int(self.last_executed_action)
        )
        if actual_action_changed:
            self.actual_switches += 1
        if executed_action is not None:
            self.last_executed_action = int(executed_action)

        info = dict(info)
        info["requested_action"] = requested_action
        info["executed_action"] = executed_action
        info["actual_phase_changed"] = actual_action_changed
        info["actual_phase_switch_count"] = self.actual_switches
        info["fairness_forced_switch"] = forced
        info["fairness_forced_switch_count"] = self.forced_switches
        info["fairness_guardrail_suppressed_switch"] = suppressed
        info["fairness_guardrail_suppressed_switch_count"] = self.suppressed_switches
        info["cadence_suppressed_switch"] = cadence_suppressed
        info["cadence_suppressed_switch_count"] = self.cadence_suppressed_switches
        info["cadence_suppression_penalty"] = self.cadence_suppression_penalty if cadence_suppressed else 0.0
        info["adaptive_cadence_release"] = bool(adaptive_release)
        info["requested_service_debt"] = requested_service_debt
        info["requested_queue_advantage"] = requested_queue_advantage
        info["fairness_guardrail_hold_active"] = hold_active
        info["fairness_guardrail_hold_steps_remaining"] = self.protected_steps_remaining
        info["max_service_debt"] = max_service_debt
        info["max_service_age"] = max(self.service_age_seconds.values(), default=0.0)
        info["hard_service_debt"] = hard_service_debt
        info["critical_service_debt"] = critical_service_debt
        info["service_age_warning_excess"] = service_age_warning_excess
        info["service_age_critical_excess"] = service_age_critical_excess
        info["service_age_budget_penalty"] = service_age_budget_penalty
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

    def _lane_queues(self):
        traffic_signal = self._traffic_signal()
        if traffic_signal is None:
            return {}
        lanes = list(getattr(traffic_signal, "lanes", []))
        queues = {}
        for lane in lanes:
            try:
                queues[lane] = float(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane))
            except Exception:
                queues[lane] = 0.0
        return queues

    def _service_debts(self):
        if self.use_service_age:
            return dict(self.service_age_seconds)
        return self._lane_waits()

    def _update_service_age(self):
        if not self.use_service_age:
            return

        traffic_signal = self._traffic_signal()
        if traffic_signal is None:
            return

        try:
            sim_time = float(traffic_signal.sumo.simulation.getTime())
        except Exception:
            sim_time = None

        if sim_time is None:
            dt = 0.0
        elif self.last_sim_time is None or sim_time < self.last_sim_time:
            dt = 0.0
        else:
            dt = max(sim_time - self.last_sim_time, 0.0)
        self.last_sim_time = sim_time

        queues = self._lane_queues()
        phase_action = self._action_value(self._green_phase())
        served_lanes = self._served_lanes_for_action(phase_action) if phase_action is not None else set()

        known_lanes = set(queues) | set(self.service_age_seconds)
        for lane in known_lanes:
            queue = queues.get(lane, 0.0)
            if queue <= 0.0 or lane in served_lanes:
                self.service_age_seconds[lane] = 0.0
            else:
                self.service_age_seconds[lane] = self.service_age_seconds.get(lane, 0.0) + dt

    def _best_service_action(self, debts):
        if not debts or max(debts.values()) < self.service_threshold_seconds:
            return None
        if not hasattr(self.action_space, "n"):
            return None

        best_action = None
        best_score = 0.0
        for action in range(int(self.action_space.n)):
            score = self._service_score(action, debts)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _service_score(self, action, debts):
        return self._service_score_above(action, debts, self.soft_service_threshold_seconds)

    def _service_score_above(self, action, debts, threshold):
        if action is None:
            return 0.0
        served_lanes = self._served_lanes_for_action(action)
        if not served_lanes:
            return 0.0
        return sum(max(debts.get(lane, 0.0) - threshold, 0.0) for lane in served_lanes)

    def _queue_score(self, action, queues):
        if action is None:
            return 0.0
        served_lanes = self._served_lanes_for_action(action)
        if not served_lanes:
            return 0.0
        return sum(max(queues.get(lane, 0.0), 0.0) for lane in served_lanes)

    def _queue_advantage(self, requested_action, current_action, queues):
        if requested_action is None or current_action is None:
            return 0.0
        if int(requested_action) == int(current_action):
            return 0.0
        return self._queue_score(requested_action, queues) - self._queue_score(current_action, queues)

    def _service_age_budget_penalty(self, warning_excess, critical_excess):
        if self.service_age_budget_penalty_clip <= 0.0:
            return 0.0
        penalty = (
            self.service_age_warning_penalty_weight * (warning_excess / self.service_age_warning_norm)
            + self.service_age_critical_penalty_weight * (critical_excess / self.service_age_critical_norm)
        )
        penalty = min(max(penalty, 0.0), self.service_age_budget_penalty_clip)
        return -penalty

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
