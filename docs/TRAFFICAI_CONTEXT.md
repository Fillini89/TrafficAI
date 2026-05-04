# TrafficAI Context

TrafficAI is a smart traffic-light research project using Eclipse SUMO,
`sumo-rl`, Gymnasium, and Stable-Baselines3 PPO.

The immediate goal is to train a robust single-intersection traffic-light agent.
The long-term goal is to use this PPO policy as the decision-making brain for a
real intersection, where perception modules provide camera-derived traffic
features.

## Current Stack

- Simulator: Eclipse SUMO.
- Environment wrapper: `sumo-rl` + Gymnasium.
- RL algorithm: Stable-Baselines3 PPO.
- Parallel training: `SubprocVecEnv` with 12 environments.
- Windows robustness: `SyncBreakerWrapper` staggers SUMO/TraCI starts to reduce
  TCP port collisions.
- Normalization: `VecNormalize` for observations and rewards.
- Monitoring: TensorBoard, component reward logging, training reports, holdout
  comparison reports.
- Perturbations: `ChaosMonkeyWrapper` can stop a random vehicle temporarily to
  simulate incidents.

## Observation Philosophy

The observation interface should stay camera-compatible. SUMO currently provides
the data, but future production input should come from perception modules such
as YOLO, object tracking, lane association, and speed estimation.

The PPO model should receive aggregated lane/intersection metrics, not raw video
and not simulator-only internals.

Current Gen10/Gen11 observation includes:

- per-lane density,
- per-lane queue,
- per-lane normalized speed,
- per-lane waiting-time proxy,
- per-lane halting count,
- per-lane vehicle count,
- per-lane empty-lane flag,
- per-lane slowdown proxy,
- current phase,
- time since last switch,
- min-green switch availability,
- normalized time of day.

Important: Gen11 fine-tuning intentionally did not change observation shape, so
it can load Gen10 weights.

## Reward Philosophy

Reward must be componentized, bounded, and inspectable. Avoid raw unbounded
penalties that can dominate PPO updates or create unstable gradients.

The controller should optimize:

- flow and throughput,
- low queue length,
- low total waiting time,
- low worst-lane waiting time,
- low CO2 / stop-start behavior,
- robustness to incidents,
- no collisions or emergency stops,
- no gridlock,
- no long-horizon lane starvation.

## Why Fairness Matters

Gen10 showed that average traffic metrics can look acceptable while a small
number of lanes accumulate extreme waiting time. This is unacceptable. A
production-like controller must provide service guarantees and avoid sacrificing
minor approaches for main-road throughput.

Fairness is measured through:

- p95 waiting time,
- worst-lane waiting time,
- starved lane count,
- starvation excess,
- final total waiting time after long simulations,
- phase hold duration,
- phase switch count.

## Future Camera Integration

The future system should be layered:

1. Cameras observe the intersection.
2. Perception models detect and track objects.
3. Tracking/lane logic converts detections into lane-level aggregates.
4. The PPO policy receives the same style of aggregate observation it learned in
   SUMO.
5. Hard safety constraints validate or override unsafe phase decisions.

The PPO policy is the traffic-control brain, not the visual perception system.

