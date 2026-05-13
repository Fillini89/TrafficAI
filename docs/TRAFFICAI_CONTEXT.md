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

Current Gen10/Gen11/Gen12/Gen13/Gen14/Gen15/Gen16 observation includes:

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

Important: Gen11 through Gen16 fine-tuning intentionally does not change
observation shape, so each generation can warm-start from a compatible final
model pair.

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

Rebuilt Gen12 adds non-compensable fairness without changing observation shape:

- a bounded `tail_wait` penalty over the worst few lane waits, aimed at p95-like
  fairness rather than only the single worst lane,
- a fairness gate that attenuates or zeros flow rewards when lane wait debt is
  severe,
- convex tail/starvation penalties above fairness thresholds,
- a bounded `short_phase` penalty for unnecessary short cycling,
- an external service-debt guardrail for severe lane wait. Rebuilt Gen12 treats
  90 seconds as soft fairness debt and 120 seconds as both reward starvation and
  emergency intervention threshold.

The first Gen13 reward-only smoothing attempt failed: the policy improved
speed/stopped burden but increased switching and regressed p95/final wait.
Rebuilt Gen13 preserves the fairness core and moves smoothing into control
guardrails:

- service-age debt tracks how long a waiting lane has not received serving
  green,
- 90/120 seconds are soft/hard service-age thresholds,
- non-urgent changes before 24 seconds actual hold are suppressed,
- reporting separates policy action changes from actual executed signal changes.

Gen14 starts from protected Gen12 rather than Gen13. It keeps Gen12's
non-compensable fairness reward and Gen13's service-age semantics, but replaces
hard cadence with adaptive cadence:

- normal non-urgent changes are suppressed only before about 16 seconds actual
  hold,
- target hold is about 24 seconds,
- service age around 75 seconds or high queue imbalance can release cadence
  after min-green,
- hard service-age debt at 120 seconds still forces service,
- protected hold after a fairness-forced service is about 12 seconds.

Gen15 starts from protected Gen14 and keeps the same reward core. It only nudges
adaptive cadence earlier: 70-second moderate service-age release, 5.0 queue
imbalance release, 130-second worse-emergency override, and a slightly stronger
cadence suppression penalty. The intent is to reduce Gen14's remaining
final-wait/service-age tail without giving back its speed and smoothness gains.

Gen16 starts from protected Gen15 and keeps Gen15's flow/speed intent. It adds a
service-age budget: warning above 150 seconds, critical above 210 seconds,
bounded wrapper-level penalty, and reporting for warning/critical service-age
duration and burden. This treats temporary overload delay as a budget to manage,
not an automatic policy failure.

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

