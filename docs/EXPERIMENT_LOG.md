# TrafficAI Experiment Log

This file is a compact memory of important generations, failures, and lessons.

## Gen8 / Gen9

Gen8 and Gen9 used the older observation format:

- density,
- queue,
- speed.

Gen9 was trained for roughly 3,000,000 steps and performed well under stress and
incident conditions. It was strong against queue growth during rush hour and
Chaos Monkey breakdowns.

Limitations:

- weaker observation richness,
- less explicit fairness instrumentation,
- no VecNormalize-based Gen10 stack,
- not directly compatible with Gen10/Gen11 observation shape.

## Gen10

Gen10 introduced the quality-first stack:

- richer camera-compatible observation,
- `VecNormalize`,
- PPO hyperparameter settings,
- reward component logging,
- TensorBoard,
- route pools and domain randomization,
- quick and full holdout evaluation.

Gen10 strengths:

- strong stress performance,
- better mean speed,
- lower stopped AUC in many stress scenarios,
- good robustness under incident pressure.

Gen10 failure:

- full 24-hour daily evaluation exposed starvation.
- Some lanes/approaches could be ignored for very long periods.
- `total_wait` could explode into extremely large values while some aggregate
  metrics still looked acceptable.

Lesson:

Quick 4-hour evaluation is not enough. Full daily evaluation is required for
fairness and starvation detection.

## Gen11

Gen11 is a fine-tune from Gen10 without changing observation shape.

Added or changed:

- stronger `worst_lane` penalty,
- new `starvation` reward component,
- new `starved_lanes` component,
- new `long_green` component,
- raw metrics for starvation and phase holding,
- `PhaseSafetyWrapper` with max-green guardrails,
- curriculum probability for long daily episodes.

The goal is to preserve Gen10 stress strength while fixing full-daily
starvation.

At approximately 1,009,596 steps, quick evaluation showed:

- daily quick performance was worse than Gen10 on mean wait and p95 wait,
- stress performance was mostly preserved,
- stopped AUC and max stopped vehicles remained competitive,
- the model appeared to be in active fairness re-learning rather than fully
  converged.

Current plan:

- continue Gen11 to 3,000,000 steps,
- run quick compare,
- then run full compare,
- judge success primarily on full-daily starvation metrics.

## Known Failure Modes

- Good mean speed can hide lane starvation.
- Good quick holdout can hide 24-hour daily collapse.
- Reward changes without observation shape changes are suitable for fine-tuning.
- Observation shape changes require new model architecture.
- Missing or mismatched `VecNormalize` stats can make a loaded policy behave
  incorrectly.
- Excessive fairness penalties can cause over-switching and reduce throughput.

