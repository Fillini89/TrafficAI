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

- Gen11 continuation reached the 3,000,000-step target,
- final Gen11 model and VecNormalize were saved,
- quick compare completed,
- full compare completed,
- use Gen11 as the rescue baseline for planning Gen12.

At completion, the training report showed stable PPO optimization metrics but
mixed raw traffic/fairness signals. The final holdout evaluations are required
before declaring Gen11 a success or failure.

Quick holdout after 3M showed a mixed result:

- Gen11 average `final_total_wait` was slightly better than Gen10.
- Gen11 average `p95_total_wait`, stopped AUC, and mean speed were worse than
  Gen10.
- Gen11 was not an obvious quick-regression disaster, but it did not prove the
  fairness fix.

Next decision point remains the full daily holdout.

Full holdout after 3M showed that Gen11 achieved the main rescue objective:

- Gen10 full-daily starvation collapse was eliminated. Average `final_total_wait`
  dropped from about 2.54M in Gen10 to about 2.22K in Gen11.
- Average `p95_total_wait` dropped from about 2.21M in Gen10 to about 6.57K in
  Gen11.
- Gen11 had the best full-holdout stopped AUC and mean speed among Baseline,
  Gen9, Gen10, and Gen11.
- Daily scenarios improved dramatically versus Gen10 across final wait, p95
  wait, stopped AUC, and speed.

Remaining Gen11 limitations:

- `p95_total_wait` is still worse than the fixed baseline average, so long-tail
  fairness is improved but not solved.
- Phase switching is high, especially on daily scenarios, suggesting a
  throughput/fairness tradeoff that should be refined in Gen12.
- Stress performance is mostly strong, but p95 wait is worse than Gen10 in
  several stress scenarios, except where Gen10 itself collapses.

Conclusion:

Gen11 is a successful anti-starvation rescue generation and a strong evidence
point for the project, but the next generation should focus on reducing p95
tail wait and excessive switching without reintroducing Gen10-style starvation.

## Gen12

The first Gen12 run was a warm-start from the final Gen11 model pair, not a
Gen11 checkpoint continuation and not a scratch architecture run.

Changed for Gen12:

- observation shape remains unchanged,
- stale completed-generation autosaves are ignored by default during startup,
- reward adds bounded `tail_wait` over the worst few lanes,
- starvation begins earlier with a 360-second threshold,
- reward adds bounded `short_phase` penalty for unnecessary short cycling,
- daily long-episode probability rises from 25% to 40%,
- `compare_models.py --tail-regression` evaluates the worst Gen11 p95
  scenarios as a focused regression screen.

Gen12 success criteria:

- preserve Gen11's fix for Gen10-style full-daily starvation collapse,
- reduce average `p95_total_wait` versus Gen11,
- reduce `phase_switch_count` versus Gen11,
- avoid meaningful regression in `final_total_wait`, stopped AUC, and mean
  speed.

Full holdout after Gen12 completed:

- Gen12 improved stopped AUC versus Gen11: about 463,962 versus 496,392.
- Gen12 slightly improved mean speed versus Gen11: about 4.07 versus 4.05.
- Gen12 worsened final wait versus Gen11: about 2,672 versus 2,231.
- Gen12 failed the main p95 objective: about 24,385 versus Gen11's 6,410.
- Daily p95 was the clear failure mode: about 37,228 versus Gen11's 6,441.
- Stress p95 also worsened moderately: about 7,262 versus Gen11's 6,368.

Conclusion:

The first Gen12 is not the next best model despite better aggregate flow metrics. It
appears to have learned to keep the network moving and reduce stopped burden
while allowing severe long-tail waiting in daily scenarios. Keep Gen11 as the
current safe rescue baseline and use Gen12 as evidence that the next reward
iteration must make tail fairness harder to trade away.

Failed Gen12 artifacts were deleted after diagnosis so the project can reuse the
Gen12 name for a rebuilt run from Gen11.

Rebuilt Gen12 plan:

- source remains final Gen11 model plus matching VecNormalize,
- observation shape remains unchanged,
- speed and throughput rewards are gated by fairness debt,
- hard fairness debt zeros flow rewards for the step,
- tail and starvation penalties use convex growth above thresholds,
- `ServiceDebtGuardrailWrapper` can force a debt-serving phase above 120 seconds,
- compare reports include hidden-starvation metrics such as worst-lane wait,
  starved lanes, starvation excess, tail wait, and fairness violation steps.

Rebuilt Gen12 full holdout:

- Average final wait improved versus Gen11: about 1,311 versus 2,701.
- Average p95 wait improved versus Gen11: about 3,406 versus 6,511.
- Gen12 won `p95_total_wait` on all 14 scenarios.
- Gen12 won `mean_total_wait` on all 14 scenarios.
- Hidden-starvation proxies improved strongly: average max worst-lane wait about
  6,539 versus Gen11's 12,395; p95 worst-lane wait about 4,111 versus 6,868.
- Cost: mean speed dropped to about 3.40 versus Gen11's 4.03.
- Cost: stopped AUC worsened to about 515,918 versus Gen11's 490,638.
- Cost: switching rose to about 5,189 versus Gen11's 4,910.

Conclusion:

Rebuilt Gen12 fixed the main p95/final-wait failure and is the current fairness
leader. It is not a pure throughput winner. Next work should decide whether to
recover speed/stopped-AUC while preserving the p95 gains, or accept this as the
fairness-first candidate.

## Known Failure Modes

- Good mean speed can hide lane starvation.
- Good quick holdout can hide 24-hour daily collapse.
- Reward changes without observation shape changes are suitable for fine-tuning.
- Observation shape changes require new model architecture.
- Missing or mismatched `VecNormalize` stats can make a loaded policy behave
  incorrectly.
- Excessive fairness penalties can cause over-switching and reduce throughput.

