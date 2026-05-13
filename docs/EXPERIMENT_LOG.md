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

## Gen13

Gen13 is planned as a conservative warm-start from the protected rebuilt Gen12
final pair:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Goal:

- preserve Gen12 p95/final-wait dominance,
- reduce signal jitter and `phase_switch_count`,
- keep hidden-starvation metrics close to Gen12,
- avoid returning to Gen11/Gen10 flow-first behavior.

Implemented before training:

- observation shape remains unchanged,
- Gen12 fairness gates remain at 90 seconds soft debt and 120 seconds hard
  debt,
- `short_phase` weight increased from 1.4 to 2.2,
- `short_phase_target`/norm increased from 28 seconds to 36 seconds,
- new small `phase_change` penalty discourages non-forced switches,
- service-debt guardrail keeps a 24-second protected hold after a
  fairness-forced switch unless a clearly worse debt appears,
- guardrail logs suppressed switches and hold activity,
- compare reports now include signal smoothness metrics.

Acceptance:

- average `p95_total_wait` no worse than Gen12 by more than about 5%,
- `final_total_wait` remains better than Gen11,
- hidden-starvation proxies remain near Gen12,
- `phase_switch_count` improves by roughly 15-25% versus Gen12,
- speed/stopped-AUC are secondary and cannot justify fairness regression.

First Gen13 result:

- Full holdout failed the objective.
- Gen13 improved speed and stopped burden versus the retested Gen12, but
  worsened average `p95_total_wait`, worsened final wait, and increased
  switching on every scenario.
- The failure diagnosis is that reward-only smoothing was too weak and many
  switch penalties were waived while aggregate lane-wait debt kept emergency
  fairness active.
- The failed final pair, best model directory, Gen13 autosaves, and latest
  checkpoint VecNormalize were deleted so the project can reuse the Gen13 name.

Rebuilt Gen13 plan:

- source remains protected rebuilt Gen12 final pair,
- observation shape remains unchanged,
- 90/120 seconds now means lane service-age debt rather than aggregate lane-wait
  sum,
- normal non-urgent switches are suppressed before 24 seconds actual hold,
- hard service-age debt at 120 seconds can override cadence after min-green,
- a forced fairness service receives a 24-second protected hold unless another
  lane reaches a worse emergency age around 150 seconds,
- compare reports separate policy action switches from actual executed signal
  switches,
- first run should be a shorter gated trial before spending a full 3M steps.

Rebuilt Gen13 full holdout after 3M:

- Average actual phase switches improved strongly versus Gen12: about 2,055
  versus 8,030, and Gen13 won actual switching on all 14 scenarios.
- Average seconds between actual switches improved from about 6.8s to 26.8s.
- The cadence guardrail worked mechanically: about 6,414 policy switch requests
  were suppressed on average, and max service-age stayed near the intended
  120-150s emergency band.
- Gen13 did not preserve Gen12 fairness: average p95 wait worsened from about
  3,415 to 4,189, final wait worsened from about 1,392 to 2,686, and Gen13 lost
  p95 wait on all 14 scenarios.
- Hidden-starvation proxies also regressed: p95 worst-lane wait worsened from
  about 4,132 to 8,544, and fairness violation steps increased.

Conclusion:

Rebuilt Gen13 solved jitter as a control problem but made the signal too
conservative for Gen12-level fairness/throughput. The next iteration should keep
actual cadence reporting and service-age semantics, but relax or make adaptive
the hard cadence so urgent moderate debt can be served before queues accumulate.

## Gen14

Gen14 is planned as a new generation warm-started from protected Gen12, not from
Gen13:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Reason:

- Gen12 remains the p95/final-wait and hidden-starvation champion.
- Gen13 proved that service-age cadence can reduce actual switching, but its
  hard cadence lost Gen12 fairness on every p95 scenario.
- Gen14 should be "Gen12 + adaptive smoothness" rather than a continuation of
  Gen13 behavior.

Implemented for Gen14:

- `TRAFFICAI_WARM_START_GEN=12` forces Gen14 startup from Gen12 even when Gen13
  final artifacts exist.
- Observation shape, PPO architecture, route format, and VecNormalize format
  remain unchanged.
- Gen12 non-compensable fairness reward remains the core objective.
- Service-age debt remains the driver-facing fairness signal: 90 seconds soft,
  120 seconds hard.
- Normal non-urgent switches are suppressed before about 16 seconds actual hold,
  not Gen13's 24 seconds.
- Target hold is about 24 seconds, not Gen13's 32 seconds.
- Moderate service age around 75 seconds or high queue imbalance can release
  cadence after min-green.
- Fairness-forced protected hold is reduced to about 12 seconds.
- Worse emergency override is around 135 seconds.

Acceptance:

- average `p95_total_wait` no worse than Gen12 by more than about 5%,
- `final_total_wait` no worse than Gen12 by more than about 10%,
- no catastrophic daily collapse,
- actual phase switches improve at least 30% versus Gen12,
- mean speed stays close to Gen12 or improves,
- `max_service_age` stays around or below the 120-135 second emergency band.

Gen14 full holdout after 3M:

- Average p95 wait was 3,531 versus Gen12's 3,390: about 4.2% worse, inside the
  5% acceptance tolerance.
- Average final wait was 1,450 versus Gen12's 1,371: about 5.8% worse, inside
  the 10% acceptance tolerance.
- Average stopped burden improved versus Gen12: about 501,800 versus 512,901.
- Mean speed improved strongly versus Gen12: about 3.69 versus 3.41, and Gen14
  won mean speed on all 14 scenarios.
- Actual phase switches dropped from about 8,033 in Gen12 to about 3,108 in
  Gen14: about 61% fewer switches, passing the smoothness target.
- Gen14 won final wait on 8 of 14 scenarios and stopped AUC on 13 of 14
  scenarios.
- Gen12 still won p95 wait on 10 of 14 scenarios, so Gen12 remains the pure
  fairness champion.
- Gen14 improved p95 versus Gen13 on every scenario, confirming that adaptive
  cadence fixed most of the Gen13 fairness regression.
- Hidden starvation was mixed: Gen14 improved versus Gen13 but remained worse
  than Gen12 on worst-lane wait proxies.
- Max service age averaged about 146 seconds, slightly above the desired
  120-135 second emergency band.

Conclusion:

Gen14 is a successful balanced candidate: it preserves most Gen12 fairness while
recovering speed and greatly reducing real signal jitter. Gen12 remains the
protected fairness champion; Gen14 is the best current tradeoff candidate for a
more investor-friendly smooth/high-speed controller. The next iteration should
focus on shaving Gen14's worst daily final-wait outliers and reducing max
service age toward 120-135 seconds without returning to Gen12-level jitter.

## Gen15

Gen15 is planned as a protected continuation from final Gen14:

```text
models/ppo_traffic_model_Gen14.zip
models/ppo_traffic_model_Gen14_vecnormalize.pkl
```

Reason:

- Gen14 is the current favorite balanced candidate.
- Gen12 remains the pure p95/fairness champion, but Gen14 has the better
  product balance: higher speed, lower stopped burden, and far fewer actual
  switches.
- The remaining Gen14 weaknesses are modest p95/final-wait gap versus Gen12,
  daily final-wait outliers, and max service age around 146 seconds.

Implemented for Gen15:

- `TRAFFICAI_WARM_START_GEN=14` forces Gen15 startup from Gen14.
- Observation shape, PPO architecture, route format, VecNormalize format, and
  reward weights remain unchanged.
- Long daily training probability rises slightly from 0.45 to 0.50.
- Adaptive service-age release lowers from 75 seconds to 70 seconds.
- Queue imbalance release lowers from 6.0 to 5.0.
- Hard service threshold stays 120 seconds.
- Protected hold stays 12 seconds.
- Worse emergency override lowers from 135 seconds to 130 seconds.
- Cadence suppression penalty strengthens from -0.05 to -0.075.

Acceptance:

- p95 and final wait should improve versus Gen14 and ideally match or beat
  Gen12,
- stopped AUC and mean speed should not regress versus Gen14 by more than about
  2%,
- actual switches should not regress versus Gen14 by more than about 10%,
- max service age should move closer to the 120-135 second band,
- no daily scenario should show catastrophic p95/final-wait collapse.

Gen15 full holdout after 3M:

- Average final wait improved versus Gen14: about 1,407 versus 1,845, a 23.7%
  reduction.
- Average p95 wait improved slightly versus Gen14: about 3,534 versus 3,559,
  a 0.7% reduction.
- Average stopped burden improved versus Gen14: about 492,200 versus 498,988.
- Mean speed improved versus Gen14: about 3.72 versus 3.68.
- Actual switches increased moderately versus Gen14: about 3,239 versus 3,095,
  a 4.7% increase, still within the 10% tolerance and far below Gen12's 8,024.
- Gen15 won stopped AUC on 8 of 14 scenarios and mean speed on 8 of 14
  scenarios.
- Gen15 won final wait on 4 scenarios; Gen12 won 5, Gen14 won 3, and Gen13 won
  2 in this retest.
- Gen15 did not catch Gen12 on p95: Gen12 still won p95 on 10 of 14 scenarios.
- Hidden-starvation results were mixed: max starvation excess improved versus
  Gen14, but worst-lane wait proxies worsened.
- Max service age regressed from Gen14's about 149 seconds to about 192 seconds,
  failing the intended service-age improvement gate.

Conclusion:

Gen15 is stronger than Gen14 on aggregate flow and final-wait performance, and
it preserves most of Gen14's smoothness. However, it is not an unconditional
promotion because service-age tails worsened materially. Treat Gen15 as the best
aggregate balanced candidate, Gen14 as the safer service-age balanced baseline,
and Gen12 as the protected pure p95/fairness champion. The next iteration should
keep Gen15's flow gains while restoring max service age toward 120-135 seconds.

## Gen16

Gen16 is planned as a protected continuation from final Gen15:

```text
models/ppo_traffic_model_Gen15.zip
models/ppo_traffic_model_Gen15_vecnormalize.pkl
```

Reason:

- Gen15 is the best aggregate-performance candidate so far.
- Gen15's main flaw is not aggregate waiting or speed; it is service-age tail
  risk, with max service age around 192 seconds.
- The target philosophy is a service-age budget, not a harsh 120-second cap:
  short warning-zone delays can be acceptable under load, but persistent or
  critical delays should become expensive.

Implemented for Gen16:

- `TRAFFICAI_WARM_START_GEN=15` forces Gen16 startup from Gen15.
- Observation shape, PPO architecture, route format, VecNormalize format, and
  core reward weights remain unchanged.
- Gen15 adaptive cadence defaults remain: 70-second adaptive release, 5.0 queue
  imbalance release, 16-second min hold, 24-second target hold, 12-second
  protected hold, and 120-second hard service threshold.
- Service-age budget zones are added in the wrapper: warning above 150 seconds,
  critical above 210 seconds.
- Warning/critical service-age excess adds a bounded wrapper-level penalty,
  clipped at 0.25 per step.
- Critical service-age debt can break protected hold after min-green.
- Reports add `service_age_over_150_steps`, `service_age_over_210_steps`, and
  `service_age_over_150_auc`.

Acceptance:

- service-age budget metrics should improve versus Gen15,
- final wait should not regress versus Gen15 by more than about 5%,
- stopped AUC and mean speed should not regress by more than about 3%,
- actual switches should not regress by more than about 10%,
- no daily scenario should show catastrophic p95/final-wait collapse.

Gen16 full holdout after 3M:

- Average final wait improved strongly versus Gen15: about 1,312 versus 1,790.
- Average p95 wait improved versus Gen15: about 3,440 versus 3,543, nearly
  matching Gen12's 3,429.
- Average stopped burden improved versus Gen15: about 479,275 versus 492,589,
  and Gen16 won stopped AUC on 11 of 14 scenarios.
- Mean speed improved versus Gen15: about 3.78 versus 3.71, and Gen16 won mean
  speed on 9 of 14 scenarios.
- Actual switches were essentially stable versus Gen15: about 3,230 versus
  3,244, far below Gen12's about 8,029.
- Hidden-starvation proxies improved versus Gen15 on average: worst-lane wait,
  p95 worst-lane wait, starvation excess, and fairness violation steps all
  moved in the right direction.
- Service-age budget did not improve: max service age rose from about 197s to
  about 208s, warning-zone steps rose from about 49 to 65, critical-zone steps
  rose from about 1 to 4, and warning-zone burden rose from about 3,358 to
  5,373.

Conclusion:

Gen16 is the strongest aggregate-performance model so far and is close to Gen12
on p95 while keeping Gen14/Gen15-like smoothness and much better speed. The
service-age budget penalty was too weak or too indirect to reduce rare
service-age tails. Treat Gen16 as the leading performance candidate, Gen14 as
the safer service-age balanced baseline, and Gen12 as the pure p95 champion. The
next iteration should either strengthen the service-age budget more directly or
evaluate whether rare 200s+ service-age events are acceptable under overload.

## Known Failure Modes

- Good mean speed can hide lane starvation.
- Good quick holdout can hide 24-hour daily collapse.
- Reward changes without observation shape changes are suitable for fine-tuning.
- Observation shape changes require new model architecture.
- Missing or mismatched `VecNormalize` stats can make a loaded policy behave
  incorrectly.
- Excessive fairness penalties can cause over-switching and reduce throughput.

