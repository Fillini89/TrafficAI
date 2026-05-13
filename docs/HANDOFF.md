# TrafficAI Handoff

This is the first file a new Codex session should read.

## Current Situation

TrafficAI is training a PPO-based traffic-light controller in SUMO. The current
protected champion is rebuilt Gen12, which is a fine-tune from Gen11.

Gen10 was strong on stress scenarios but failed full 24-hour daily evaluation
because of lane starvation. Some approaches could wait for extreme periods while
aggregate speed and stopped metrics still looked acceptable.

Gen11 was created to fix this without changing observation shape.

## Current Objective

Gen11 continuation from checkpoint reached the 3,000,000-step target, saved the
final model pair, and completed quick/full holdout evaluation.

The first Gen12 warm-start from the final Gen11 model pair completed and failed
the main tail-fairness objective. Its artifacts were deleted so the next
training run can reuse the Gen12 name while warm-starting again from Gen11.

Rebuilt Gen12 completed training and full holdout. It is the new best model for
the primary p95/final-wait fairness objective, but it trades away speed and
stopped-AUC versus Gen11 and switches more often than desired.

The first Gen13 smoothing attempt failed full holdout: it improved speed and
stopped burden, but worsened p95/final wait and increased switching. Its model
lineage was deleted so the project can reuse the Gen13 name.

Rebuilt Gen13 completed full holdout after 3M. It dramatically reduced actual
switching, but it did not preserve Gen12 p95/final-wait fairness. Treat it as a
smoothness success and fairness regression, not as the new champion.

Gen14 completed full 3M training and full holdout. It warm-started explicitly
from protected Gen12, not from Gen13, and added adaptive smoothness rather than
hard Gen13 cadence. Gen14 is not the pure p95 champion, but it is the strongest
balanced candidate so far: near-Gen12 p95/final wait, much smoother signals,
lower stopped burden, and higher mean speed.

The active experiment is Gen15: warm-start from protected final Gen14 with
minimal-risk micro-tuning. Gen15 keeps the reward weights and observation shape
unchanged. It only nudges adaptive service-age cadence earlier to reduce Gen14's
remaining daily final-wait outliers and max service age while preserving speed
and smoothness.

Gen15 completed full 3M training and full holdout. It improved aggregate final
wait, p95 wait, stopped burden, and mean speed versus Gen14, and actual switches
stayed within tolerance. However, max service age worsened materially, so Gen15
is not an unconditional replacement. Treat Gen15 as the best aggregate balanced
candidate, Gen14 as the safer service-age balanced baseline, and Gen12 as the
pure p95/fairness champion.

The active experiment is Gen16: warm-start from protected final Gen15 and keep
Gen15's aggregate-flow gains while adding service-age budget control. Gen16 uses
warning/critical service-age zones rather than a hard 120-second cap: warning
above 150 seconds, critical above 210 seconds, bounded penalty, and critical
debt allowed to break protected hold after min-green.

Gen16 completed full 3M training and full holdout. It is the strongest aggregate
performance candidate so far: final wait, p95 wait, stopped burden, mean speed,
and hidden-starvation proxies improved versus Gen15 while actual switching stayed
flat. The service-age budget goal did not pass: max service age and warning /
critical budget metrics worsened versus Gen15.

## Expected Continuation Files

```text
checkpoints/ppo_traffic_model_autosave_Gen11_1009596_steps.zip
checkpoints/vecnormalize_latest.pkl
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

## Correct Gen12 Training Startup

Current default startup should ignore stale autosaves for generations that
already have a final model artifact and should warm-start a new generation from
the latest final model pair:

```powershell
python train_agent.py
```

Expected Gen12 output now includes:

```text
Stale autosave ignored for Gen 11
Warm-starting Gen 12 from Gen 11: models\ppo_traffic_model_Gen11.zip
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen11_vecnormalize.pkl
```

The failed Gen12 final artifacts and autosaves should be absent. If a new
rebuilt Gen12 autosave exists and no final Gen12 model exists yet, startup should
resume that Gen12 autosave. To force a historical Gen11 checkpoint continuation,
set `TRAFFICAI_CONTINUE_CHECKPOINT=1`.

## Correct Gen13 Training Startup

Gen12 is now protected and should not be deleted. Normal startup should
warm-start Gen13 from the final Gen12 pair:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Expected Gen13 startup check:

```text
Stale autosave ignored for Gen 12
Warm-starting Gen 13 from Gen 12: models\ppo_traffic_model_Gen12.zip
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen12_vecnormalize.pkl
Startup check complete for Gen 13
```

Gen13 keeps the observation shape and PPO architecture unchanged. It changes
reward/control behavior only:

- 90/120 seconds now means lane service-age debt, not aggregate lane-wait sum.
- Flow reward gating remains non-compensable under hard fairness debt.
- Reward is rebased toward Gen12; hard cadence lives outside the policy.
- Non-urgent phase changes are suppressed before 24 seconds actual hold.
- Hard service-age debt at 120 seconds can override cadence after min-green.
- A fairness-forced service gets a 24-second protected hold unless another lane
  reaches a worse emergency age around 150 seconds.
- Reports separate policy action changes from actual executed signal changes.

## Correct Gen14 Training Startup

Gen14 must warm-start from Gen12 even though Gen13 exists. Use:

```powershell
$env:TRAFFICAI_WARM_START_GEN="12"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
```

Expected output:

```text
Warm-starting Gen 14 from Gen 12: models\ppo_traffic_model_Gen12.zip
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen12_vecnormalize.pkl
Startup check complete for Gen 14
```

Forbidden output:

```text
Warm-starting Gen 14 from Gen 13
```

Gen14 control intent:

- source policy and VecNormalize stats are Gen12;
- Gen12 and Gen13 artifacts stay intact;
- 90/120 seconds still means lane service-age debt;
- normal non-urgent changes are suppressed only before about 16 seconds actual
  hold;
- target hold is about 24 seconds;
- service age around 75 seconds or high queue imbalance can release cadence
  after min-green;
- hard service-age debt at 120 seconds still forces service after min-green;
- protected hold after fairness-forced service is about 12 seconds;
- worse emergency override is around 135 seconds.

## Correct Gen15 Training Startup

Gen15 must warm-start from Gen14 and keep Gen14 artifacts intact:

```powershell
$env:TRAFFICAI_WARM_START_GEN="14"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
```

Expected output:

```text
Warm-starting Gen 15 from Gen 14: models\ppo_traffic_model_Gen14.zip
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen14_vecnormalize.pkl
Startup check complete for Gen 15
```

Gen15 micro-tuning:

- reward weights unchanged;
- observation/PPO architecture unchanged;
- long daily probability is 0.50;
- adaptive service release is 70 seconds;
- queue imbalance release is 5.0;
- hard service threshold stays 120 seconds;
- protected hold stays 12 seconds;
- worse emergency override is 130 seconds;
- cadence suppression penalty is -0.075.

## Correct Gen16 Training Startup

Gen16 must warm-start from Gen15 and keep Gen15 artifacts intact:

```powershell
$env:TRAFFICAI_WARM_START_GEN="15"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
```

Expected output:

```text
Warm-starting Gen 16 from Gen 15: models\ppo_traffic_model_Gen15.zip
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen15_vecnormalize.pkl
Startup check complete for Gen 16
```

Gen16 service-age budget:

- adaptive cadence defaults remain Gen15-like;
- warning zone starts at 150 seconds;
- critical zone starts at 210 seconds;
- budget penalty is bounded and clipped at 0.25 per step;
- reports include `service_age_over_150_steps`,
  `service_age_over_210_steps`, and `service_age_over_150_auc`.

## Historical Gen11 Checkpoint Startup

Gen11 has already reached the 3,000,000-step target. Do not resume training
again unless the user explicitly chooses a new continuation or Gen12
warm-start experiment.

For historical/debug context, the correct Gen11 checkpoint startup was:

```powershell
python train_agent.py
```

Correct output:

```text
Autosave detected for Gen 11. Continuing training with: checkpoints\ppo_traffic_model_autosave_Gen11_1009596_steps.zip
Resuming from step 1009596...
```

If output says `Warm-starting Gen 12 from Gen 11`, that is now correct for the
current Gen12 objective.

On the Alienware Area-51 migration, `PPO.load(...)` crashed with Windows access
violation `-1073741819`. The project now uses `sb3_compat.load_ppo_compat(...)`
for training/evaluation loads, and the local venv was aligned to
`numpy==1.26.4` and `torch==2.8.0+cpu`.

Startup check after the compatibility fix succeeded with:

```text
Autosave detected for Gen 11...
Resuming from step 1009596...
Startup check complete for Gen 11. Completed: 1009596, remaining: 1990404.
```

Gen11 training completion saved:

```text
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Training report last logged step: `3008444`. Training metrics were mixed:
PPO stability looked sane, but raw starvation/wait metrics still require
holdout judgment.

Quick holdout after 3M completed and wrote:

```text
outputs/reports/holdout/holdout_eval_summary_quick.csv
outputs/reports/holdout/holdout_eval_dashboard_quick.png
outputs/reports/holdout/holdout_eval_scorecard_quick.csv
outputs/reports/holdout/holdout_eval_report_quick.md
```

Quick aggregate averages:

```text
Baseline final_wait=2503.25 p95=2483.25 speed=3.80 stopped_auc=134625.5
Gen9     final_wait=2246.75 p95=3183.00 speed=5.19 stopped_auc=100002.75
Gen10    final_wait=1630.25 p95=2266.25 speed=5.75 stopped_auc=84498.00
Gen11    final_wait=1532.25 p95=2996.50 speed=5.49 stopped_auc=88167.75
```

Interpretation: Gen11 quick is slightly better than Gen10 on average final wait,
but worse on p95 wait, stopped AUC, and mean speed. Full daily evaluation is now
the deciding test.

`compare_models.py --report-only quick|full` can rebuild the investor-facing
dashboard, scorecard, and Markdown report from an existing holdout summary CSV.

`compare_models.py` now supports opt-in parallel holdout evaluation via
`--jobs N`. Default remains serial (`--jobs 1`). On Windows, start with
`--jobs 4`; increase only if TraCI remains stable.

## What To Watch During Training

The fairness fix is working only if:

- `reward/raw_starved_lanes` decreases or stays low,
- `reward/raw_starvation_excess` decreases,
- `reward/raw_worst_lane_wait` decreases,
- `reward/raw_phase_hold_seconds` does not stay huge,
- `reward/long_green` moves closer to 0,
- `reward/starvation` moves closer to 0,
- `safety/phase_forced_switches` is not constantly high,
- PPO stability metrics remain sane.

## Evaluation Result

For the next fresh full evaluation, prefer the conservative parallel command:

```powershell
python compare_models.py --full --models 3 --no-plots --jobs 4
```

Full holdout after Gen11 3M completed and wrote:

```text
outputs/reports/holdout/holdout_eval_summary_full.csv
outputs/reports/holdout/holdout_eval_dashboard_full.png
outputs/reports/holdout/holdout_eval_scorecard_full.csv
outputs/reports/holdout/holdout_eval_report_full.md
```

Gen11 full aggregate averages:

```text
Baseline final_wait=3076.64 p95=4492.50 speed=2.70 stopped_auc=817530.50
Gen9     final_wait=2002.29 p95=187959.07 speed=3.51 stopped_auc=518548.07
Gen10    final_wait=2542195.29 p95=2206479.43 speed=3.24 stopped_auc=909056.36
Gen11    final_wait=2222.50 p95=6566.93 speed=4.02 stopped_auc=494316.07
```

Interpretation: Gen11 fixed the Gen10 full-daily starvation collapse by reducing
Gen10 final wait and p95 wait by about 99.9% overall. Gen11 also achieved the
best stopped AUC and best mean speed on the full holdout. Remaining weakness:
Gen11 p95 wait is still worse than fixed baseline on average and phase switching
is high, so Gen11 is a successful rescue generation but not the final production
policy.

Original success condition:

Full daily `final_total_wait` and `p95_total_wait` should no longer show the
Gen10 starvation failure. Stress performance should remain competitive with
Gen10 and clearly better than the fixed baseline.

Status: Gen11 passed the main anti-starvation rescue condition versus Gen10, but
did not solve long-tail p95 fairness versus the fixed baseline.

Gen12 full aggregate averages:

```text
Baseline final_wait=3241.29 p95=4481.93 speed=2.70 stopped_auc=815993.29
Gen10    final_wait=1953389.71 p95=1641851.50 speed=3.17 stopped_auc=835563.64
Gen11    final_wait=2230.64 p95=6409.86 speed=4.05 stopped_auc=496392.43
Gen12    final_wait=2672.21 p95=24385.29 speed=4.07 stopped_auc=463961.93
```

Interpretation: failed Gen12 improved stopped AUC by about 6.5% versus Gen11 and
slightly improved mean speed, but it worsened final wait by about 19.8% and p95
wait by about 280%. Daily p95 is the failure: Gen12 daily average p95 was about
37.2K versus Gen11's 6.44K. Do not continue failed Gen12. Its final model pair,
best model directory, Gen12 autosaves, and `checkpoints/vecnormalize_latest.pkl`
were removed so the next run rebuilds Gen12 from Gen11.

Rebuilt Gen12 changes:

- flow rewards are gated by fairness debt,
- severe fairness debt zeros speed/throughput reward for that step,
- tail/starvation penalties use convex growth after thresholds,
- service-debt guardrail can force service when lane wait exceeds 120 seconds,
- compare reports include hidden-starvation metrics.

Rebuilt Gen12 full aggregate averages:

```text
Baseline final_wait=3291.43 p95=4497.21 speed=2.70 stopped_auc=817867.43
Gen10    final_wait=2539105.00 p95=2211394.43 speed=3.29 stopped_auc=918717.79
Gen11    final_wait=2700.64 p95=6510.79 speed=4.03 stopped_auc=490637.50
Gen12    final_wait=1310.64 p95=3405.79 speed=3.40 stopped_auc=515917.86
```

Interpretation: rebuilt Gen12 improved final wait by about 51.5% and p95 wait by
about 47.7% versus Gen11, and it beat every agent on p95 in all 14 scenarios.
It also reduced hidden-starvation proxies: average max worst-lane wait dropped
from about 12,395 to 6,539, and p95 worst-lane wait from about 6,868 to 4,111.
Cost: mean speed fell about 15.7% versus Gen11, stopped AUC worsened about 5.2%,
and switching increased slightly. Treat Gen12 as the current fairness winner,
not as a throughput winner.

## If The User Is Ambiguous

The user may say "continue training", "run the model", or "resume Gen11" loosely.
Clarify by inspecting local files and command output. Do not assume a new
generation is desired.

Default assumption right now:

Treat rebuilt Gen12 as the protected pure p95/fairness leader. Treat Gen14 as
the safer service-age balanced baseline. Treat Gen16 as the leading aggregate
performance candidate, but not a complete service-age fix.

