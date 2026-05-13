# TrafficAI Training Runbook

Use this file for repeatable training and evaluation on any machine.

## Environment Setup

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Current stable continuation environment:

```text
Python 3.12.x
Stable-Baselines3 2.8.0
PyTorch 2.8.0+cpu
NumPy 1.26.4
Gymnasium 1.2.3
sumo-rl 1.4.5
SUMO 1.26.0
```

Avoid NumPy 2.x for the current training environment. On the Alienware
migration, `numpy==2.4.4` plus newer Torch builds caused native crashes while
loading SB3 PPO zip files.

SUMO must be installed and available to `sumo-rl`/TraCI. If SUMO tools are not
on the Python path, configure the machine-specific environment before training.

## Visual SUMO Playback

Use `test_agent.py` to watch a trained model in SUMO GUI. The script loads the
matching `*_vecnormalize.pkl` file and uses generation-aware control wrappers,
so Gen16 playback uses the same service-age budget/adaptive cadence stack used
in evaluation.

Gen16 on a light daily route:

```powershell
.\venv\Scripts\python.exe test_agent.py --gen 16 --route light --seconds 3600
```

Useful route presets:

```powershell
.\venv\Scripts\python.exe test_agent.py --list-routes
```

Recommended visual sweep:

```powershell
.\venv\Scripts\python.exe test_agent.py --gen 16 --route light --seconds 3600
.\venv\Scripts\python.exe test_agent.py --gen 16 --route heavy --seconds 3600
.\venv\Scripts\python.exe test_agent.py --gen 16 --route stress --seconds 3600
.\venv\Scripts\python.exe test_agent.py --gen 16 --route extreme --seconds 3600
```

## CPU Parallelism

Default training uses 12 SUMO environments. On larger CPUs, test higher values
gradually:

```powershell
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

Keep `TRAFFICAI_NUM_CPU` conservative until startup and several autosaves are
stable. More parallel SUMO workers can improve wall-clock throughput, but can
also increase Windows TraCI/socket pressure.

## Smoke Training

Use smoke mode only for integration checks:

```powershell
$env:TRAFFICAI_SMOKE_TEST="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_SMOKE_TEST -ErrorAction SilentlyContinue
```

Smoke runs are not model-quality evidence.

## Start Gen16 From Gen15

Gen15 is the best aggregate balanced candidate. Gen16 should warm-start from
Gen15 and add service-age budget control without changing observation shape,
PPO architecture, route format, VecNormalize format, or core reward weights:

```powershell
$env:TRAFFICAI_WARM_START_GEN="15"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_STARTUP_CHECK -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_NUM_CPU -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_TORCH_NUM_THREADS -ErrorAction SilentlyContinue
```

Expected output includes:

```text
Warm-starting Gen 16 from Gen 15
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen15_vecnormalize.pkl
Startup check complete for Gen 16
```

Start the full 3M run:

```powershell
$env:TRAFFICAI_WARM_START_GEN="15"
Remove-Item Env:TRAFFICAI_CONTINUE_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_TOTAL_TIMESTEPS -ErrorAction SilentlyContinue
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

Gen16 keeps Gen15 adaptive cadence and adds a service-age budget: warning above
150 seconds, critical above 210 seconds, bounded warning/critical penalty, and
critical debt allowed to break protected hold after min-green.

## Start Gen15 From Gen14

Gen14 is the protected balanced favorite. Gen15 should warm-start from Gen14 and
make only minimal-risk adaptive-cadence tuning:

```powershell
$env:TRAFFICAI_WARM_START_GEN="14"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_STARTUP_CHECK -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_NUM_CPU -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_TORCH_NUM_THREADS -ErrorAction SilentlyContinue
```

Expected output includes:

```text
Warm-starting Gen 15 from Gen 14
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen14_vecnormalize.pkl
Startup check complete for Gen 15
```

Start the full 3M run:

```powershell
$env:TRAFFICAI_WARM_START_GEN="14"
Remove-Item Env:TRAFFICAI_CONTINUE_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_TOTAL_TIMESTEPS -ErrorAction SilentlyContinue
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

Gen15 keeps reward weights unchanged. It uses 70-second adaptive service release,
5.0 queue imbalance release, 120-second hard service threshold, 12-second
protected hold, 130-second worse-emergency override, and -0.075 cadence
suppression penalty. Long daily training probability is 0.50.

## Start Gen14 From Gen12

Gen12 is the protected fairness champion. Gen13 is kept as a diagnostic
smoothness reference, but Gen14 must not use Gen13 as its source policy. Force
the warm-start source explicitly:

```powershell
$env:TRAFFICAI_WARM_START_GEN="12"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_STARTUP_CHECK -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_NUM_CPU -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_TORCH_NUM_THREADS -ErrorAction SilentlyContinue
```

Expected output includes:

```text
Warm-starting Gen 14 from Gen 12
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen12_vecnormalize.pkl
Startup check complete for Gen 14
```

Forbidden output:

```text
Warm-starting Gen 14 from Gen 13
```

Recommended gated trial on the Alienware CPU:

```powershell
$env:TRAFFICAI_WARM_START_GEN="12"
$env:TRAFFICAI_TOTAL_TIMESTEPS="750000"
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

After a successful 750k tail gate, continue the same Gen14 run to 3M from the
latest Gen14 checkpoint:

```powershell
Remove-Item Env:TRAFFICAI_TOTAL_TIMESTEPS -ErrorAction SilentlyContinue
$env:TRAFFICAI_CONTINUE_CHECKPOINT="1"
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

Gen14 uses adaptive cadence: 16-second normal hold suppression, 24-second target
hold, 75-second moderate service-age release, 120-second hard service-age
intervention, 12-second protected hold, and 135-second worse-emergency override.

## Start Gen13 Warm-Start

Gen12 is now the protected fairness champion. Do not delete or overwrite its
artifacts before Gen13 training. The next normal training run should warm-start
`Gen13` from:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Recommended startup check:

```powershell
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_STARTUP_CHECK -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_NUM_CPU -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_TORCH_NUM_THREADS -ErrorAction SilentlyContinue
```

Expected output includes:

```text
Stale autosave ignored for Gen 12
Warm-starting Gen 13 from Gen 12
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen12_vecnormalize.pkl
Startup check complete for Gen 13
```

Start training on the Alienware CPU conservatively:

```powershell
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

Rebuilt Gen13 keeps observation shape unchanged. It treats 90/120 seconds as
lane service-age debt, suppresses non-urgent phase changes before 24 seconds of
actual hold, and keeps a 24-second protected hold after fairness-forced service.

## Start Rebuilt Gen12 Warm-Start

The first Gen12 completed and failed the primary p95 tail-wait objective. Its
model/checkpoint lineage was deleted after diagnosis. The next normal training
run rebuilds `Gen12` from the final Gen11 pair using non-compensable fairness
reward and the service-debt guardrail.

Expected source pair:

```text
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Recommended startup check:

```powershell
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_STARTUP_CHECK -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_NUM_CPU -ErrorAction SilentlyContinue
```

Expected output includes:

```text
Stale autosave ignored for Gen 11
Warm-starting Gen 12 from Gen 11
Loaded VecNormalize stats from: models\ppo_traffic_model_Gen11_vecnormalize.pkl
Startup check complete for Gen 12
```

Forbidden startup output:

```text
Autosave detected for Gen 12
```

If that appears, stale Gen12 checkpoint artifacts are still present and must be
removed before retraining.

Start training:

```powershell
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

If Gen12 is interrupted before a final Gen12 model is saved, the next startup
should resume the Gen12 autosave automatically.

## Continue Historical Gen11 Checkpoint

Gen11 has already reached the 3,000,000-step target. This section is now mostly
for provenance or emergency debugging; do not continue it again unless that is a
deliberate new decision.

Expected files:

```text
checkpoints/ppo_traffic_model_autosave_Gen11_1009596_steps.zip
checkpoints/vecnormalize_latest.pkl
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Run with explicit checkpoint continuation:

```powershell
$env:TRAFFICAI_CONTINUE_CHECKPOINT="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_CONTINUE_CHECKPOINT -ErrorAction SilentlyContinue
```

Correct startup:

```text
Autosave detected for Gen 11. Continuing training with: checkpoints\ppo_traffic_model_autosave_Gen11_1009596_steps.zip
Resuming from step 1009596...
```

If startup says `Warm-starting Gen 12 from Gen 11`, the historical checkpoint
override was not applied.

To check startup without entering the training loop:

```powershell
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_STARTUP_CHECK -ErrorAction SilentlyContinue
Remove-Item Env:TRAFFICAI_NUM_CPU -ErrorAction SilentlyContinue
```

Expected check output includes `Autosave detected for Gen 11`,
`Resuming from step 1009596`, and `Startup check complete for Gen 11`.

## Marathon Mode

On Windows, `marathon.ps1` restarts training after socket-related failures.

```powershell
./marathon.ps1
```

Manual `Ctrl+C` should stop the marathon and not restart training.

## TensorBoard

Start TensorBoard:

```powershell
python -m tensorboard.main --logdir outputs/tensorboard
```

Open:

```text
http://localhost:6006/
```

Key metrics:

- `reward/raw_starved_lanes`
- `reward/raw_starvation_excess`
- `reward/raw_fairness_debt`
- `reward/raw_flow_gate`
- `reward/raw_hard_fairness_debt`
- `reward/raw_worst_lane_wait`
- `reward/raw_tail_wait_mean`
- `reward/raw_phase_hold_seconds`
- `reward/raw_phase_change_penalty`
- `reward/long_green`
- `reward/starvation`
- `reward/tail_wait`
- `reward/phase_change`
- `reward/reward_total`
- `safety/fairness_forced_switches`
- `safety/actual_phase_switches`
- `safety/cadence_suppressed_switches`
- `safety/fairness_guardrail_suppressed_switches`
- `safety/fairness_guardrail_hold_active`
- `safety/adaptive_cadence_releases`
- `safety/requested_service_debt`
- `safety/requested_queue_advantage`
- `safety/max_service_debt`
- `safety/max_service_age`
- `safety/phase_forced_switches`
- `train/approx_kl`
- `train/clip_fraction`
- `train/explained_variance`

Interpretation:

- `raw_starved_lanes` should trend toward 0 or stay low.
- `raw_starvation_excess` should decrease.
- `raw_flow_gate` should drop below 1.0 when fairness debt is active; this is
  expected and means flow rewards are no longer compensating severe tail wait.
- `raw_worst_lane_wait` should decrease.
- `raw_tail_wait_mean` should stay bounded and trend down.
- `max_service_debt` above 120 seconds is an emergency intervention condition,
  not an acceptable steady-state target.
- `raw_phase_hold_seconds` should not stay extremely high.
- `long_green` and `starvation` penalties should move closer to 0 over time.
- `fairness_forced_switches` should occur when service debt is severe, but
  should not be constantly high late in training.
- `phase_forced_switches` should not be constantly high.
- For rebuilt Gen13, service-age debt should stay bounded while actual switch
  count falls versus Gen12.
- For Gen14, adaptive cadence releases should appear when moderate service debt
  or large queue imbalance justifies earlier service; they should not replace
  hard fairness service as the dominant behavior.
- `cadence_suppressed_switches` can appear early; if it remains very high late
  in training, the policy is still requesting twitchy actions.
- `fairness_guardrail_suppressed_switches` can appear when hysteresis prevents
  repeated debt-phase hopping; it should not replace good learned behavior.
- High explained variance is useful but does not prove policy quality.

## Quick Evaluation

After a checkpoint or completed generation:

```powershell
python compare_models.py --models 3
```

Quick output goes to:

```text
outputs/reports/holdout/holdout_eval_summary_quick.csv
outputs/reports/holdout/holdout_eval_dashboard_quick.png
outputs/reports/holdout/holdout_eval_scorecard_quick.csv
outputs/reports/holdout/holdout_eval_report_quick.md
outputs/reports/comparisons/*.png
```

Use quick evaluation to detect obvious regressions. Do not use it as final
proof of success.

On machines with Matplotlib/Tcl issues, use:

```powershell
python compare_models.py --models 3 --no-plots
```

To rebuild investor-facing reports from an existing quick CSV without rerunning
SUMO:

```powershell
python compare_models.py --report-only quick
```

## Tail Regression Evaluation

Gen15 should be checked against the worst p95 scenarios before a full holdout:

```powershell
python compare_models.py --tail-regression --models 5 --no-plots --jobs 4
python compare_models.py --tail-regression --full --models 5 --no-plots --jobs 4
```

For Gen16, `--models 5` compares protected Gen12, Gen13 smoothness reference,
protected Gen14 balanced favorite, Gen15 aggregate candidate, new Gen16, and the
fixed baseline.

For a narrower rebuilt Gen13 gate, compare only protected Gen12, rebuilt Gen13,
and the fixed baseline:

```powershell
python compare_models.py --tail-regression --full --models 2 --no-plots --jobs 4
```

Tail-regression outputs use separate suffixes:

```text
outputs/reports/holdout/holdout_eval_summary_tail_quick.csv
outputs/reports/holdout/holdout_eval_summary_tail_full.csv
```

Presentation reports can be rebuilt without SUMO:

```powershell
python compare_models.py --report-only tail_quick
python compare_models.py --report-only tail_full
```

## Full Evaluation

Run after a completed generation reaches its target steps:

```powershell
python compare_models.py --full --models 5 --no-plots
```

Parallel full evaluation is opt-in. Start conservatively on Windows:

```powershell
python compare_models.py --full --models 5 --no-plots --jobs 4
```

If TraCI stays stable, try faster settings:

```powershell
python compare_models.py --full --models 5 --no-plots --jobs 8
python compare_models.py --full --models 5 --no-plots --jobs 12
```

For the lightest full run, skip per-run metrics capture:

```powershell
python compare_models.py --full --models 5 --no-plots --jobs 8 --no-run-metrics
```

Full output goes to:

```text
outputs/reports/holdout/holdout_eval_summary_full.csv
outputs/reports/holdout/holdout_eval_dashboard_full.png
outputs/reports/holdout/holdout_eval_scorecard_full.csv
outputs/reports/holdout/holdout_eval_report_full.md
```

Gen14 success criteria:

- `p95_total_wait` is no worse than Gen12 by more than about 5%,
- `final_total_wait` is no worse than Gen12 by more than about 10%,
- hidden-starvation metrics remain close to Gen12,
- actual `phase_switch_count` improves by at least 30% versus Gen12,
- mean speed should remain close to Gen12 or improve,
- speed/stopped-AUC improvements are welcome but secondary to fairness.

Gen14 3M full holdout outcome:

- Passed the tolerance gates versus Gen12: p95 was about 4.2% worse and final
  wait about 5.8% worse.
- Passed the smoothness gate: actual switches dropped by about 61% versus
  Gen12.
- Improved mean speed and stopped burden versus Gen12.
- Did not replace Gen12 as the pure p95 champion; use Gen12 as the protected
  fairness baseline and Gen14 as the balanced candidate.

Gen15 promotion criteria:

- p95 and final wait better than Gen14, ideally at or below Gen12,
- stopped AUC and mean speed no worse than Gen14 by more than about 2%,
- actual switches no worse than Gen14 by more than about 10%,
- max service age closer to 120-135 seconds than Gen14's about 146 seconds,
- no catastrophic daily scenario regression.

Gen15 3M full holdout outcome:

- Passed aggregate flow gates versus Gen14: final wait down about 23.7%, p95
  down about 0.7%, stopped AUC down about 1.4%, and mean speed up about 0.9%.
- Passed smoothness tolerance versus Gen14: actual switches rose about 4.7%,
  still within the 10% limit and far below Gen12.
- Failed the service-age gate: max service age rose from about 149 seconds to
  about 192 seconds.
- Treat Gen15 as the best aggregate balanced candidate, but keep Gen14 as the
  safer service-age balanced baseline until the service-age tail is fixed.

Gen16 promotion criteria:

- `service_age_over_150_steps`, `service_age_over_210_steps`,
  `service_age_over_150_auc`, and `max_service_age` improve versus Gen15,
- final wait no worse than Gen15 by more than about 5%,
- stopped AUC and mean speed no worse than Gen15 by more than about 3%,
- actual switches no worse than Gen15 by more than about 10%,
- no catastrophic daily p95/final-wait regression.

Gen16 3M full holdout outcome:

- Passed aggregate performance gates versus Gen15: final wait, p95 wait,
  stopped AUC, mean speed, and hidden-starvation proxies improved.
- Passed smoothness tolerance: actual switches stayed essentially flat versus
  Gen15 and far below Gen12.
- Failed the service-age budget gate: max service age, warning-zone steps,
  critical-zone steps, and warning-zone burden all worsened versus Gen15.
- Treat Gen16 as the leading aggregate-performance candidate, while Gen14
  remains the safer service-age balanced baseline.

The Gen13 scorecard/report also includes smoothing metrics:

- average seconds between switches,
- estimated switches per hour,
- policy action switches versus actual executed signal switches,
- fairness-forced switch count,
- guardrail-suppressed switch count,
- average and p95 phase-hold duration.

Use `--control-profile generation` by default. This evaluates Gen12 under its
trained legacy control stack, rebuilt Gen13 under hard service-age cadence, and
Gen14 under the adaptive service-age cadence. `--control-profile current` is
only for stress-testing all supported models under the newest wrappers.

## Output Layout

```text
outputs/
  reports/
    comparisons/
    holdout/
    training/
  runtime/
    sumo/
      train/
      eval/
    eval_callback/
  tensorboard/
```

Runtime data is ignored by git. Compact reports and final model artifacts are
intended to be portable.

