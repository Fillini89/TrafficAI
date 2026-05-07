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
- `reward/long_green`
- `reward/starvation`
- `reward/tail_wait`
- `reward/reward_total`
- `safety/fairness_forced_switches`
- `safety/max_service_debt`
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

Gen12 should be checked against the worst Gen11 p95 scenarios before a full
holdout:

```powershell
python compare_models.py --tail-regression --models 3 --no-plots --jobs 4
python compare_models.py --tail-regression --full --models 3 --no-plots --jobs 4
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

Run after Gen11 reaches 3,000,000 steps:

```powershell
python compare_models.py --full --models 3 --no-plots
```

Parallel full evaluation is opt-in. Start conservatively on Windows:

```powershell
python compare_models.py --full --models 3 --no-plots --jobs 4
```

If TraCI stays stable, try faster settings:

```powershell
python compare_models.py --full --models 3 --no-plots --jobs 8
python compare_models.py --full --models 3 --no-plots --jobs 12
```

For the lightest full run, skip per-run metrics capture:

```powershell
python compare_models.py --full --models 3 --no-plots --jobs 8 --no-run-metrics
```

Full output goes to:

```text
outputs/reports/holdout/holdout_eval_summary_full.csv
outputs/reports/holdout/holdout_eval_dashboard_full.png
outputs/reports/holdout/holdout_eval_scorecard_full.csv
outputs/reports/holdout/holdout_eval_report_full.md
```

Success criteria:

- full daily `final_total_wait` no longer explodes into extreme values,
- full daily `p95_total_wait` improves vs Gen10,
- full daily worst-lane starvation improves,
- stress performance remains near Gen10 and above Gen9/Baseline,
- speed and throughput remain acceptable,
- phase switching does not become chaotic.

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

