# TrafficAI Training Runbook

Use this file for repeatable training and evaluation on any machine.

## Environment Setup

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

SUMO must be installed and available to `sumo-rl`/TraCI. If SUMO tools are not
on the Python path, configure the machine-specific environment before training.

## Smoke Training

Use smoke mode only for integration checks:

```powershell
$env:TRAFFICAI_SMOKE_TEST="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_SMOKE_TEST -ErrorAction SilentlyContinue
```

Smoke runs are not model-quality evidence.

## Continue Current Gen11 Checkpoint

Expected files:

```text
checkpoints/ppo_traffic_model_autosave_Gen11_1009596_steps.zip
checkpoints/vecnormalize_latest.pkl
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Run:

```powershell
python train_agent.py
```

Correct startup:

```text
Autosave detected for Gen 11. Continuing training with: checkpoints\ppo_traffic_model_autosave_Gen11_1009596_steps.zip
Resuming from step 1009596...
```

If startup says `Warm-starting Gen 12 from Gen 11`, stop if the goal is to
continue the current Gen11 run.

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
- `reward/raw_worst_lane_wait`
- `reward/raw_phase_hold_seconds`
- `reward/long_green`
- `reward/starvation`
- `reward/reward_total`
- `safety/phase_forced_switches`
- `train/approx_kl`
- `train/clip_fraction`
- `train/explained_variance`

Interpretation:

- `raw_starved_lanes` should trend toward 0 or stay low.
- `raw_starvation_excess` should decrease.
- `raw_worst_lane_wait` should decrease.
- `raw_phase_hold_seconds` should not stay extremely high.
- `long_green` and `starvation` penalties should move closer to 0 over time.
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
outputs/reports/comparisons/*.png
```

Use quick evaluation to detect obvious regressions. Do not use it as final
proof of success.

## Full Evaluation

Run after Gen11 reaches 3,000,000 steps:

```powershell
python compare_models.py --full --models 3 --no-plots
```

Full output goes to:

```text
outputs/reports/holdout/holdout_eval_summary_full.csv
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

