# TrafficAI Troubleshooting

## Windows Native Crash While Loading PPO Zip

Symptom:

```text
PPO.load(...)
exit=-1073741819
```

`-1073741819` is Windows access violation `0xC0000005`. During the Alienware
Area-51 migration, this happened before training started and was not a SUMO
socket exhaustion.

Known fixes used in this repo:

- Keep the training venv on `numpy==1.26.4`.
- Keep PyTorch aligned with the saved Gen11 artifact metadata:
  `torch==2.8.0+cpu`.
- Use `sb3_compat.load_ppo_compat(...)` instead of direct `PPO.load(...)`.

The compatibility loader reads the SB3 zip metadata and PyTorch weights, builds
a fresh PPO object, then applies the saved policy and optimizer state. It avoids
native crashes triggered by direct SB3 `PPO.load(...)` on some Windows
dependency combinations.

## Marathon Exit Message

`marathon.ps1` restarts training after any non-zero exit code except manual
`Ctrl+C`. Treat its restart message as "inspect the preceding output", not as
proof that the problem is a socket failure.

## Matplotlib Tcl/Tk Error

Symptom:

```text
_tkinter.TclError: Can't find a usable init.tcl
```

This can happen when Matplotlib selects a Tk backend on a machine where Python's
Tcl/Tk install is incomplete. `compare_models.py` forces the non-interactive
`Agg` backend so holdout evaluation can run headlessly.

Presentation artifacts can be rebuilt without SUMO:

```powershell
python compare_models.py --report-only quick
python compare_models.py --report-only full
```

## Parallelism Tuning

Use environment variables instead of editing reward, observation, or curriculum
settings:

```powershell
$env:TRAFFICAI_NUM_CPU="16"
$env:TRAFFICAI_TORCH_NUM_THREADS="1"
./marathon.ps1
```

Raise `TRAFFICAI_NUM_CPU` gradually. Windows TraCI stability can degrade if too
many SUMO workers start at once.

Use `TRAFFICAI_STARTUP_CHECK=1` with `TRAFFICAI_NUM_CPU=1` to verify checkpoint
discovery and model loading without entering the training loop.

## Checkpoint Discovery Looks Wrong

For the Gen12 experiment, old Gen11 autosaves are stale because Gen11 already
has a final model pair. Normal startup should print:

```text
Stale autosave ignored for Gen 11
Warm-starting Gen 12 from Gen 11
```

If you intentionally need to resume the old Gen11 autosave for provenance:

```powershell
$env:TRAFFICAI_CONTINUE_CHECKPOINT="1"
python train_agent.py
Remove-Item Env:TRAFFICAI_CONTINUE_CHECKPOINT -ErrorAction SilentlyContinue
```

If a Gen12 autosave exists and no final Gen12 model exists yet, startup should
resume that Gen12 autosave automatically.

For the Gen13 experiment, rebuilt Gen12 is protected. Normal startup should
print:

```text
Stale autosave ignored for Gen 12
Warm-starting Gen 13 from Gen 12
```

If startup resumes a Gen12 autosave unexpectedly, verify that the final Gen12
model pair exists and avoid deleting it:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

If rebuilding Gen13 under the same name after a failed run, remove only the
failed Gen13 lineage before startup:

```text
models/ppo_traffic_model_Gen13.zip
models/ppo_traffic_model_Gen13_vecnormalize.pkl
models/best_Gen13/
checkpoints/ppo_traffic_model_autosave_Gen13_*_steps.zip
checkpoints/vecnormalize_latest.pkl
```

Keep Gen12 final artifacts and reports.

For the Gen14 experiment, Gen12 is still the source model even though Gen13
exists. Set the explicit warm-start generation:

```powershell
$env:TRAFFICAI_WARM_START_GEN="12"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
python train_agent.py
```

Expected:

```text
Warm-starting Gen 14 from Gen 12
```

Forbidden:

```text
Warm-starting Gen 14 from Gen 13
```

If startup still uses Gen13, check that `TRAFFICAI_WARM_START_GEN=12` is set in
the same PowerShell session and that the Gen12 model pair exists:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

For the Gen15 experiment, Gen14 is the protected source model. Set:

```powershell
$env:TRAFFICAI_WARM_START_GEN="14"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
python train_agent.py
```

Expected:

```text
Warm-starting Gen 15 from Gen 14
```

If startup resumes a stale Gen14 checkpoint unexpectedly, remove
`TRAFFICAI_CONTINUE_CHECKPOINT` from the session unless the intent is exact
checkpoint continuation:

```powershell
Remove-Item Env:TRAFFICAI_CONTINUE_CHECKPOINT -ErrorAction SilentlyContinue
```

Verify the Gen14 source pair exists:

```text
models/ppo_traffic_model_Gen14.zip
models/ppo_traffic_model_Gen14_vecnormalize.pkl
```

For the Gen16 experiment, Gen15 is the protected source model. Set:

```powershell
$env:TRAFFICAI_WARM_START_GEN="15"
$env:TRAFFICAI_STARTUP_CHECK="1"
$env:TRAFFICAI_NUM_CPU="1"
python train_agent.py
```

Expected:

```text
Warm-starting Gen 16 from Gen 15
```

If startup resumes a stale Gen15 checkpoint unexpectedly, remove
`TRAFFICAI_CONTINUE_CHECKPOINT` unless exact checkpoint continuation is intended.

Verify the Gen15 source pair exists:

```text
models/ppo_traffic_model_Gen15.zip
models/ppo_traffic_model_Gen15_vecnormalize.pkl
```

For holdout evaluation, `compare_models.py --jobs N` runs independent
scenario/agent evaluations in parallel. If SUMO or TraCI starts failing:

- reduce `--jobs`,
- increase `--worker-stagger-seconds`,
- use `--no-plots --no-run-metrics` for the lightest full evaluation path.

Conservative Alienware/Windows starting point:

```powershell
python compare_models.py --full --models 3 --no-plots --jobs 4
```
