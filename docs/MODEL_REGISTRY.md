# TrafficAI Model Registry

This registry records which artifacts matter when moving between machines.

## Important Rule

A Stable-Baselines PPO model should be moved together with its matching
`VecNormalize` statistics.

Model only:

```text
models/ppo_traffic_model_Gen11.zip
```

is incomplete for normalized Gen10/Gen11 policy use.

Model plus normalization:

```text
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

is the correct portable final model pair.

## Current Final Models

Trackable final models:

```text
models/ppo_traffic_model_Gen10.zip
models/ppo_traffic_model_Gen10_vecnormalize.pkl
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Gen10 is useful as a baseline and as the source of Gen11 fine-tuning.

Gen11 is the active fairness fine-tune generation.

## Current Checkpoint Continuation Artifacts

For exact continuation of the same Gen11 run:

```text
checkpoints/ppo_traffic_model_autosave_Gen11_1009596_steps.zip
checkpoints/vecnormalize_latest.pkl
```

These are normally ignored by git, but can be force-added for cross-machine
handoff:

```powershell
git add -f checkpoints\ppo_traffic_model_autosave_Gen11_1009596_steps.zip
git add -f checkpoints\vecnormalize_latest.pkl
```

## Continuation vs Warm-Start

Continuation from checkpoint:

- resumes the same generation,
- keeps the current training step count,
- should be used when continuing Gen11 from about 1,009,596 to 3,000,000 steps.

Warm-start from final model:

- starts a new generation from existing weights,
- resets the target run accounting for the new generation,
- is useful for Gen12 or a new controlled experiment.

## Artifacts Usually Not Worth Committing

- `outputs/runtime/`
- `outputs/tensorboard/`
- `models/best_Gen*/`
- frequent autosave checkpoints except explicit handoff checkpoints.

## Artifacts Worth Committing

- source code,
- `AGENTS.md`,
- `docs/*.md`,
- network files,
- route files,
- final model zip + matching VecNormalize file,
- compact holdout summaries,
- selected comparison/training report PNGs.

