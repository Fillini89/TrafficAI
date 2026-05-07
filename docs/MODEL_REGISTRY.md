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

Gen11 is the current rescue/fairness baseline.

The current Gen11 final pair was saved after continuation to the 3,000,000-step
target:

```text
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Full holdout confirms Gen11 is the current best portable model for avoiding the
Gen10 daily starvation collapse. Keep Gen10 as the failure baseline and Gen11 as
the rescue/fairness baseline for future Gen12 warm-start experiments.

## Rebuilt Gen12 Artifacts

The first Gen12 failed the p95 tail-wait objective. Its final model pair,
`models/best_Gen12/`, Gen12 autosaves, and latest checkpoint VecNormalize were
deleted after diagnosis.

The rebuilt Gen12 should warm-start from:

```text
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

Expected rebuilt Gen12 pair after training:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Rebuilt Gen12 is a reward/control fine-tune only and remains
observation-compatible with Gen11.

Historical failed Gen12 full holdout: do not promote it over Gen11. It improved
stopped AUC and speed slightly, but failed the primary p95 tail-wait objective.

Rebuilt Gen12 full holdout: promote as the current fairness leader. It improved
average p95 wait and final wait versus Gen11 and won p95 on every full-holdout
scenario. Gen11 remains the throughput/speed baseline.

## Current Checkpoint Continuation Artifacts

For exact continuation of the same Gen11 run:

```text
checkpoints/ppo_traffic_model_autosave_Gen11_1009596_steps.zip
checkpoints/vecnormalize_latest.pkl
```

Gen11 has already been continued past the 3,000,000-step target. Keep these
checkpoint artifacts for provenance, but use the final model pair for holdout
evaluation and future warm-starts.

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
- was used to continue Gen11 from about 1,009,596 to 3,000,000 steps.
- is now explicit for old completed generations via
  `TRAFFICAI_CONTINUE_CHECKPOINT=1`.

Warm-start from final model:

- starts a new generation from existing weights,
- resets the target run accounting for the new generation,
- is useful for Gen12 or a new controlled experiment.
- is the current default path when a stale checkpoint belongs to a generation
  that already has a final model artifact.

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

