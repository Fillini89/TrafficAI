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
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
models/ppo_traffic_model_Gen13.zip
models/ppo_traffic_model_Gen13_vecnormalize.pkl
models/ppo_traffic_model_Gen14.zip
models/ppo_traffic_model_Gen14_vecnormalize.pkl
models/ppo_traffic_model_Gen15.zip
models/ppo_traffic_model_Gen15_vecnormalize.pkl
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

## Gen13 Artifacts

Gen13 should warm-start from the protected rebuilt Gen12 pair:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Expected Gen13 pair after training:

```text
models/ppo_traffic_model_Gen13.zip
models/ppo_traffic_model_Gen13_vecnormalize.pkl
```

The first Gen13 final pair failed and was deleted along with `models/best_Gen13/`,
Gen13 autosaves, and `checkpoints/vecnormalize_latest.pkl`. Historical reports
were kept for diagnosis.

Rebuilt Gen13 is observation-compatible with Gen12. Its purpose is to preserve
Gen12 p95/final-wait dominance while reducing actual signal jitter through
service-age fairness debt and hard cadence guardrails. Do not delete Gen12
artifacts before or during Gen13; Gen12 remains the champion comparison baseline.

Rebuilt Gen13 completed training and full holdout, but should not replace Gen12:
it reduced actual switching substantially, while p95/final wait and hidden
starvation proxies regressed versus Gen12. Keep Gen12 as the promoted champion.

## Gen14 Artifacts

Gen14 should warm-start explicitly from protected Gen12, not from Gen13:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
```

Use `TRAFFICAI_WARM_START_GEN=12` so startup creates Gen14 while loading Gen12
weights and VecNormalize stats even though a final Gen13 pair exists.

Expected Gen14 pair after training:

```text
models/ppo_traffic_model_Gen14.zip
models/ppo_traffic_model_Gen14_vecnormalize.pkl
```

Gen14 is observation-compatible with Gen12/Gen13. Its purpose is to preserve
Gen12 p95/final-wait fairness while recovering some of Gen13's smoothness via
adaptive cadence, not hard cadence. Keep Gen12 and Gen13 artifacts intact.

Gen14 completed 3M training and full holdout. It should be kept as the current
balanced candidate: it did not beat Gen12 on average p95 wait, but stayed within
the planned p95/final-wait tolerance while reducing actual switches by about 61%
and improving mean speed and stopped burden versus Gen12. Gen12 remains the
protected pure fairness champion.

## Gen15 Artifacts

Gen15 should warm-start explicitly from protected Gen14:

```text
models/ppo_traffic_model_Gen14.zip
models/ppo_traffic_model_Gen14_vecnormalize.pkl
```

Use `TRAFFICAI_WARM_START_GEN=14` so startup creates Gen15 while loading Gen14
weights and VecNormalize stats. Keep Gen14 final artifacts intact.

Expected Gen15 pair after training:

```text
models/ppo_traffic_model_Gen15.zip
models/ppo_traffic_model_Gen15_vecnormalize.pkl
```

Gen15 is a minimal-risk micro-tune, not a new architecture. It keeps reward
weights unchanged and only adjusts adaptive service-age cadence to target lower
p95/final wait and max service age while preserving Gen14's speed and smoothness.

Gen15 completed 3M training and full holdout. Keep the final pair as an
aggregate-performance candidate: it improved final wait, p95 wait, stopped AUC,
and mean speed versus Gen14, with only a moderate actual-switch increase.
However, max service age worsened materially, so Gen15 should not fully replace
Gen14 until that service-age tail is fixed or accepted as a tradeoff.

## Gen16 Artifacts

Gen16 should warm-start explicitly from protected Gen15:

```text
models/ppo_traffic_model_Gen15.zip
models/ppo_traffic_model_Gen15_vecnormalize.pkl
```

Use `TRAFFICAI_WARM_START_GEN=15` so startup creates Gen16 while loading Gen15
weights and VecNormalize stats. Keep Gen15 final artifacts intact.

Expected Gen16 pair after training:

```text
models/ppo_traffic_model_Gen16.zip
models/ppo_traffic_model_Gen16_vecnormalize.pkl
```

Gen16 is a service-age budget fine-tune, not a new architecture. It keeps Gen15
adaptive cadence and aggregate-flow intent, while adding warning/critical
service-age budget penalties and budget reporting.

Gen16 completed 3M training and full holdout. Keep the final pair as the leading
aggregate-performance candidate: it improved final wait, p95 wait, stopped AUC,
mean speed, and hidden-starvation proxies versus Gen15 while preserving smooth
actual switching. It did not solve the service-age tail; max service age and
warning/critical budget metrics worsened versus Gen15.

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

