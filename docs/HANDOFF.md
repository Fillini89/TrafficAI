# TrafficAI Handoff

This is the first file a new Codex session should read.

## Current Situation

TrafficAI is training a PPO-based traffic-light controller in SUMO. The current
active model is Gen11, which is a fine-tune from Gen10.

Gen10 was strong on stress scenarios but failed full 24-hour daily evaluation
because of lane starvation. Some approaches could wait for extreme periods while
aggregate speed and stopped metrics still looked acceptable.

Gen11 was created to fix this without changing observation shape.

## Current Objective

Continue Gen11 training from its checkpoint until 3,000,000 total steps. Then run
quick and full holdout comparisons.

Do not make new reward or observation changes before the 3M Gen11 evaluation
unless the user explicitly asks. The current experiment should remain clean.

## Expected Continuation Files

```text
checkpoints/ppo_traffic_model_autosave_Gen11_1009596_steps.zip
checkpoints/vecnormalize_latest.pkl
models/ppo_traffic_model_Gen11.zip
models/ppo_traffic_model_Gen11_vecnormalize.pkl
```

## Correct Training Startup

Run:

```powershell
python train_agent.py
```

Correct output:

```text
Autosave detected for Gen 11. Continuing training with: checkpoints\ppo_traffic_model_autosave_Gen11_1009596_steps.zip
Resuming from step 1009596...
```

If output says `Warm-starting Gen 12 from Gen 11`, stop and inspect checkpoint
discovery unless the user explicitly wants Gen12.

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

## Evaluation Plan

After Gen11 reaches 3M:

```powershell
python compare_models.py --models 3
python compare_models.py --full --models 3 --no-plots
```

Main success condition:

Full daily `final_total_wait` and `p95_total_wait` should no longer show the
Gen10 starvation failure. Stress performance should remain competitive with
Gen10 and clearly better than the fixed baseline.

## If The User Is Ambiguous

The user may say "continue training", "run the model", or "resume Gen11" loosely.
Clarify by inspecting local files and command output. Do not assume a new
generation is desired.

Default assumption right now:

Continue the existing Gen11 checkpoint run to 3,000,000 steps.

