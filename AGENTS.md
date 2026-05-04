# TrafficAI Codex Instructions

This repository is TrafficAI: a SUMO + reinforcement learning project for
traffic-light control.

Before making non-trivial changes, read these files in order:

1. `docs/HANDOFF.md`
2. `docs/TRAFFICAI_CONTEXT.md`
3. `docs/TRAINING_RUNBOOK.md`
4. `docs/MODEL_REGISTRY.md`
5. `docs/EXPERIMENT_LOG.md`

## Project Goal

Train a PPO traffic-light controller that improves throughput and speed while
preventing starvation of individual lanes or approaches. A model that moves the
main flow quickly but lets a minor lane wait forever is considered a failed
policy.

## Non-Negotiable Rules

- Do not change observation shape when the goal is to continue fine-tuning an
  existing generation. A changed observation dimension requires a new model
  architecture and usually training from scratch.
- Always keep Stable-Baselines models paired with their matching
  `*_vecnormalize.pkl` file.
- Use checkpoint files only when continuing the same run and step count.
  Use final model files for warm-starting a new generation.
- Quick holdout evaluation is useful for regression checks, but full daily
  evaluation is required to detect long-horizon starvation.
- Treat fairness metrics as first-class success criteria:
  `p95_total_wait`, `raw_worst_lane_wait`, `raw_starved_lanes`,
  `raw_starvation_excess`, `final_total_wait`, and `phase_switch_count`.
- Keep hard traffic-safety constraints outside the policy when possible:
  `min_green`, `yellow_time`, conflict-free phase definitions, and max-green
  guardrails.
- Do not remove or revert user changes unless explicitly asked.

## Current Default Task

The current active experiment is Gen11 fine-tuning from Gen10. If the user asks
to continue training, first verify whether the intended action is:

- continue the same Gen11 checkpoint run,
- warm-start a new generation from the latest final model,
- or train from scratch.

At the moment, the intended path is usually to continue Gen11 from its checkpoint
toward 3,000,000 total steps.

