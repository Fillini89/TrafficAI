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

Gen15 has completed training and full holdout. Gen12 remains the pure fairness
champion, Gen14 remains the safer service-age balanced baseline, and Gen15 is
the best aggregate balanced candidate but has a max-service-age regression.
Gen16 is the active experiment from Gen15 using service-age budget control. If
the user asks to continue training, first verify whether the intended action is:

- continue an unfinished checkpoint run,
- warm-start Gen16 from Gen15 with `TRAFFICAI_WARM_START_GEN=15`,
- warm-start a new generation from Gen14 to preserve service-age safety,
- warm-start a different new generation from the latest final model,
- or train from scratch.

Current important final pairs:

```text
models/ppo_traffic_model_Gen12.zip
models/ppo_traffic_model_Gen12_vecnormalize.pkl
models/ppo_traffic_model_Gen14.zip
models/ppo_traffic_model_Gen14_vecnormalize.pkl
models/ppo_traffic_model_Gen15.zip
models/ppo_traffic_model_Gen15_vecnormalize.pkl
```

Do not delete or overwrite Gen14 or Gen15 unless explicitly asked. The next
iteration should address Gen15's max service-age tail with warning/critical
service-age budget metrics without giving back its final-wait, speed, and
stopped-burden gains.

