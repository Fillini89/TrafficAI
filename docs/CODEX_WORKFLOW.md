# Codex Workflow For TrafficAI

Use this repository as the source of truth, not chat history.

## Why These Files Exist

Codex chats may not follow the user across machines. Project memory should live
inside the repository so any new Codex session can recover context quickly.

## Startup Routine For New Codex Sessions

1. Read `AGENTS.md`.
2. Read `docs/HANDOFF.md`.
3. Check `git status --short`.
4. Inspect model/checkpoint files if training is requested.
5. Confirm whether the task is continuation, warm-start, evaluation, or code
   change.

## Documentation Roles

- `AGENTS.md`: mandatory project rules for Codex.
- `docs/HANDOFF.md`: current active state and next action.
- `docs/TRAFFICAI_CONTEXT.md`: architecture and long-term direction.
- `docs/TRAINING_RUNBOOK.md`: commands and metric interpretation.
- `docs/MODEL_REGISTRY.md`: which model artifacts matter.
- `docs/EXPERIMENT_LOG.md`: generation history and lessons.

## Updating Memory

After meaningful changes, update at least one of:

- `docs/HANDOFF.md` for current task state,
- `docs/EXPERIMENT_LOG.md` for experiment results,
- `docs/MODEL_REGISTRY.md` for new model artifacts,
- `docs/TRAINING_RUNBOOK.md` for command or output layout changes.

Keep these docs compact. They should summarize decisions and current state, not
replace detailed TensorBoard or CSV artifacts.

