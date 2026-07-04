# Project Agent Instructions

- This repository is the clean v2 workspace for the official 2026 Nornickel hackathon task `Скажи мне, кто твой шлиф`.
- We are working on the v2 iteration. The canonical active workdir is `/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2`; start new local work there, not in the legacy broad checkout unless explicitly requested.
- Always check `~/.claude/CLAUDE.md`, then this file or `CLAUDE.md`, then `docs/session-activity.md` (live coordination), then `docs/session-sync.md` before meaningful work.
- **Live session coordination (required for parallel work):** Before starting meaningful work, read `docs/session-activity.md` and add/update your own entry there recording what you are doing *now* — a short task description, the files/directories you are actively editing, any `docs/plans/NN_` / `docs/specs/` names or numbers you are claiming, and a status (`active` / `blocked` / `done`). Refresh the entry when your focus shifts, and mark it `done` (or prune it) when you finish or hand off. Multiple sessions run concurrently, so consult this file first to avoid editing the same file at once, duplicating work, or reusing an in-flight plan/spec number. `docs/session-activity.md` is ephemeral coordination state (who is doing what right now); `docs/session-sync.md` remains the durable handoff of implementation state — keep the two distinct.
- Treat `AGENTS.md` and `CLAUDE.md` as the same project instruction source. `CLAUDE.md` should remain a symlink to this file.
- Keep the v2 scope narrow: optical microscopy only, official ore-task labels, high-resolution tiled inference, masks, metrics, reports, and weak-supervision training utilities.
- Do not port the old broad OM/SEM/XRD/product UI unless the user explicitly promotes that scope.
- The dataset entry is currently a local copy of the verified official dataset; do not duplicate or replace the full dataset again unless explicitly requested.
- Core implementation plans are `docs/plans/25_standalone-ore-classifier-project.md` and `docs/plans/26_weak-supervision-sulfide-binary-model.md`.
- Keep all v2 UI-specific docs under `docs/ui/v2/`, including UI specs, UI plans, UI notes, and the v2 UI candidate backlog. Do not add new v2 UI docs under root `docs/specs/`, `docs/plans/`, or `docs/notes/`.
- Update `docs/session-sync.md` and `ChangeLog.md` after meaningful changes to implementation state, benchmark results, task priorities, or known blockers.
- Important research findings must not live only in conversation. Save them under `docs/notes/`, `docs/plans/`, or `docs/decisions/`, and link them from the handoff if they affect next work.
