# Project Agent Instructions

- This repository is the clean v2 workspace for the official 2026 Nornickel hackathon task `Скажи мне, кто твой шлиф`.
- We are working on the v2 iteration. The canonical active workdir is `/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2`; start new local work there, not in the older `../2026_Nornikel_Hackaton` checkout unless explicitly requested.
- Always check `~/.claude/CLAUDE.md`, then this file or `CLAUDE.md`, then `docs/session-sync.md` before meaningful work.
- Treat `AGENTS.md` and `CLAUDE.md` as the same project instruction source. `CLAUDE.md` should remain a symlink to this file.
- Keep the v2 scope narrow: optical microscopy only, official ore-task labels, high-resolution tiled inference, masks, metrics, reports, and weak-supervision training utilities.
- Do not port the old broad OM/SEM/XRD/product UI unless the user explicitly promotes that scope.
- The dataset is a symlink to `../2026_Nornikel_Hackaton/dataset`; do not duplicate the full dataset in this repository unless explicitly requested.
- Core implementation plans are `docs/plans/25_standalone-ore-classifier-project.md` and `docs/plans/26_weak-supervision-sulfide-binary-model.md`.
- Keep all v2 UI-specific docs under `docs/ui/v2/`, including UI specs, UI plans, UI notes, and the v2 UI candidate backlog. Do not add new v2 UI docs under root `docs/specs/`, `docs/plans/`, or `docs/notes/`.
- Update `docs/session-sync.md` and `ChangeLog.md` after meaningful changes to implementation state, benchmark results, task priorities, or known blockers.
- Important research findings must not live only in conversation. Save them under `docs/notes/`, `docs/plans/`, or `docs/decisions/`, and link them from the handoff if they affect next work.
