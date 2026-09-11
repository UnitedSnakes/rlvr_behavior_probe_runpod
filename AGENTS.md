# Agent instructions for research-workbench

This branch is the internal research workbench. It intentionally keeps agent handoff notes, experiment provenance, debugging history, and implementation details that are not meant for the public-facing `main` branch.

## Branch policy

- `main` is the research-facing branch. Do not merge or rebase `main` into this branch.
- Do not merge this branch, `codex/*`, or other workbench branches wholesale into `main`.
- If a specific change from `main` is needed here, cherry-pick only the exact commit(s) required after checking the diff.
- If a specific workbench change should become public-facing, make a small clean commit for that change and cherry-pick it to `main`; do not merge the branch.
- Never use a broad "sync with main" operation unless the user explicitly asks for it in the current conversation.
- Before any branch-changing operation, run or inspect a branch/commit comparison and state which direction the change will flow.

## What to preserve here

Keep `CLAUDE.md`, `docs/superpowers/`, internal plans/checkpoints, debugging records, historical scripts, and other context useful for future agents or experiment provenance.

The public-facing README and repository presentation live on `main`. This branch may be messier by design.
