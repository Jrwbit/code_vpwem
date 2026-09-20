# Repository working rules

## Git is mandatory

- Every code, configuration, documentation, test, script, or module change made for a task must be recorded in Git before the task is handed back.
- Create an atomic commit after each independently meaningful module or coherent change set; do not leave completed work only in the working tree.
- Stage only files belonging to the current change. Preserve unrelated pre-existing changes and never use a blanket `git add -A` in a dirty worktree.
- Inspect `git status` before staging and again after committing, and report the resulting commit hash.
- Never commit secrets, credentials, private keys, datasets, model weights, caches, generated runtime output, or other large artifacts. Add an appropriate ignore rule when needed.
- Run relevant checks before committing whenever practical. If a check cannot be run or fails for a pre-existing reason, record that clearly in the handoff.
