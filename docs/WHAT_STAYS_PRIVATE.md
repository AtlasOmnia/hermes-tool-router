# What Stays Private

Your full Hermes workspace is private operational data, not part of this public project.

Keep private:

- `~/.hermes/.env`
- `~/.hermes/config.yaml` and backups
- `~/.hermes/auth.json`, OAuth files, Google credentials, tokens
- `~/.hermes/logs/`
- `~/.hermes/sessions/`, `session-log/`, `.hermes_history`
- `~/.hermes/state.db`, `response_store.db`, `kanban.db`, and related `-wal`/`-shm` files
- `~/.hermes/memories/`, `mnemosyne/`, `skills/`, `profiles/`, `skins/`
- `~/.hermes/router_feedback/events.jsonl`
- machine-specific scripts such as provider switching, launchctl restart helpers, and personal patch automation

The public project should contain only router code, public docs, test fixtures, and sanitized install guidance.
