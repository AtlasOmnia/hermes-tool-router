# Security Notes

This folder was prepared to avoid personal Hermes runtime data.

Before publishing, scan again for:

- API keys or bearer tokens
- local usernames and LAN IPs
- provider profiles
- private prompt history
- logs and SQLite databases
- personal skills or memories
- deleted diff lines from local-only patches

The router feedback code should store only metadata such as route names, toolsets, token counts, timestamps, and error categories. It must not store raw user prompts.