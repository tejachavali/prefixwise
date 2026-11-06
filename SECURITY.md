# Security policy

Prefixwise sits between an agent and a model API, so it sees API keys in transit and prompt bodies. Reports about anything that could leak either, or let someone else spend a key, are especially welcome.

## Reporting a vulnerability

Please **don't** open a public issue. Use GitHub's private vulnerability reporting on this repository (**Security → Report a vulnerability**). Include what you found, how to reproduce it, and the version (`prefixwise --version`). You'll get an acknowledgement within a few days. Fixes are released, and credited if you want, once a patched version is out.

## What Prefixwise does and doesn't do with your data

- It **stores** hashes of prompt blocks, token counts, cache-miss causes, and the session, task and agent labels you send. **Never** prompt text, completions or API keys. The ledger file is owner-only (0600).
- It **forwards** your client's auth headers to the upstream you configured, and nowhere else. It makes no other network calls: no telemetry.
- It **listens** on 127.0.0.1 by default. Beyond localhost, set `PREFIXWISE_AUTH_TOKEN`. With `PREFIXWISE_API_KEY` set, it refuses to listen beyond localhost without one.

## Supported versions

The latest release on `main`. This is an early project (0.x); fixes are not backported.
