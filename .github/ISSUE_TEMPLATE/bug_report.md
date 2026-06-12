---
name: Bug report
about: Something is broken or behaves differently than documented
title: "bug: <one-line summary>"
labels: ["bug", "triage"]
assignees: []
---

## What happened

<!-- One or two sentences. What you saw vs. what you expected. -->

## Reproduction

Minimal steps to reproduce. The more specific the better.

```bash
# Example
docker compose up -d
curl -X POST http://localhost:8000/jobs -d '...'
```

## Expected behavior

<!-- What you think should have happened. -->

## Environment

- OS: <!-- e.g. macOS 15, Ubuntu 24.04, Windows 11 -->
- Python: <!-- python --version -->
- Node: <!-- node --version -->
- Running via: <!-- docker compose | local dev | other -->
- Commit / version: <!-- git rev-parse HEAD or release tag -->
- GPU (if relevant): <!-- e.g. RTX 4090, H100, none -->

## Logs / screenshots

<details>
<summary>Relevant log lines</summary>

```
paste here
```

</details>

## Additional context

<!-- Anything else useful: recent changes you made, hardware quirks, etc. -->
