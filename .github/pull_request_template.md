<!--
Thanks for the PR! Please fill in the sections below. CI will block until tests pass.
-->

## Summary

<!-- 1–3 sentences. What changed and why. Link the issue this resolves: "Closes #123". -->

## Type of change

- [ ] `feat` — new user-visible capability
- [ ] `fix` — bug fix
- [ ] `refactor` — internal restructuring with no behavior change
- [ ] `perf` — performance improvement
- [ ] `docs` — documentation only
- [ ] `test` — adding or fixing tests
- [ ] `build` / `ci` — tooling, dependencies, CI
- [ ] `chore` — housekeeping that touches no production code

## Areas touched

- [ ] Backend (`backend/`)
- [ ] Frontend (`frontend/`)
- [ ] ML / PyTorch (`ml/`)
- [ ] C++ performance layer (`cpp/`)
- [ ] Docker / docker-compose
- [ ] Database schema (`backend/app/db/migrations/`)
- [ ] Docs (`docs/`, `README.md`)

## Checklist

- [ ] Tests written / updated (see [CONTRIBUTING.md](../CONTRIBUTING.md) for what counts)
- [ ] Docs updated (API, OBSERVABILITY, ARCHITECTURE, or `.env.example` if relevant)
- [ ] `ruff check` and `ruff format` pass locally (or `pre-commit run --all-files`)
- [ ] Frontend: `npm run lint` and `npm run build` pass locally
- [ ] No secrets, API keys, or credentials committed
- [ ] If a new env var was added, it appears in `.env.example` AND the README env-var table

## How to test

<!-- Curl, screenshots, repro steps. "Just look at the CI" is fine for refactor PRs. -->

## Breaking changes

<!-- If yes, describe migration path here AND tag the commit with `BREAKING CHANGE:` in the footer. -->

None.
