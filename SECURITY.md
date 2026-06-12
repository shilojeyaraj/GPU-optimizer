# Security Policy

## Supported Versions

This project is pre-1.0. Only the `main` branch receives security updates.

| Version | Supported |
| ------- | --------- |
| main    | ✅        |
| < 1.0   | ❌        |

## Reporting a Vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**

Report security issues by email to **security@coincidencelabs.com** with the subject line `SECURITY: GPU-Optimizer`. Include:

- A description of the vulnerability and its potential impact
- Reproduction steps or a proof-of-concept
- Your contact info for follow-up

You can expect:

- An acknowledgement within **3 business days**
- A status update within **7 business days**
- A coordinated disclosure timeline (typically 30–90 days depending on severity)

## Scope

In-scope:
- The FastAPI backend (`backend/`) and Celery worker
- The Next.js frontend (`frontend/`)
- The C++ pybind11 module (`cpp/`)
- Docker images built from `Dockerfile.backend` / `Dockerfile.frontend`

Out-of-scope:
- Third-party services (Neon, Vercel, Anthropic, Perplexity) — report to those vendors directly
- Self-hosted deployment misconfigurations (use the docker-compose stack as the canonical reference)
- Denial-of-service against your own development environment

## Hardening notes

The default `docker-compose.yml` is for **local development only**. For production:

- Replace the hardcoded Postgres credentials (`gpuopt:gpuopt`)
- Replace the hardcoded Grafana admin password
- Disable Grafana anonymous access (`GF_AUTH_ANONYMOUS_ENABLED=false`)
- Put the backend behind TLS termination
- Set restrictive `CORS_ORIGINS`
- Use a secrets manager for `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` instead of a `.env` file
