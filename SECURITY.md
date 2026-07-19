# Security Policy

## Reporting a Vulnerability

If you find a security issue, **please do not open a public GitHub issue.** Instead, email the maintainer directly (see the profile on the repository page) with:

- A description of the issue
- Steps to reproduce
- The versions / commit affected
- Any suggested remediation

You should receive an acknowledgement within a few business days. I will keep you updated as the issue is triaged and fixed.

## Deployment hardening checklist

If you deploy Nexus AI:

- [ ] **Set `AUTH_SECRET`** to a random string of at least 32 characters. Without it, the API runs in single-user dev mode with no authentication.
- [ ] **Never expose `/repos` to untrusted callers** unless you sandbox the filesystem — it reads local paths chosen by the caller.
- [ ] **Restrict Postgres access** to the application host; use TLS for the connection string.
- [ ] **Front the API with a reverse proxy** (nginx, Caddy, Cloudflare) that terminates TLS and enforces rate limits.
- [ ] **Do not log `Authorization` headers or request bodies** — the built-in middleware only logs method/path/status/latency.
- [ ] **Rotate `AUTH_SECRET` on suspected compromise** — this invalidates all existing tokens.
- [ ] **Restrict CORS** if you deploy the UI on a different origin — the built-in UI is same-origin only.

## Supported versions

Only the latest `main` receives security fixes.
