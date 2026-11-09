# Security policy and threat model

AIO Local is a **single-user, local-only** application. This document states what that means, what
is defended, and what is explicitly out of scope. It is written for a first-time user so the
guarantees are not overstated.

## Supported version

Only the current release (v0.2.x) is supported.

## How to report a problem

Open a private security advisory on the repository, or contact the maintainer directly. Do not open a
public issue for a vulnerability. Please include the version, the steps, and what you expected.

## Threat model

### Defended

| Threat | Defence |
|---|---|
| Another machine on the LAN reaching the app | server binds `127.0.0.1` only |
| A malicious website in your browser calling the local API (CSRF / DNS rebinding) | every API call requires a per-install bearer token; `Host` must be `127.0.0.1`/`localhost`/`[::1]`; cross-origin `Origin`/`Referer` are rejected; custom headers force a CORS preflight that is never granted; UI is same-origin only |
| Another local user account reading the token | token encrypted with Windows DPAPI under the current user |
| Path traversal through artifact or model IDs | all filesystem joins go through `safe_path()` / allow-listed ID patterns |
| Command injection in model pull / OVMS launch | fixed executable path, argument lists (never `shell=True`), strict `org/name` regex |
| Prompt text stored in logs | prompts are not written at info level; logs redact tokens and truncate bodies |
| Malicious model repository name | `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$` |
| XSS in the transcript | the renderer escapes HTML first, then applies a small allow-list of Markdown; links are limited to `http(s)` and open with `noopener noreferrer` |
| Clickjacking | `X-Frame-Options: DENY` + CSP `frame-ancestors 'none'` |
| Data exfiltration by the app | no telemetry, no remote logging, no auto-update, no hidden HTTP calls; the only outbound request is a model pull you start explicitly |

### Not defended (be honest)

- **A compromised Windows account.** If malware runs as you, it can read the decrypted token, the
  database, and every prompt. Local AI does not change that.
- **A malicious model you download.** Model files are not sandboxed; only download from sources you
  trust, and prefer the official `OpenVINO/*` repositories.
- **Prompt injection inside documents or web content you paste.** The memory system validates
  database writes, but a model can still be persuaded to say wrong things. Treat output as untrusted.
- **Supply-chain attacks on dependencies.** Dependencies are pinned at install time; the build
  script does not fetch arbitrary code. Review `pyproject.toml` before installing.
- **Physical access** to an unlocked machine.

## What the app never does

- It never opens a listening socket on a non-loopback address.
- It never executes text produced by a model.
- It never deletes memory rows when you edit a message (append-only by design).
- It never uploads your conversations, files, prompts, or telemetry.
- It never runs as Administrator, and the installer does not request elevation.

## If you think something is wrong

1. Stop the app.
2. Copy `%LOCALAPPDATA%\AIOLocal\logs\` somewhere safe.
3. Do **not** paste your token or private conversations into a public issue.
4. Report privately with the log excerpt and reproduction steps.

## Hardening checklist for testers

- [ ] `netstat -ano | findstr 7317` shows `127.0.0.1:7317`, never `0.0.0.0`.
- [ ] Opening `http://<your-LAN-IP>:7317` from another device fails.
- [ ] Requests without the token return `401`.
- [ ] A request with `Origin: https://evil.example` returns `403`.
- [ ] `%LOCALAPPDATA%\AIOLocal\config\token.bin` is unreadable gibberish, not plain text.
