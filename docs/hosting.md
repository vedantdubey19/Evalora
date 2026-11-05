# Private Evalora hosting

This deployment provides one owner workspace, not public registration or isolated customer accounts. Everyone with the dashboard password or API token can view all reports and use the same provider budget. Use separate deployments for separate trust boundaries.

## Deployment design

- Caddy serves HTTPS and requires a bcrypt-backed Basic login for every dashboard request, including WebSockets, uploads and media. It overwrites a private gateway header; the dashboard refuses production access without that header. The password is generated, not stored as plaintext in the gateway configuration.
- API routes use a separate high-entropy bearer token. Health returns only process status. Interactive API documentation is disabled in production. Caddy routes `/api/*` to the API and all other paths to the dashboard.
- API request limits use shared SQLite counters. Generation and judging share a daily provider-attempt budget and concurrent-call leases across the API and dashboard. Failures and retries consume budget; cached hits do not. Exceeding local provider admission returns `429` on the API; experiments record a safe quota/rate reason and stop subsequent calls according to existing circuit rules.
- Named Docker volumes retain reports, counters and TLS certificates across container recreation. Reports use the existing atomic checkpoint writer. Authenticated API endpoints list up to 200 report files and download a selected JSON/CSV file up to 10 MB.
- App containers run as a non-root user with a read-only filesystem, dropped capabilities, bounded memory/processes, and temporary scratch space. Only the gateway publishes ports. Access logs are disabled for the API, and Caddy access logging is not enabled. Do not enable debug or request-body logging on a shared server.

Limits default to 30 authenticated API requests per fixed minute, two dashboard run starts per minute, 200 provider attempts per UTC day, and two admitted provider calls concurrently. These are global owner-workspace limits. Fixed windows can permit bursts around boundaries. Call counts are not a monetary spending cap. Set a provider-side billing budget separately. Leases release on completion and expire after the configured timeout plus 30 seconds to recover crashed workers; they are admission safeguards, not forced termination of an external request.

## Prepare the server

Use a Linux host with Docker Engine and Compose v2, a domain pointing at the host, and inbound TCP 80/443. The two application services have 2 GiB memory limits each; size the host accordingly. No cloud resources are provisioned by these files.

Clone the repository on the server. Run this from its root with Python and the existing dotenv dependency installed:

```sh
python -m deploy.configure --domain evalora.example.com --email owner@example.com
```

Enter your Gemini API key into the hidden prompt. The tool uses Docker to hash a generated dashboard password and writes `.env.production` and `.credentials.local` with mode `0600`. It prints no credentials and refuses to overwrite existing files. Move the dashboard credentials into your password manager and keep the environment file private. These files are ignored by Git and excluded from the Docker build context.

```sh
docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production up -d --build --wait
```

Open `https://YOUR_DOMAIN` and use the generated dashboard credentials. API clients use `Authorization: Bearer <token>` at `https://YOUR_DOMAIN/api/...`. Use the token from a secret manager or private client configuration, rather than command-line arguments or shell history. The dashboard password does not authenticate API requests; the API token does not authenticate the dashboard.

For existing local reports, copy only the desired JSON/CSV files into `/var/lib/evalora/reports` on the application volume while services are stopped. Keep the files owned by UID/GID 10001. Do not copy local `.env` files, credentials or caches into reports.

## Back up and recover

Create a consistent backup inside the running application volume, then copy it to a protected local location:

```sh
docker compose --env-file .env.production exec -T api python -m deploy.backup /var/lib/evalora/backup.tar.gz
docker compose --env-file .env.production cp api:/var/lib/evalora/backup.tar.gz ./backup.tar.gz
```

The backup contains private reports and a SQLite snapshot of limit state, not API keys or deployment secrets. It refuses to overwrite an existing archive. Protect the downloaded archive, encrypt off-server copies, and remove the in-volume archive after exporting it so the next backup can run. Store deployment secrets separately in your password manager.

To restore a trusted backup, stop the app services, restore `reports/` and `state/` under `/var/lib/evalora` with ownership 10001:10001, then restart. Do not restore untrusted archives. Test restoration before relying on backups. Volume persistence is not an off-server backup. `docker compose down` preserves named volumes; `down -v` deletes them. Deleting the limits database also resets budgets.

## Operations

- Rotate credentials by updating the private environment file and recreating affected services. Rotate the API token, gateway secret and cookie secret independently; use new random values of at least 32 characters. Generate a new bcrypt password hash using Caddy's hidden-input command. Browser Basic authentication is stateless; use a private browser window and close it when finished. There is no application session logout or self-service password recovery.
- Keep one dashboard process; the current run manager is process-local. Resume checkpoints after a restart rather than assuming the worker survived. The API can share SQLite state across processes on this single host. This design does not support multi-host replicas or network-filesystem SQLite.
- Reports are retained until the owner removes them. Monitor disk space and perform backups before deletion. A full or unavailable limit store fails closed; it does not silently disable quotas. Storage failures stop report-writing jobs.
- Use exact caching for hosted runs. Guarded embedding mode requires separate model storage and capacity planning; the hosted configuration pins exact mode.
- Update base images and dependencies deliberately, run tests and the container acceptance check, then deploy. A running process health check does not prove provider access or judge quality.

## Verification

`make verify` covers authentication, host validation, quota concurrency and expiry, restart persistence, chunked-body limits, private backups and report path checks, alongside the existing offline workflows. CI additionally builds the image and runs `python -m deploy.smoke` on disposable containers. It checks HTTPS routing, dashboard login, API token enforcement, limits, and reports surviving recreation without paid provider calls. The smoke command creates disposable credentials and should run only in a fresh CI checkout.

Implementation references: [Caddy Basic authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth), [reverse proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy), [Streamlit configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml), and [Docker Compose service controls](https://docs.docker.com/reference/compose-file/services/).
