# Usage Guide

## Prerequisites

- Docker and Docker Compose
- (Optional) An Anthropic API key, for live LLM-grounded agent reasoning. Without it, the platform runs in offline/grounded mode — Stage 6 (Section in `ARCHITECTURE.md`) produces the same structured output without a live model call.

## Quick start

```bash
# 1. Set your Anthropic API key (optional)
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Build and start everything
docker compose up -d --build

# 3. Open the platform
open http://localhost:8080
```

**Always use `docker compose up -d --build`, not bare `up -d`.** Bare `up -d` silently reuses cached image layers and will not pick up source-code changes — this has caused confusion in this project's own development before (a "fix" that appeared to do nothing was actually just running against a stale image).

To confirm a rebuild actually happened:

```bash
docker images bpmn-agentic-platform-backend --format "table {{.Repository}}\t{{.ID}}\t{{.CreatedAt}}"
```

`CREATED AT` should match when you last edited the source, not an earlier date.

## Running a pipeline

1. Upload a `.xes`, `.xes.gz`, `.csv`, or `.tsv` event log via the frontend, or `POST /api/datasets/upload`.
2. Trigger a run via the UI or `POST /api/pipeline/run/{dataset_id}`.
3. Watch progress through the six stages (Architecture doc has the full breakdown); results, including the episode trace log, are available once the run completes.

### Uploading large files

nginx enforces `client_max_body_size 200M` (see `nginx/nginx.conf`). Raw XES files compress extremely well — BPI Challenge 2012's 74MB raw file compresses to ~3.3MB gzipped, a ~95% reduction — so the reliable fix for a large raw `.xes` file is to compress it first, not to raise the nginx limit.

**On Linux/macOS:**
```bash
gzip -k -9 your_file.xes
```

**On Windows PowerShell**, there's no built-in `gzip`, but `.NET`'s `GZipStream` works without installing anything:
```powershell
$in  = [System.IO.File]::OpenRead("your_file.xes")
$out = [System.IO.File]::Create("your_file.xes.gz")
$gz  = New-Object System.IO.Compression.GZipStream($out, [System.IO.Compression.CompressionLevel]::Optimal)
$in.CopyTo($gz)
$gz.Close(); $out.Close(); $in.Close()
```

A ready-to-run script version of this (`Compress-XesToGz.ps1`) with before/after magic-byte verification is included in this repo under `scripts/`. If Windows blocks it from running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Compress-XesToGz.ps1 -InputPath "your_file.xes"
```

`-NoProfile` matters if you have a custom PowerShell profile that re-applies a stricter execution policy after launch — it skips loading that profile for this one invocation, so the `Bypass` flag you passed actually sticks.

If you still see a "not digitally signed" error after that, check whether the policy is enforced above the process level:
```powershell
Get-ExecutionPolicy -List
```
If `MachinePolicy` or `UserPolicy` shows anything other than `Undefined`, it's set via Group Policy or registry and can't be overridden from inside PowerShell at all — that needs changing via `gpedit.msc` or the registry directly.

The compressed `.xes.gz` still needs to genuinely be a gzip stream, not just a renamed file — the backend checks the actual byte content (`0x1F 0x8B` magic bytes), not the filename, so a plain rename does nothing.

## API reference

All routes are reachable both directly against the backend (`:8000`) and through nginx (`:8080/api/...`).

```
GET  /health                              health check
GET  /api/info                            platform info
GET  /api/datasets/                       list datasets
GET  /api/datasets/{dataset_id}           dataset detail
GET  /api/datasets/{dataset_id}/preview   preview parsed content
POST /api/datasets/upload                 upload a single file
POST /api/datasets/upload-many            batch upload
POST /api/pipeline/run/{dataset_id}       run the full six-stage pipeline
POST /api/pipeline/run-batch              run the pipeline across multiple datasets
GET  /api/pipeline/batch/{batch_id}       batch run status
GET  /api/pipeline/runs/{dataset_id}      list runs for a dataset
GET  /api/pipeline/runs                   list all runs
POST /api/pipeline/compare                cross-dataset comparison report
GET  /api/pipeline/result/{run_id}        full result for one run
GET  /api/pipeline/audit/{run_id}         audit log / lineage trail
GET  /api/pipeline/report/{run_id}        generated PDF-style report
POST /api/pipeline/report/compare         comparison report across runs
POST /api/pipeline/ask/{dataset_id}       ask the agent a question about a dataset
GET  /api/pipeline/episodes/{run_id}      episode trace log (JSONL)
```

Interactive API docs (Swagger UI) are available at `/docs` once the stack is running.

## Interpreting a run

Every report includes, per computed feature, an explicit provenance string (formula and data source), and every effect-size comparison table flags which baseline the learned policy actually beat. If a Cohen's d cell shows `—` rather than a number, that's deliberate (see `SYSTEM_DESIGN.md`'s note on zero-variance handling) — it means the effect size is undefined for that comparison, not zero.

## Sharing a running instance with one external reviewer (e.g., your supervisor)

Two separate problems, solved independently, so a weakness in one doesn't compromise the other:

1. **Transport** — getting an HTTPS URL that reaches your local machine, without port-forwarding your router or exposing your home IP.
2. **Access control** — making sure only someone with the right credentials can actually use that URL once they have it, since a tunnel URL is not a secret by itself (it can leak, get guessed, or get crawled).

Solve both. A tunnel alone is not access control — anyone who obtains the URL can use it.

### Step 1 — Access control (do this first, regardless of tunnel choice)

The platform's nginx layer supports HTTP Basic Auth on every route except `/health`, off by default. Turn it on:

```bash
# Generate the credentials file (requires the apache2-utils / httpd-tools package,
# or use an online bcrypt-htpasswd generator if you'd rather not install anything)
htpasswd -c nginx/.htpasswd bart
# You'll be prompted to set a password -- pick one you'll share with your
# supervisor over a different channel than the repo (email, Slack, in person).
```

Then uncomment the two matching sections:
- `auth_basic` / `auth_basic_user_file` lines in `nginx/nginx.conf`
- the `.htpasswd` volume mount in `docker-compose.yml`'s `nginx` service

```bash
docker compose up -d --build
```

`nginx/.htpasswd` is already covered by `.gitignore` — verify before your next commit:

```bash
git status --ignored | grep htpasswd
```

It should show up under ignored files, not staged.

### Step 2 — Transport: get a public HTTPS URL

Two options, in order of setup effort.

**Option A — ngrok (fastest, good for a one-off review session)**

```bash
# Install ngrok, then:
ngrok http 8080
```

This prints a temporary `https://....ngrok-free.app` URL that tunnels straight to your local nginx (port 8080), TLS terminated by ngrok. Combined with Basic Auth from Step 1, your supervisor needs both the URL *and* the password to get in. The URL changes every time you restart ngrok on the free tier — send the new one each session, or use a paid ngrok plan for a reserved subdomain if you'll be doing this repeatedly.

**Option B — Cloudflare Tunnel + Cloudflare Access (more setup, no shared password at all)**

Instead of a shared password, this restricts access to a specific email address — your supervisor logs in with `b.a.lameijer@uva.nl` and gets a one-time login code sent to that address, no credential for you to generate or transmit.

```bash
# Install cloudflared, then create a tunnel (requires a free Cloudflare account):
cloudflared tunnel login
cloudflared tunnel create bpmn-thesis-review
cloudflared tunnel route dns bpmn-thesis-review review.yourdomain.com   # or use the free *.trycloudflare.com URL for a quick one-off
cloudflared tunnel --url http://localhost:8080 run bpmn-thesis-review
```

Then, in the Cloudflare Zero Trust dashboard, add an Access policy on that hostname restricting entry to `b.a.lameijer@uva.nl` specifically. With this option you can actually leave nginx's Basic Auth off, since Cloudflare Access is doing the identity check before traffic even reaches your machine — though leaving both on is harmless defense-in-depth if you'd rather keep it simple and consistent with Option A.

### What never gets committed

- `nginx/.htpasswd` (Basic Auth credentials)
- `.env` (Postgres password, `SECRET_KEY`, Anthropic API key)
- any `cloudflared` config directory or tunnel credentials file, if you go with Option B

All three are already in `.gitignore`. Before any push, a quick sanity check:

```bash
git status
git diff --cached --stat
```

If anything under those paths shows up as staged, stop and unstage it (`git restore --staged <path>`) before committing.

### When the review session is over

Stop the tunnel process (`Ctrl+C` on `ngrok` or `cloudflared`). This immediately kills the public URL — the platform is back to being reachable only from `localhost` on your own machine. Nothing further to clean up.

## Tearing down

```bash
docker compose down       # stop containers, keep data/ and volumes
docker compose down -v    # stop containers and remove all state (Postgres volume, etc.)
```

`data/uploads`, `data/runs`, and `data/chroma` are bind-mounted from the host, so `docker compose down -v` does not delete files already written there — only the named Docker volumes (Postgres). Delete `./data` manually if you want a fully clean slate.
