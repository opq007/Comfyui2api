# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-23
**Commit:** b48f1bd
**Branch:** main

## OVERVIEW

OpenAI-compatible FastAPI gateway over one or more ComfyUI backends. Python package `comfyui2api` (src-layout, uv + setuptools) serves `/v1` and a bundled React dashboard at `/ui`. Comfy instances and public model slugs live in SQLite, not env vars.

## WORKFLOW (MANDATORY)

### New features — grill first, code second
- Any new requirement, feature, or behavior change: **do not implement until intent is fully clarified**.
- Load and follow the **`grill-me` skill**. If that skill is not available in the session, equivalent: consult **Metis** and use the **`question` tool** to interrogate the user.
- Grill until these are explicit (guessing any of them is a stop): goal, in/out of scope, non-goals, affected surfaces (API / UI / workflow / packaging), acceptance criteria, and what "done" looks like.
- After grilling, restate the understood requirement and wait for confirmation before writing code. Trivial one-line typo/config fixes are exempt.

### Bug / problem fixes — root cause, then the real fix
- Start from first principles: reproduce, isolate the causal path, name the invariant that broke.
- Fix the **root cause**. Do not patch symptoms, add workarounds, shotgun-edit, or "make the test pass" without understanding why it failed.
- If the reported cause is wrong, say so and fix the actual one.

### Verification — no evidence, not done
- Feature work and bugfixes are **not complete** until self-verification passes against the acceptance criteria / original failure.
- Python: run the focused unittest(s), then `uv run --locked --no-sync python -m unittest discover -s tests -v` when the change can affect more than one module. Frontend: `pnpm build` in `frontend/` (this is the typecheck). UI-visible changes also need `.\scripts\build-frontend.ps1` so `/ui` is not stale.
- Record what you ran and the result. Shipping without that evidence is incomplete.

## STRUCTURE

```
./
├── src/comfyui2api/     # the only Python package (flat; no subpackages)
│   ├── __main__.py      # CLI: bare/`ui`/`serve`
│   ├── app.py           # create_app(); module-level `app = create_app()`
│   ├── desktop_entry.py # PyInstaller windowed EXE trampoline → main(["ui"])
│   ├── cli_entry.py     # PyInstaller console EXE trampoline → main()
│   └── webui_dist/      # generated Vite copy; do not hand-edit
├── frontend/            # React 19 + Vite 7 + pnpm; production base `/ui/`
├── comfyui-api-workflows/  # ComfyUI File→Export (API) JSON + `.comfyui2api/*.params.json` sidecars
├── packaging/comfyui2api.spec
├── scripts/             # build-frontend.ps1, build_windows.ps1
├── tests/               # stdlib unittest (not pytest)
├── runs/ data/ logs/    # runtime; gitignored
├── start.ps1 / start.bat
└── docker-compose.yml   # no Dockerfile; host port 8460
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Boot / CLI / frozen window | `src/comfyui2api/__main__.py` | Bare `python -m comfyui2api` is UI mode (127.0.0.1, opens browser). `serve` binds `0.0.0.0`. |
| Routes / OpenAI compat | `src/comfyui2api/app.py` | ~1.9k lines; `app = create_app()` at import — env must be set *before* import/reload. |
| Admin REST/WS | `src/comfyui2api/admin_routes.py` | `ADMIN_TOKEN` only; no `API_TOKEN` fallback. |
| Config / paths | `src/comfyui2api/config.py` | Frozen: data next to EXE; assets from `_MEIPASS`. |
| Model routing | `model_catalog.py`, `instance_pool.py`, `backend_store.py` | Public `model` is a catalog slug, not a workflow filename. |
| Workflow JSON + sidecars | `workflow_registry.py`, `workflow_params.py`, `comfy_workflow.py` | Sidecars only in `WORKFLOWS_DIR/.comfyui2api/<stem>.params.json`. |
| Jobs / Comfy HTTP+WS | `jobs.py`, `job_scheduler.py`, `comfy_client.py` | |
| Signed media URLs | `signed_urls.py` | Secret = `SIGNED_URL_SECRET` or `API_TOKEN`. |
| Mount `/ui` | `webui.py` | Forces JS MIME (Windows registry often maps `.js` → `text/plain`). |
| Dashboard | `frontend/src/` | Same-origin `/v1` + `/runs`. Vite proxy is **dev-only**. |
| Test helpers | `tests/helpers.py` | Seed instances/models; mark pool healthy. No `conftest.py`. |

## CONVENTIONS

- Package manager: **uv** (`uv.lock` is source of truth). `requirements.txt` is `uv export` output — do not edit by hand.
- Frontend: **pnpm** (`frontend/pnpm-lock.yaml`). No eslint/prettier/vitest. `pnpm build` = `tsc --noEmit && vite build`.
- No ruff/mypy/pytest/pre-commit config. Do not invent those commands.
- Tests are **unittest**. Each file inserts `src/` on `sys.path`. HTTP tests use `fastapi.testclient.TestClient`.
- After frontend changes, rebuild into the package: `.\scripts\build-frontend.ps1` → `src/comfyui2api/webui_dist`. `start.ps1` auto-builds only if `webui_dist/assets` is empty.
- Workflows: ComfyUI **Export (API)** JSON in `WORKFLOWS_DIR`. Watcher only reloads `*.json` directly in that dir plus `.params.json` sidecars (not nested folders).
- Public jobs take `model` (enabled catalog slug). Do not route by workflow filename.
- Admin Bearer is `ADMIN_TOKEN`. Public `/v1` Bearer is `API_TOKEN`. WS also accepts `?token=` / `?access_token=` (browsers cannot set WS headers).
- Async OpenAI-style create: header `x-comfyui-async: 1`. Default is sync.

## ANTI-PATTERNS (THIS PROJECT)

- Do not start without both `API_TOKEN` and `ADMIN_TOKEN` — `load_config()` raises `ConfigError` (fail-closed). Empty `API_TOKEN` is not "open API".
- Do not accept `API_TOKEN` on `/v1/admin/*` or `/ui` gate. Frontend copy: 公网管理台仅接受 ADMIN_TOKEN.
- Do not reintroduce `COMFYUI_BASE_URL` into Python. Instances are registered in `/ui`. `start.ps1` still *probes* that env for a warning only.
- Do not put business logic in `desktop_entry.py` / `cli_entry.py` (PyInstaller trampolines).
- Do not write runtime data under `_MEIPASS`. `runtime_base_dir()` is the EXE directory when frozen.
- Do not hand-edit `webui_dist/` or `frontend/dist/`.
- Do not call `load_config()` / import `comfyui2api.app` before env is patched in tests — `app = create_app()` runs at import. Pattern: `patch.dict` then `importlib.reload`.
- Do not replace `secrets.compare_digest` in auth/signed-URL paths.
- Do not drop the signed-query strip list (`sig`, `exp`, `authorization`, `api_key`, `token`, `access_token`) — keep `app.py` and `admin_routes.py` in sync.
- Do not `accept()` admin WS *after* raising 401. Accept first, send JSON error, close 1008 (avoids uvicorn mapping to HTTP 403).
- Do not add `X-Forwarded-For` trust to `POST /v1/admin/shutdown` (loopback-only).
- Do not put a token on `GET /health` (desktop wait + CI smoke + compose).
- Sidecar `version` must be `1`; `kind` must match detected workflow kind.
- Do not implement a new feature without grilling (`grill-me` / Metis + `question`) and confirmed acceptance criteria.
- Do not "fix" a bug by treating symptoms. Find the root cause first.
- Do not mark a feature or fix done without self-test evidence (unittest and/or `pnpm build` / frontend rebuild as applicable).

## COMMANDS

```powershell
uv sync --locked
uv run --locked --no-sync -m comfyui2api serve          # API, 0.0.0.0:8000
uv run -m comfyui2api ui                                # 127.0.0.1 + open /ui
.\start.ps1                                             # uv sync, maybe frontend build, port fallback
.\start.ps1 -SkipFrontendBuild
.\scripts\build-frontend.ps1                            # pnpm build → src/comfyui2api/webui_dist
.\scripts\build_windows.ps1                             # frontend + PyInstaller onedir → dist/comfyui2api/

# tests (stdlib unittest; no pytest in repo)
uv run --locked --no-sync python -m unittest discover -s tests -v
uv run --locked --no-sync python -m unittest tests.test_cli -v
uv run --locked --no-sync python -m unittest tests.test_cli.CliTests.test_parse_args_defaults_to_ui_command_later -v
```

Frontend typecheck is gated on `pnpm build`, not a separate script.

```powershell
cd frontend; pnpm install; pnpm build
```

Docker: `API_TOKEN` + `ADMIN_TOKEN` required; maps **8460** (not 8000). Image `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`. No Dockerfile.

## NOTES

- Python `>=3.11`. CI/compose use 3.13. Local tests have been run as 3.11.
- `JOB_RETENTION_DAYS` (if set, including `0`) overrides `JOB_RETENTION_SECONDS`.
- Restart marks in-flight jobs failed: `Task interrupted by server restart.`
- Tests mock ComfyUI; set `ENABLE_WORKFLOW_WATCH=0`. `test_workflow_params.py` reads real files `comfyui-api-workflows/z_image_turbo_fp16.json`, `kaggle_flux2_klein_9b_kv_image_edit_api.json`, `qwen_image21_turbo_aio_api.json`, and `moss_image_aio_api.json` (the two AIO workflow tests skip when the workflow or sidecar is absent). `test_bundled_sidecars_resolve_against_their_workflows` checks every bundled `.comfyui2api/*.params.json` against its workflow, and `test_audio_speech.py` runs the real `voxcpm2_base_api.json` workflow; both skip when the bundled fixtures are absent.
- Two `test_admin_routes` cases skip unless `powershell` is on PATH (UTF-8 `.env` parser in `start.ps1`).
- CI (`.github/workflows/release-windows.yml`) builds Windows onedir and smokes the EXEs; it does **not** run `unittest`.
- PyInstaller: `comfyui2api.exe` (noconsole, `desktop_entry`) + `comfyui2api-cli.exe` (console). Spec silently omits UI if `webui_dist` is missing — always run frontend build first.
- `start.ps1` reads `.env` as UTF-8 (PS 5.1 ANSI would corrupt Chinese comments) and only forwards non-empty values (`python-dotenv` uses `override=False`).
- Image URL fetch: only global public IPs, max 3 redirects, stream-capped by `MAX_IMAGE_BYTES`.
