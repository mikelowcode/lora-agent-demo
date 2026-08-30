# Tauri shell (Phase C of native `.app` packaging)

Wraps the Phase A static frontend build in a native macOS shell, and
spawns both Phase B PyInstaller bundles (`backend/packaging/dist/
localist-backend/`, `.../localist-mcp/`) as child processes on launch —
killing them on quit.

## Prerequisite: build the sidecars first

Tauri doesn't build these itself — `bundle.resources` in `tauri.conf.json`
just references `../../backend/packaging/dist/` directly, so it must
already exist:

```bash
cd backend/packaging
source ../.venv-packaging/bin/activate   # see backend/packaging/README.md
pyinstaller localist-backend.spec
pyinstaller localist-mcp.spec
```

## Why not Tauri's sidecar/`externalBin` mechanism

Checked against Tauri's own docs before building this: `externalBin` is
single-binary only. Our PyInstaller builds are onedir (an executable plus
a large `_internal/` dependency folder it needs alongside it) — that
doesn't fit. Instead, each onedir folder is bundled whole via the more
general `bundle.resources` (which does support directory-structure-
preserving folder bundling), resolved to a real path at runtime via
`app.path().resolve(..., BaseDirectory::Resource)`, and spawned with
plain Rust `std::process::Command` — not the shell plugin's JS-invokable
`Command`/sidecar API, whose capability system only allow-lists fixed
command names, not runtime-resolved paths. Nothing here is invoked from
JavaScript — spawning is app-lifecycle-driven — so no shell plugin or
capability JSON is needed at all; see `src/lib.rs`.

## Build

```bash
cd localist-ui
npx tauri build           # release — also produces a .dmg
npx tauri build --debug   # debug — same bundling, unoptimized, faster
```

Output: `src-tauri/target/{release,debug}/bundle/macos/Localist.app`.

## A real bug this caught: `RunEvent::ExitRequested` doesn't fire here

`src/lib.rs`'s cleanup handler originally matched only on
`RunEvent::ExitRequested` — Tauri's docs describe it as the normal
"about to exit" hook. Live-tested (AppleScript `tell application
"Localist" to quit`, the same Apple Event Cmd+Q/Dock-quit send): this
app's default configuration goes straight to `RunEvent::Exit` with no
`ExitRequested` at all. Confirmed by adding temporary `eprintln!`
instrumentation and watching a real quit — both sidecar processes were
left running (`ps aux` showed them still alive after the app itself had
exited) until the handler was changed to match both events. Fixed; kept
matching both since `.take()` on the stored `Option<Child>` makes
handling either (or both) harmless.

## What's verified so far (Phase C)

Both debug and release `.app` builds: launched via `open` (the real user
flow, not a dev shortcut) and directly, both sidecars start automatically
and respond on their real ports, the real webview makes genuine periodic
`GET /health`/`GET /agents` calls that reach the packaged backend with no
CORS errors (confirms `main.py`'s `tauri://localhost` origin fix), a
screenshot confirmed the real frontend renders correctly (not blank), and
quitting via AppleScript reliably leaves zero orphaned processes. Not yet
covered: first-run config UX (Phase D — a fresh `.app` still defaults to
`foundry`, unreachable without setup, same as source-tree today), code
signing/notarization (Phase E), and wiring `tauri build` to trigger the
PyInstaller build itself rather than assuming it's already done.
