# PyInstaller spec — Localist backend (port 8001), base-only build.
#
# onedir mode (not onefile): startup is near-instant and a missing-module
# error is far easier to diagnose than in a single self-extracting
# executable. Phase C (Tauri sidecar) can revisit onefile if needed for
# final distribution.
#
# No MLX/Vision/PyMuPDF anywhere here — this is the pure-Python base
# install only (see pyproject.toml's base `dependencies`, no `[mlx]`/
# `[ocr]` extras). Confirmed via grep that every module-level import in
# backend/src/localist/ referencing those extras is already deferred, so
# PyInstaller's static analysis never even sees them.
#
# uvicorn's dynamic loop/protocol selection is covered automatically by
# pyinstaller-hooks-contrib's hook-uvicorn.py (collect_submodules('uvicorn'),
# auto-discovered — no manual hiddenimports needed for it).
#
# collect_submodules('mcp') was tried here and removed: it actually
# imports every submodule to verify it, and mcp.cli.cli requires an
# optional `typer` dependency we don't install (we only use mcp's
# FastMCP/server pieces, never its CLI) — that import failure crashed
# spec evaluation entirely. Letting PyInstaller's normal static analysis
# follow our own `from mcp... import ...` statements is sufficient; add
# specific hiddenimports here only if a real frozen-binary run surfaces a
# genuine ModuleNotFoundError.
#
# Build: pyinstaller localist-backend.spec  (from backend/packaging/)
# Run:   ./dist/localist-backend/localist-backend

block_cipher = None

a = Analysis(
    ['run_backend.py'],
    pathex=['../src'],
    binaries=[],
    # Read-only, ship-with-the-app resources WikiAgent requires (validated,
    # not best-effort — see wiki_agent.py). Found at runtime via
    # localist.paths.get_resource_root() -> sys._MEIPASS when frozen.
    datas=[
        ('../SCHEMA.md', '.'),
        ('../templates', 'templates'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='localist-backend',
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='localist-backend',
)
