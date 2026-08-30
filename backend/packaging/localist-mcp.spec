# PyInstaller spec — localist-mcp (port 8003), base-only build.
# See localist-backend.spec for the shared reasoning (onedir, no MLX/
# Vision/PyMuPDF, uvicorn hook auto-discovery, why mcp's submodules aren't
# force-collected).
#
# Build: pyinstaller localist-mcp.spec  (from backend/packaging/)
# Run:   ./dist/localist-mcp/localist-mcp

block_cipher = None

a = Analysis(
    ['run_mcp_server.py'],
    pathex=['../src'],
    binaries=[],
    datas=[],
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
    name='localist-mcp',
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
    name='localist-mcp',
)
