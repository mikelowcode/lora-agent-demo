"""
PyInstaller entrypoint — Localist backend (port 8001).

Not `main.py`'s own "if __name__" block: that one is a stale dev-only
convenience (reload=True, a string import path that predates the src/
layout restructure), never called by start_localist.sh. A frozen build
needs genuinely different behavior — no reload (doesn't make sense in a
single frozen process), and the real `app` object passed directly rather
than resolved by import string, so uvicorn never needs its own import
machinery to find `localist.main` inside the frozen bundle.

See backend/packaging/README.md for how this gets built.
"""

from __future__ import annotations

import uvicorn

from localist.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host      = "127.0.0.1",
        port      = 8001,
        reload    = False,
        log_level = "info",
    )
