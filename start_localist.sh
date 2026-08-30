#!/usr/bin/env bash
# start_localist.sh — Localist Framework service launcher
#
# Part of: Localist CLI
#
# Starts:
#   • Localist backend    (FastAPI / uvicorn) — port 8001
#   • Localist MCP server (FastAPI / uvicorn) — port 8003
#   • Localist frontend   (SvelteKit / vite)  — port 5173
#
# The inference engine (oMLX, MLX-LM, Ollama, LM Studio, etc.) is managed
# separately. Localist is inference-engine-agnostic.
#
# The standalone Fetcher microservice (port 8002) was retired in Phase 2 —
# its /extract path now lives in-process on localist-mcp as the fetch_url
# MCP tool. See backend/mcp_server/url_fetch.py.
#
# Usage:
#   ./start_localist.sh          — start all services
#   ./start_localist.sh --stop   — kill any running instances on ports 8001/8003/5173
#
# Logs:
#   logs/backend.log
#   logs/mcp_server.log
#   logs/frontend.log
#
# Ctrl+C stops all services cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/localist-ui"
VENV_PYTHON="$BACKEND_DIR/.venv/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# --stop flag: kill any running instances and exit
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping Localist services..."
    lsof -ti tcp:8001 | xargs kill -TERM 2>/dev/null && echo "  backend (8001) stopped." || echo "  backend (8001) not running."
    lsof -ti tcp:8003 | xargs kill -TERM 2>/dev/null && echo "  localist-mcp (8003) stopped." || echo "  localist-mcp (8003) not running."
    lsof -ti tcp:5173 | xargs kill -TERM 2>/dev/null && echo "  frontend (5173) stopped." || echo "  frontend (5173) not running."
    exit 0
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    echo "Run: cd backend && python -m venv .venv && source .venv/bin/activate && pip install -e \".[dev]\""
    exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "ERROR: node_modules not found at $FRONTEND_DIR/node_modules"
    echo "Run: cd localist-ui && npm install"
    exit 1
fi

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
    echo "WARNING: $BACKEND_DIR/.env not found — environment variables may be missing."
fi

# ---------------------------------------------------------------------------
# First-run prompt: local embedding model (EmbeddingGemma via MLX)
#
# LOCALIST_EMBEDDING_ENGINE_ENABLED defaults to false (backend/src/localist/
# main.py's Settings) — this is the one place a user gets asked about the
# ~400MB download instead of it happening silently. Only fires when: .env
# exists but doesn't already set the key (respects an explicit choice,
# including a previous run of this same prompt), this is Apple Silicon (the
# only platform EmbeddingEngine can run on), the `[mlx]` extra looks
# installed, and stdin is a real terminal (never blocks a non-interactive
# run — CI, a backgrounded invocation, etc.).
# ---------------------------------------------------------------------------
if [[ -f "$BACKEND_DIR/.env" ]] \
    && ! grep -q '^LOCALIST_EMBEDDING_ENGINE_ENABLED=' "$BACKEND_DIR/.env" \
    && [[ "$(uname -s)" == "Darwin" && "$(uname -m)" =~ ^(arm64|aarch64)$ ]] \
    && [[ -t 0 ]] \
    && "$VENV_PYTHON" -c "import mlx_embeddings" &>/dev/null; then
    echo ""
    echo "Local embedding model not yet configured (EmbeddingGemma via MLX, ~400MB,"
    echo "downloaded once from Hugging Face on first use). Without it, memory/RAG"
    echo "retrieval runs in keyword-only mode — still fully functional, just not"
    echo "semantic. You can change this later via LOCALIST_EMBEDDING_ENGINE_ENABLED"
    echo "in backend/.env."
    read -r -p "Download and enable it now? [y/N] " EMBED_REPLY
    if [[ "$EMBED_REPLY" =~ ^[Yy]$ ]]; then
        echo "LOCALIST_EMBEDDING_ENGINE_ENABLED=true" >> "$BACKEND_DIR/.env"
        echo "  Enabled — will download on the next backend startup."
    else
        echo "LOCALIST_EMBEDDING_ENGINE_ENABLED=false" >> "$BACKEND_DIR/.env"
        echo "  Skipped — running in keyword-only mode. Change anytime in backend/.env."
    fi
    echo ""
fi

# Warn if ports already in use (do not abort — let uvicorn surface the error)
for PORT in 8001 8003 5173; do
    if lsof -ti tcp:$PORT &>/dev/null; then
        echo "WARNING: port $PORT is already in use. Run ./start_localist.sh --stop first."
    fi
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

echo ""
echo "  ██╗      ██████╗  ██████╗ █████╗ ██╗     ██╗███████╗████████╗"
echo "  ██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██║██╔════╝╚══██╔══╝"
echo "  ██║     ██║   ██║██║     ███████║██║     ██║███████╗   ██║   "
echo "  ██║     ██║   ██║██║     ██╔══██║██║     ██║╚════██║   ██║   "
echo "  ███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║███████║   ██║   "
echo "  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝   "
echo ""
echo "  Localist Framework — local-first agentic assistant"
echo ""
echo "  Backend      → http://127.0.0.1:8001  (log: logs/backend.log)"
echo "  localist-mcp → http://127.0.0.1:8003  (log: logs/mcp_server.log)"
echo "  Frontend     → http://127.0.0.1:5173  (log: logs/frontend.log)"
echo ""
echo "  Ctrl+C to stop all services."
echo ""

# ---------------------------------------------------------------------------
# Launch services
# `localist` is pip-installed editable into backend/.venv (src/ layout), so
# these resolve from any cwd — cd backend/ here only for log-path convenience
# and to keep --reload-dir/--reload-exclude paths short.
# ---------------------------------------------------------------------------
cd "$BACKEND_DIR"

"$VENV_PYTHON" -m uvicorn localist.main:app \
    --host 127.0.0.1 \
    --port 8001 \
    --reload \
    --reload-exclude 'src/localist/mcp_server/*' \
    > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

"$VENV_PYTHON" -m uvicorn localist.mcp_server.main:app \
    --host 127.0.0.1 \
    --port 8003 \
    --reload \
    --reload-dir src/localist/mcp_server \
    > "$LOG_DIR/mcp_server.log" 2>&1 &
MCP_PID=$!

(cd "$FRONTEND_DIR" && npm run dev > "$LOG_DIR/frontend.log" 2>&1) &
FRONTEND_PID=$!

# ---------------------------------------------------------------------------
# Cleanup on Ctrl+C or unexpected exit
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Stopping Localist services..."
    kill -TERM "$BACKEND_PID" 2>/dev/null && echo "  backend stopped."
    kill -TERM "$MCP_PID" 2>/dev/null && echo "  localist-mcp stopped."
    kill -TERM "$FRONTEND_PID" 2>/dev/null && echo "  frontend stopped."
    kill "$TAIL_BACKEND_PID" "$TAIL_MCP_PID" "$TAIL_FRONTEND_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Tail all logs interleaved with service prefix
# ---------------------------------------------------------------------------
tail -f "$LOG_DIR/backend.log" | sed 's/^/[backend] /' &
TAIL_BACKEND_PID=$!

tail -f "$LOG_DIR/mcp_server.log" | sed 's/^/[mcp] /' &
TAIL_MCP_PID=$!

tail -f "$LOG_DIR/frontend.log" | sed 's/^/[frontend] /' &
TAIL_FRONTEND_PID=$!

# Wait — if any service exits unexpectedly, surface it
wait "$BACKEND_PID" "$MCP_PID" "$FRONTEND_PID"

kill "$TAIL_BACKEND_PID" "$TAIL_MCP_PID" "$TAIL_FRONTEND_PID" 2>/dev/null
echo ""
echo "A Localist service exited unexpectedly. Check logs/ for details."
exit 1
