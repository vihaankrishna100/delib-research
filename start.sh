#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deliberation Map Simulation ==="
echo ""

# Check API key
if [ -z "$GOOGLE_API_KEY" ]; then
  if [ -f "$PROJECT_DIR/backend/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/backend/.env" | xargs)
  fi
fi

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY not set."
  echo "Set it in backend/.env or export GOOGLE_API_KEY=..."
  exit 1
fi

# Start backend
echo "Starting backend on :8001..."
cd "$PROJECT_DIR/backend"
source venv/bin/activate
uvicorn main:app --reload --port 8001 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on :3001..."
cd "$PROJECT_DIR/frontend"
npm run dev -- --port 3001 &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8001"
echo "Frontend: http://localhost:3001"
echo "API docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

cleanup() {
  echo "Stopping..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait
