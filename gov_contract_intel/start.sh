#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# start.sh — one-click setup + launch for Gov Contract Intel
# Run once to set up, or any time to start the app.
# -----------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

VENV=".venv"
REQUIREMENTS="requirements.txt"

# ── 1. Create venv if missing (version check only needed here) ────────────────
if [ ! -d "$VENV" ]; then
    if ! command -v python3 &>/dev/null; then
        echo "ERROR: python3 not found. Install Python 3.10+ from https://python.org and retry."
        exit 1
    fi
    PY_VER=$(python3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
    PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
        echo "ERROR: Python 3.10+ required to create the environment (found $PY_VER)."
        echo "  Option 1: Install Python 3.10+ from https://python.org"
        echo "  Option 2: Pre-create the venv with a newer Python, then re-run:"
        echo "    /path/to/python3.11 -m venv .venv && ./start.sh"
        exit 1
    fi
    echo "Creating virtual environment with Python $PY_VER..."
    python3 -m venv "$VENV"
fi

# ── 2. Activate ───────────────────────────────────────────────────────────────
source "$VENV/bin/activate"
PY_VER=$(python3 -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
echo "Python $PY_VER detected."

# ── 4. Install / update dependencies ─────────────────────────────────────────
MARKER="$VENV/.deps_installed"
if [ ! -f "$MARKER" ] || [ "$REQUIREMENTS" -nt "$MARKER" ]; then
    echo "Installing dependencies (first run may take a few minutes)..."
    pip install --upgrade pip --quiet

    # Install PyTorch before sentence-transformers so we control the build.
    # macOS already ships CPU/MPS builds via PyPI — the whl/cpu index only
    # has Linux/Windows x86_64 and will install a broken binary on Apple Silicon.
    if [[ "$OSTYPE" == "darwin"* ]]; then
        pip install torch --quiet
    else
        pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    fi

    pip install -r "$REQUIREMENTS" --quiet
    touch "$MARKER"
    echo "Dependencies installed."
fi

# ── 5. Check required model files ─────────────────────────────────────────────
MISSING=0
for f in "models/xgb_amount_model.joblib" "models/xgb_modification_model.joblib"; do
    if [ ! -f "$f" ]; then
        echo "WARNING: Missing model file: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "  Prediction features will not work until model files are present."
fi

if [ ! -d "models/all-MiniLM-L6-v2" ]; then
    echo "NOTE: Embedding model not found locally — will download (~87 MB) on first use."
fi

# ── 6. Ensure data directory exists ───────────────────────────────────────────
mkdir -p data

# ── 7. Launch ─────────────────────────────────────────────────────────────────
echo ""
echo "Starting Gov Contract Intel at http://localhost:5001"
echo "Press Ctrl+C to stop."
echo ""
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python app.py
