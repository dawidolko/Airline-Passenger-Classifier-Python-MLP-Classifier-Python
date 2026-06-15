#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON="${PYTHON:-python3}"

venv_ok() {
  [[ -x .venv/bin/python ]] && .venv/bin/python -m pip --version &>/dev/null
}

if ! venv_ok; then
  echo "Creating / repairing the .venv environment..."
  rm -rf .venv
  "$PYTHON" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "=== 1/2: Airline Passenger Satisfaction — training and evaluation ==="
python run_experiment.py

echo
echo "=== 2/2: Streamlit — http://localhost:8501 ==="
exec streamlit run streamlit_app.py
