#!/usr/bin/env bash
# Uruchamia eksperyment + Streamlit (macOS/Linux).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON="${PYTHON:-python3}"

venv_ok() {
  [[ -x .venv/bin/python ]] && .venv/bin/python -m pip --version &>/dev/null
}

if ! venv_ok; then
  echo "Tworzenie / naprawa środowiska .venv (brak działającego pip)..."
  rm -rf .venv
  "$PYTHON" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if ! python -m pip --version &>/dev/null; then
  echo "Błąd: pip w .venv nadal nie działa. Spróbuj ręcznie:"
  echo "  rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "=== 1/2: Wine Quality — ewaluacja MLP / RF / XGBoost (5-fold CV) ==="
python run_experiment.py

echo ""
echo "=== 2/2: Aplikacja Streamlit — http://localhost:8501 ==="
exec streamlit run streamlit_app.py
