@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv...
  py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo === 1/2: Airline Passenger Satisfaction — training and evaluation ===
python run_experiment.py
if errorlevel 1 (
  echo Error during the experiment.
  pause
  exit /b 1
)

echo.
echo === 2/2: Streamlit — http://localhost:8501 ===
streamlit run streamlit_app.py
