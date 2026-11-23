@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found. See README.md for setup.
    pause
    exit /b 1
)
if not exist ".cache\mlflow" mkdir ".cache\mlflow"
echo MLflow will be available at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri sqlite:///./.cache/mlflow/mlflow.db --host 127.0.0.1 --port 5000
if errorlevel 1 (
    echo MLflow could not start. Install requirements.txt and check whether port 5000 is free.
    pause
    exit /b 1
)
