@echo off
REM Start OMH Mock Server (Windows)

echo Starting OMH Mock Server...
echo ================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r omh_mock_requirements.txt
)

REM Start the server
echo.
echo Server will be available at:
echo   - API: http://localhost:8001
echo   - Docs: http://localhost:8001/docs
echo   - Mock Tokens: http://localhost:8001/mock/tokens
echo.
echo Press Ctrl+C to stop the server
echo ================================
echo.

python omh_mock_server.py

