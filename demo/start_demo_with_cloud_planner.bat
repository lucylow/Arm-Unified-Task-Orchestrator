@echo off
REM Complete Demo Script for AutoRL ARM Edition - Windows Version
REM Starts backend with cloud planner + ARM integration + mock data
REM Then builds and installs mobile app on emulator

echo ============================================================
echo AutoRL ARM Edition - Complete Demo Setup
echo Cloud Planner + ARM Integration + Mock Data
echo ============================================================
echo.

cd /d "%~dp0\.."

echo Step 1: Checking prerequisites...
echo ------------------------------------------------------------
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)
echo [OK] Python found

where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found
    pause
    exit /b 1
)
echo [OK] Node.js found

where adb >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ADB not found. Please install Android SDK Platform Tools.
    pause
    exit /b 1
)
echo [OK] ADB found

echo.
echo Step 2: Checking for Android emulator...
echo ------------------------------------------------------------
adb devices | findstr /C:"device" | findstr /V "List" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] No Android device or emulator connected.
    echo.
    echo Please start Android emulator first:
    echo   1. Open Android Studio
    echo   2. Go to Tools ^> Device Manager
    echo   3. Start your ARM emulator (ARM 64 v8a)
    echo   4. Wait for it to fully boot
    echo   5. Run this script again
    echo.
    pause
    exit /b 1
)

echo [OK] Device/emulator found
adb devices

REM Check ARM architecture
for /f "tokens=1" %%i in ('adb devices ^| findstr "emulator"') do (
    for /f "delims=" %%j in ('adb shell getprop ro.product.cpu.abi') do (
        echo %%j | findstr /C:"arm64-v8a" >nul
        if !ERRORLEVEL! EQU 0 (
            echo [OK] ARM architecture confirmed: %%j
        ) else (
            echo %%j | findstr /C:"armeabi-v7a" >nul
            if !ERRORLEVEL! EQU 0 (
                echo [OK] ARM architecture confirmed: %%j
            ) else (
                echo [WARNING] Architecture is %%j (expected arm64-v8a)
            )
        )
    )
    goto :found_device
)

:found_device
echo.
echo Step 3: Checking backend dependencies...
echo ------------------------------------------------------------
cd backend

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

python -c "import fastapi" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing backend dependencies...
    pip install -r requirements.txt
)

echo [OK] Backend dependencies OK

echo.
echo Step 4: Checking frontend dependencies...
echo ------------------------------------------------------------
cd ..\frontend

if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

echo [OK] Frontend dependencies OK

echo.
echo Step 5: Building mobile app...
echo ------------------------------------------------------------
cd ..\mobile\android

if not exist "app\build\outputs\apk\debug\app-debug.apk" (
    echo Building APK...
    call gradlew.bat assembleDebug
) else (
    echo [OK] APK already built
)

set APK_PATH=app\build\outputs\apk\debug\app-debug.apk

if not exist "%APK_PATH%" (
    echo [ERROR] APK not found. Build failed.
    pause
    exit /b 1
)

echo [OK] APK built successfully

echo.
echo Step 6: Installing APK on emulator...
echo ------------------------------------------------------------
adb install -r "%APK_PATH%"

if %ERRORLEVEL% EQU 0 (
    echo [OK] APK installed successfully
) else (
    echo [WARNING] APK installation failed (may already be installed)
)

echo.
echo Step 7: Launching app...
echo ------------------------------------------------------------
adb shell am start -n com.autorl/.MainActivity
echo [OK] App launched

echo.
echo ============================================================
echo [SUCCESS] Setup Complete!
echo ============================================================
echo.
echo Next Steps:
echo.
echo 1. Start Backend Server (in a new terminal):
echo    cd backend
echo    venv\Scripts\activate
echo    set AUTORL_MODE=demo
echo    python -m servers.master_backend
echo.
echo 2. Start Frontend Dashboard (in another terminal):
echo    cd frontend
echo    npm run dev
echo.
echo 3. Open Browser:
echo    http://localhost:5173
echo.
echo 4. Test the Complete Flow:
echo    - In emulator: Tap 'Start Task' button
echo    - In dashboard: Watch real-time task execution
echo    - In backend logs: See cloud planner + ARM inference
echo.
echo Verification Commands:
echo.
echo   # Check backend health:
echo   curl http://localhost:5000/api/health
echo.
echo   # View mobile app logs:
echo   adb logcat -s ModelRunner:I MainActivity:I
echo.
echo   # Test API endpoint:
echo   curl -X POST http://localhost:5000/api/tasks -H "Content-Type: application/json" -d "{\"device_id\": \"emulator-5554\", \"instruction\": \"Send money to John\"}"
echo.
echo For complete guide:
echo   See docs\ANDROID_EMULATOR_TESTING.md
echo.
pause

