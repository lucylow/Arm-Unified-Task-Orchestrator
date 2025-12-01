@echo off
REM Demo script for AutoRL Arm Edition - Windows Batch Version
REM Specifically designed for Android Emulator testing

echo ============================================================
echo AutoRL Arm Edition - Android Emulator Demo Script
echo ============================================================
echo.

cd /d "%~dp0\.."

echo Step 1: Checking for ADB
echo ------------------------------------------------------------
where adb >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ADB not found in PATH!
    echo.
    echo Please install Android SDK Platform Tools:
    echo Download: https://developer.android.com/studio/releases/platform-tools
    echo.
    echo Or add Android SDK to your PATH:
    echo   setx PATH "%PATH%;%LOCALAPPDATA%\Android\Sdk\platform-tools"
    echo.
    pause
    exit /b 1
)

echo [OK] ADB found
echo.

echo Step 2: Checking for connected Android device/emulator
echo ------------------------------------------------------------
adb devices | findstr /C:"device" | findstr /V "List" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] No Android device or emulator connected!
    echo.
    echo TO USE ANDROID EMULATOR:
    echo   1. Open Android Studio
    echo   2. Go to Tools ^> Device Manager
    echo   3. Create a new AVD with ARM 64 v8a architecture
    echo      (IMPORTANT: Must be ARM, not x86_64!)
    echo   4. Start the emulator (click Play button)
    echo   5. Wait for emulator to fully boot
    echo   6. Run 'adb devices' to verify connection
    echo.
    echo   See complete guide: docs\ANDROID_EMULATOR_TESTING.md
    echo.
    echo TO USE PHYSICAL DEVICE:
    echo   1. Enable USB debugging on your Android device
    echo   2. Connect via USB cable
    echo   3. Run 'adb devices' to verify connection
    echo.
    pause
    exit /b 1
)

echo [OK] Device/emulator found
echo.
echo Connected devices:
adb devices
echo.

REM Check if it's an emulator
for /f "tokens=1" %%i in ('adb devices ^| findstr "emulator"') do (
    set DEVICE_NAME=%%i
    echo [INFO] Detected Android Emulator: %%i
    echo        Verifying ARM architecture...
    for /f "delims=" %%j in ('adb shell getprop ro.product.cpu.abi') do (
        set CPU_ABI=%%j
        echo %%j | findstr /C:"arm64-v8a" >nul
        if !ERRORLEVEL! EQU 0 (
            echo        [OK] ARM architecture confirmed: %%j
        ) else (
            echo %%j | findstr /C:"armeabi-v7a" >nul
            if !ERRORLEVEL! EQU 0 (
                echo        [OK] ARM architecture confirmed: %%j
            ) else (
                echo        [WARNING] Architecture is %%j (expected arm64-v8a or armeabi-v7a)
                echo        This emulator may not work correctly. Please create an ARM-based AVD.
            )
        )
    )
    goto :found_device
)

:found_device
echo.

echo Step 3: Checking for APK
echo ------------------------------------------------------------
set APK_PATH=mobile\android\app\build\outputs\apk\debug\app-debug.apk
if not exist "%APK_PATH%" (
    echo [ERROR] APK not found at %APK_PATH%
    echo.
    echo Please build the APK first:
    echo   cd mobile\android
    echo   gradlew.bat assembleDebug
    echo.
    pause
    exit /b 1
)

echo [OK] APK found
echo.

echo Step 4: Installing APK
echo ------------------------------------------------------------
echo Installing APK on device/emulator...
adb install -r "%APK_PATH%"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] APK installation failed (may already be installed)
) else (
    echo [OK] APK installed successfully
)
echo.

echo Step 5: Launching AutoRL app
echo ------------------------------------------------------------
adb shell am start -n com.autorl/.MainActivity
echo [OK] App launched
echo.

echo Step 6: Collecting device information
echo ------------------------------------------------------------
echo.
echo Device Info:
adb shell getprop ro.product.model
adb shell getprop ro.product.cpu.abi
echo.

echo Memory Info:
adb shell dumpsys meminfo com.autorl | findstr "TOTAL"
echo.

echo ============================================================
echo [SUCCESS] Demo completed!
echo ============================================================
echo.
echo Next steps:
echo   1. Look at your emulator/device screen
echo   2. Tap 'Start Task' button in the AutoRL app
echo   3. View inference results on screen
echo   4. Check logcat for detailed timing:
echo      adb logcat -s ModelRunner:I MainActivity:I
echo.
echo Testing commands:
echo   # View real-time logs:
echo   adb logcat -s ModelRunner:I MainActivity:I ^| findstr "Inference"
echo.
echo   # Test offline capability:
echo   adb shell cmd connectivity airplane-mode enable
echo   REM Then tap 'Start Task' again - should still work!
echo   adb shell cmd connectivity airplane-mode disable
echo.
echo For complete testing guide:
echo   See docs\ANDROID_EMULATOR_TESTING.md
echo.
pause

