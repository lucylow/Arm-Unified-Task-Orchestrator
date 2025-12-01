# 📱 Android Emulator Quick Reference Card

**Print this page for quick reference while testing!**

## ✅ Pre-Flight Checklist

- [ ] Android Studio installed
- [ ] ARM emulator created (ARM 64 v8a, **NOT x86_64**)
- [ ] Emulator running and fully booted
- [ ] `adb devices` shows `emulator-5554 device`
- [ ] `adb shell getprop ro.product.cpu.abi` shows `arm64-v8a`

## 🚀 Quick Commands

### Verify Setup
```bash
adb devices                    # Check connection
adb shell getprop ro.product.cpu.abi  # Verify ARM architecture
```

### Build & Install
```bash
cd mobile/android
./gradlew assembleDebug       # Build APK
adb install -r app/build/outputs/apk/debug/app-debug.apk  # Install
adb shell am start -n com.autorl/.MainActivity  # Launch
```

### View Logs
```bash
adb logcat -s ModelRunner:I MainActivity:I
adb logcat -s ModelRunner:I | grep "Inference completed"
```

### Test Offline
```bash
adb shell cmd connectivity airplane-mode enable   # Enable
# Tap "Start Task" - should still work!
adb shell cmd connectivity airplane-mode disable  # Disable
```

### Performance Monitoring
```bash
adb shell dumpsys meminfo com.autorl | grep TOTAL
adb shell top -n 1 | grep com.autorl
```

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| `adb devices` shows nothing | Restart emulator, then `adb kill-server && adb start-server` |
| Architecture shows `x86_64` | Create new AVD with ARM system image |
| APK install fails | `adb uninstall com.autorl` then reinstall |
| App crashes | Check logs: `adb logcat -s AndroidRuntime:E` |

## 📍 File Locations

- **APK**: `mobile/android/app/build/outputs/apk/debug/app-debug.apk`
- **Complete Guide**: `docs/ANDROID_EMULATOR_TESTING.md`
- **Demo Script**: `./demo/run_demo.sh` (Unix) or `demo\run_demo_emulator.bat` (Windows)

## 🎯 Expected Results

- ✅ App launches without crashes
- ✅ "Start Task" button works
- ✅ Inference completes in 30-50ms
- ✅ Logs show: `ModelRunner: Inference completed in XX ms`
- ✅ Works in airplane mode (offline)

---

**For detailed instructions, see [ANDROID_EMULATOR_TESTING.md](ANDROID_EMULATOR_TESTING.md)**

