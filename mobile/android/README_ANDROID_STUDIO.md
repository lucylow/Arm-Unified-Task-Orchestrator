# Setting Up AUTO in Android Studio

This guide will help you open and run the AUTO application in Android Studio.

## Prerequisites

1. **Android Studio** (latest version recommended)
   - Download from: https://developer.android.com/studio
   - Make sure you have Android SDK installed (API 24-34)

2. **Android SDK Components**
   - Android SDK Build-Tools
   - Android SDK Platform-Tools
   - Android SDK Platform (API 34)
   - Android Emulator

## Opening the Project

1. **Open Android Studio**
2. **Select "Open"** from the welcome screen
3. **Navigate to** `mobile/android` folder in this project
4. **Click "OK"** to open the project

Android Studio will:
- Detect the Gradle project
- Download Gradle wrapper if needed
- Sync the project
- Download dependencies

## First-Time Setup

When you first open the project, Android Studio may ask you to:

1. **Sync Gradle Files** - Click "Sync Now" if prompted
2. **Install Missing SDK Components** - Follow the prompts to install required SDK components
3. **Create local.properties** - Android Studio will create this automatically with your SDK path

## Running on Emulator

1. **Create an Emulator** (if you don't have one):
   - Click `Tools` > `Device Manager` (or `AVD Manager`)
   - Click `Create Device`
   - Select a device (e.g., Pixel 6)
   - Select a system image (API 34 recommended)
   - Click `Finish`

2. **Run the App**:
   - Click the green "Run" button (▶️) in the toolbar
   - Or press `Shift + F10`
   - Select your emulator from the device list
   - The app will build and install automatically

3. **The AUTO App**:
   - The app will launch automatically
   - You'll see the AUTO interface
   - Click "Start Task" to run inference

## Generating App Icons

The app references launcher icons that need to be generated:

1. **Right-click** on `app/src/main/res` folder
2. Select **New** > **Image Asset**
3. Choose **Launcher Icons (Adaptive and Legacy)**
4. Configure your icon (or use the default)
5. Click **Next** and **Finish**

Android Studio will generate all required icon sizes automatically.

## Troubleshooting

### Gradle Sync Issues
- If Gradle sync fails, try: `File` > `Invalidate Caches` > `Invalidate and Restart`
- Make sure you have internet connection for downloading dependencies

### Build Errors
- Ensure you have the correct Android SDK installed
- Check that `local.properties` exists (Android Studio creates this automatically)
- Verify Java 17 is installed and configured
- **Missing Icons**: If you see errors about missing `ic_launcher`, generate icons using Image Asset Studio (see above)

### Emulator Issues
- Make sure HAXM or Hyper-V is enabled for emulator acceleration
- Try creating a new emulator with different settings

## Project Structure

```
mobile/android/
├── app/                    # Main application module
│   ├── src/main/
│   │   ├── java/          # Kotlin source code
│   │   ├── res/           # Resources (layouts, strings, etc.)
│   │   └── AndroidManifest.xml
│   └── build.gradle       # App-level build config
├── build.gradle           # Project-level build config
├── settings.gradle        # Project settings
└── gradle.properties      # Gradle properties
```

## App Information

- **App Name**: AUTO
- **Package**: com.autorl
- **Min SDK**: 24 (Android 7.0)
- **Target SDK**: 34 (Android 14)
- **Language**: Kotlin
- **Architecture**: MVVM with Hilt dependency injection

## Features

- On-device AI inference using PyTorch Mobile
- Material Design 3 UI
- MVVM architecture
- Coroutines for async operations
- Performance monitoring

