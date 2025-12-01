# iOS App Setup Guide

This guide will help you set up the AutoRL iOS application in Xcode.

## Prerequisites

- macOS with Xcode 14.0 or later
- iOS 15.0+ deployment target
- CocoaPods (optional, for PyTorch Mobile)

## Quick Setup

### Option 1: Create New Xcode Project (Recommended)

1. Open Xcode
2. Create a new project:
   - Choose "iOS" → "App"
   - Product Name: `AutoRL`
   - Interface: `SwiftUI`
   - Language: `Swift`
   - Minimum Deployment: `iOS 15.0`

3. Copy all files from `mobile/ios/AutoRL/` to your new project:
   - `AutoRLApp.swift`
   - `ContentView.swift`
   - `MainViewModel.swift`
   - `ModelRunner.swift`
   - `APIService.swift`
   - `Utils.swift`
   - `Info.plist`

4. Add files to Xcode:
   - Right-click on your project in Navigator
   - Select "Add Files to AutoRL..."
   - Select all the Swift files

5. Update `Info.plist` with the contents from `mobile/ios/AutoRL/Info.plist`

### Option 2: Use Existing Project Structure

If you prefer to use the provided project structure:

1. Open `mobile/ios/AutoRL.xcodeproj` in Xcode
2. If the project doesn't open correctly, create a new project and add the files manually

## Configuration

### 1. Backend URL Configuration

Edit `APIService.swift` to set your backend URL:

```swift
#if targetEnvironment(simulator)
self.baseURL = "http://localhost:8000"
#else
// For physical device, use your computer's IP address
self.baseURL = "http://192.168.1.100:8000"  // Replace with your IP
#endif
```

### 2. Add Model File

1. Add your PyTorch Mobile model file (`model_mobile_quant.pt`) to the project
2. In Xcode:
   - Drag the file into the project navigator
   - Ensure "Copy items if needed" is checked
   - Add to target: `AutoRL`
   - In Build Phases → Copy Bundle Resources, ensure the model file is included

### 3. PyTorch Mobile (Optional)

If you want to use PyTorch Mobile for iOS:

1. Install CocoaPods (if not already installed):
   ```bash
   sudo gem install cocoapods
   ```

2. Install dependencies:
   ```bash
   cd mobile/ios
   pod install
   ```

3. Open `AutoRL.xcworkspace` (not `.xcodeproj`)

4. Note: PyTorch Mobile iOS may require additional setup. The app includes a mock fallback for testing.

## Build and Run

1. Select a simulator or connected device
2. Press `Cmd + R` to build and run
3. The app should launch and show the AutoRL interface

## Testing

### Run Unit Tests

```bash
xcodebuild test -workspace AutoRL.xcworkspace -scheme AutoRL -destination 'platform=iOS Simulator,name=iPhone 15'
```

Or in Xcode:
- Press `Cmd + U` to run tests

## Troubleshooting

### Model Not Found

- Verify the model file is in the app bundle
- Check the file name matches `model_mobile_quant.pt`
- Ensure the file is included in "Copy Bundle Resources"

### Network Connection Issues

**For Simulator:**
- Use `localhost:8000` or `127.0.0.1:8000`

**For Physical Device:**
- Use your computer's IP address (e.g., `192.168.1.100:8000`)
- Ensure your device and computer are on the same network
- Check firewall settings

### App Transport Security

The `Info.plist` includes ATS exceptions for localhost. For production, configure proper SSL certificates.

### PyTorch Mobile Not Available

The app includes a mock implementation that will work without PyTorch Mobile. To use real inference:

1. Install PyTorch Mobile iOS SDK
2. Update `ModelRunner.swift` with actual PyTorch Mobile API calls
3. See PyTorch Mobile iOS documentation for details

## Project Structure

```
AutoRL/
├── AutoRLApp.swift          # App entry point
├── ContentView.swift         # Main SwiftUI views
├── MainViewModel.swift       # ViewModel (MVVM)
├── ModelRunner.swift         # Model inference
├── APIService.swift          # Backend API
├── Utils.swift               # Utilities
├── Info.plist                # App configuration
└── Assets.xcassets/          # App assets
```

## Next Steps

1. Test the app with the backend running
2. Add your PyTorch model file
3. Configure backend URL for your environment
4. Customize UI as needed
5. Add additional features (screenshot capture, device control, etc.)

## Support

For issues or questions, refer to:
- Main project README: `README.md`
- iOS-specific README: `mobile/ios/README.md`
- Android implementation: `mobile/android/` (for reference)

