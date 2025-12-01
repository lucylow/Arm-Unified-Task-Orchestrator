# AutoRL iOS Application

iOS mobile application for the AutoRL ARM Unified Task Orchestrator, built with Swift and SwiftUI.

## Features

- **MVVM Architecture**: Clean separation of concerns with ViewModels and Views
- **PyTorch Mobile Integration**: On-device AI model inference using PyTorch Mobile
- **ARM Optimization**: Optimized for ARM64 architecture (Apple Silicon)
- **Modern UI**: SwiftUI-based interface matching Android app functionality
- **Backend Integration**: REST API communication with the AutoRL backend
- **Real-time Updates**: Support for WebSocket connections (future enhancement)

## Requirements

- iOS 15.0+
- Xcode 14.0+
- Swift 5.9+
- CocoaPods (for dependency management)

## Project Structure

```
AutoRL/
├── AutoRLApp.swift          # App entry point
├── ContentView.swift         # Main SwiftUI views
├── MainViewModel.swift       # ViewModel for state management
├── ModelRunner.swift         # PyTorch model inference
├── APIService.swift          # Backend API communication
├── Utils.swift               # Utility functions
└── Info.plist               # App configuration
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd mobile/ios
pod install
```

### 2. Open Project

```bash
open AutoRL.xcworkspace
```

**Note**: Use `.xcworkspace` (not `.xcodeproj`) when using CocoaPods.

### 3. Configure Backend URL

Edit `APIService.swift` to set the correct backend URL:

```swift
#if targetEnvironment(simulator)
self.baseURL = "http://localhost:8000"
#else
// For physical device, use your computer's IP address
self.baseURL = "http://YOUR_IP_ADDRESS:8000"
#endif
```

### 4. Add Model File

1. Add your PyTorch Mobile model (`model_mobile_quant.pt`) to the Xcode project
2. Ensure it's included in the app bundle (check "Copy Bundle Resources" in Build Phases)

### 5. Build and Run

1. Select a simulator or connected device
2. Press `Cmd + R` to build and run

## Architecture

### MVVM Pattern

- **View**: SwiftUI views (`ContentView`, `MainView`)
- **ViewModel**: `MainViewModel` manages state and business logic
- **Model**: `ModelRunner` handles model inference, `APIService` handles networking

### Key Components

#### ModelRunner
- Singleton actor for thread-safe model operations
- Handles PyTorch Mobile model loading and inference
- Preprocesses images and postprocesses model outputs
- Falls back to mock implementation if PyTorch Mobile is unavailable

#### APIService
- Handles all backend API communication
- Uses async/await for modern Swift concurrency
- Supports task execution, device management, and analytics

#### MainViewModel
- ObservableObject for SwiftUI binding
- Manages UI state (loading, error, success)
- Coordinates between ModelRunner and APIService

## API Integration

The app communicates with the backend at `http://localhost:8000` (or configured URL).

### Endpoints Used

- `GET /health` - Health check
- `POST /api/v1/execute` - Execute a task
- `GET /api/v1/devices` - List devices
- `GET /api/v1/analytics` - Get analytics

## Model Inference

The app uses PyTorch Mobile for on-device inference:

1. **Model Loading**: Loads `model_mobile_quant.pt` from app bundle
2. **Preprocessing**: Resizes and normalizes input images
3. **Inference**: Runs model on-device using ARM-optimized operations
4. **Postprocessing**: Converts output to readable results

### Fallback Mode

If PyTorch Mobile is not available, the app uses a mock implementation for testing.

## Development

### Running Tests

```bash
# Run unit tests
xcodebuild test -workspace AutoRL.xcworkspace -scheme AutoRL -destination 'platform=iOS Simulator,name=iPhone 15'
```

### Debugging

- Use Xcode's debugger for breakpoints
- Check console logs for model loading and inference
- Use Network Link Conditioner to test network scenarios

## Troubleshooting

### Model Not Loading

1. Verify model file is in the app bundle
2. Check file name matches `model_mobile_quant.pt`
3. Check console logs for specific error messages

### Network Connection Issues

1. For simulator: Use `localhost:8000`
2. For physical device: Use your computer's IP address
3. Check backend is running and accessible
4. Verify ATS (App Transport Security) settings in `Info.plist`

### PyTorch Mobile Not Available

The app includes fallback mock implementation. To use PyTorch Mobile:

1. Add PyTorch Mobile pod to `Podfile`
2. Run `pod install`
3. Import PyTorch Mobile in `ModelRunner.swift`

## Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Screenshot capture and analysis
- [ ] Device control capabilities
- [ ] Core ML integration as alternative to PyTorch Mobile
- [ ] Enhanced error handling and retry logic
- [ ] Offline mode support

## License

Same as the main AutoRL project.

