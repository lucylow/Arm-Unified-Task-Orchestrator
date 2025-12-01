# AutoRL Mobile Simulation Components

## 🎬 Cinematic Mobile UI for Agent Demonstrations

This directory contains a complete, production-ready mobile simulation UI that makes your AutoRL demos look like real, native mobile automation — not boring web iframes.

## 📁 Components

### Core Components

1. **DeviceSimulator.jsx** - Realistic phone chrome with notch, bezel, and shadow
2. **MobileScreen.jsx** - Renders multiple app UIs with action API
3. **GestureOverlay.jsx** - Animated tap ripples, swipe trails, typing indicators
4. **AttentionHeatmap.jsx** - Shows AI attention/confidence visualization
5. **OCROverlay.jsx** - Displays detected UI elements with labels
6. **MobileController.jsx** - Orchestrates simulated runs and recovery flows
7. **CrossAppDemo.jsx** - Multi-app workflow demonstration
8. **DashboardIntegration.jsx** - Complete integration example with WebSocket

## 🚀 Quick Start

### Installation

```bash
# Already installed if you have framer-motion
npm install framer-motion
```

### Basic Usage

```jsx
import { DeviceSimulator, MobileScreen, MobileController } from './components/mobile';
import { useRef, useState } from 'react';

function Dashboard() {
  const mobileRef = useRef(null);
  const [events, setEvents] = useState([]);
  
  const onEvent = (ev) => setEvents(s => [ev, ...s].slice(0, 120));
  
  return (
    <div style={{ display: 'flex', gap: 20 }}>
      <DeviceSimulator>
        <MobileScreen 
          ref={mobileRef} 
          initialApp="banking_v1"
          onAction={(a) => console.log('action', a)}
        />
      </DeviceSimulator>
      
      <MobileController mobileRef={mobileRef} onEvent={onEvent} />
    </div>
  );
}
```

### Complete Integration

```jsx
import MobileDemoPanel from './components/mobile/DashboardIntegration';

function Dashboard() {
  return (
    <div>
      {/* Replace your iframe with: */}
      <MobileDemoPanel mode="single" />
      
      {/* Or for cross-app demo: */}
      <MobileDemoPanel mode="cross-app" />
    </div>
  );
}
```

## 📱 Available Apps

### Banking V1
- Traditional bank interface
- Send Money button
- Recipient and amount inputs
- Confirm transaction flow

### Banking V2
- Updated interface (for app update demos)
- Pay and Quick Transfer buttons
- Different field layouts
- Demonstrates semantic search and recovery

### Calendar
- Event display
- Send Invite functionality
- Cross-app workflow trigger

### Chat
- Message display
- System notifications
- Conversation UI

### Email
- Compose functionality
- To, Subject, Body fields
- Send email action

## 🎮 MobileScreen API

```javascript
// Access via ref
const mobileRef = useRef(null);

// Tap at coordinates
await mobileRef.current.tap(x, y);

// Type text into element
await mobileRef.current.type('elementId', 'text');

// Swipe gesture
await mobileRef.current.swipe({ x: 100, y: 200 }, { x: 100, y: 400 });

// Get element bounding box
const box = mobileRef.current.getElementBox('send-money');
// Returns: { x, y, w, h }

// Switch app
mobileRef.current.switchApp('banking_v2');
```

## 🎨 Visual Features

### Gesture Animations
Trigger visual gestures programmatically:

```javascript
// Tap animation
window.dispatchEvent(new CustomEvent('autorlgesture', { 
  detail: { type: 'tap', x: 180, y: 320 } 
}));

// Swipe animation
window.dispatchEvent(new CustomEvent('autorlgesture', { 
  detail: { type: 'swipe', x: 100, y: 300, to: { x: 200, y: 300 } } 
}));

// Typing indicator
window.dispatchEvent(new CustomEvent('autorlgesture', { 
  detail: { type: 'type', x: 180, y: 320, text: '$20.00' } 
}));
```

### Attention Heatmap
Pass confidence data to visualize AI attention:

```javascript
<MobileScreen 
  ref={mobileRef}
  attentionData={{
    'send-money': 0.95,
    'recipient': 0.87,
    'amount': 0.92
  }}
/>
```

### OCR Overlay
Automatically displays detected elements with confidence scores.

## 🔌 WebSocket Integration

The `DashboardIntegration` component includes full WebSocket support:

```javascript
// Backend sends:
{
  "type": "task_update",
  "task_id": "task_123",
  "status": "executing",
  "progress": 0.65
}

// Frontend receives and displays in event log
```

### Supported WebSocket Messages

- `connection_established` - Initial connection
- `task_update` - Task progress updates
- `attention` - Attention heatmap data
- `gesture` - Trigger visual gestures
- `heartbeat` - Keep-alive ping

## 🎭 Demo Scenarios

### 1. Basic Mobile Automation
```jsx
<MobileController mobileRef={mobileRef} onEvent={onEvent} />
// Click "Run Mobile Demo"
```

### 2. App Update Recovery
```jsx
// Click "Run + Inject App Update"
// Simulates UI changing mid-task
// Agent performs semantic search and recovery
```

### 3. Error Recovery
```jsx
// Click "Run + Inject Error"
// Shows error detection and recovery flow
```

### 4. Cross-App Workflow
```jsx
<MobileDemoPanel mode="cross-app" />
// Demonstrates Calendar → Chat → Email coordination
```

## 🎨 Styling

All styles are in `mobile.css`. Key classes:

- `.device-shell` - Phone container
- `.device-bezel` - Phone frame
- `.device-screen` - Screen content area
- `.mobile-primary` - Primary action button
- `.mobile-secondary` - Secondary button
- `.mobile-input` - Text input field
- `.mobile-confirm` - Confirmation button
- `.attention-overlay` - Heatmap styling
- `.ocr-label` - OCR label styling
- `.gesture-ripple` - Tap animation
- `.ai-brain` - Brain visualization

## 🔧 Customization

### Adding New Apps

Edit `MobileScreen.jsx`:

```javascript
const APPS = {
  your_app: {
    id: 'your_app',
    title: 'Your App Title',
    primaryButton: { id: 'action-btn', label: 'Action' },
    fields: [
      { id: 'field1', placeholder: 'Field 1', value: '' }
    ]
  }
};
```

### Customizing Gestures

Edit `GestureOverlay.jsx` to add new animation types.

### Adjusting Attention Visualization

Edit `AttentionHeatmap.jsx` to change colors, opacity, or animation.

## 📊 Event Log

Events are logged with the following structure:

```javascript
{
  event: 'perception',          // Event type
  text: 'Capturing screenshot', // Description
  timestamp: Date.now(),         // Timestamp
  plan: ['step1', 'step2'],     // Optional: action plan
  success: true                  // Optional: success flag
}
```

### Event Types

- `perception` - Screenshot/OCR capture
- `planning` - LLM planning phase
- `action` - Agent action execution
- `completed` - Task completed
- `error` - Error occurred
- `warning` - Warning message
- `recovery_*` - Recovery flow stages
- `memory_saved` - Episode saved
- `cross_app_start` - Cross-app workflow
- `app_update` - App UI changed
- `app_switch` - Switched apps

## 🎬 Competition Demo Tips

### For Hackathon Judges
1. Start with "Run Mobile Demo" to show basic automation
2. Click "Force App Update" mid-run to demonstrate adaptation
3. Use "Run + Inject Error" to showcase recovery
4. Switch to "Cross-App Demo" for multi-app coordination

### Keyboard Shortcuts (Optional)
Add to your Dashboard component:

```javascript
useEffect(() => {
  const handler = (e) => {
    if (e.code === 'Space') runSim();
    if (e.key === 'u') mobileRef.current.switchApp('banking_v2');
    if (e.key === 'c') setMode('cross-app');
  };
  window.addEventListener('keydown', handler);
  return () => window.removeEventListener('keydown', handler);
}, []);
```

## 🔄 Real Device Integration

To replace simulation with real device stream:

1. Capture screenshots from Appium
2. Convert to base64
3. Replace `MobileScreen` content with `<img src={base64} />`
4. Overlay gesture animations on top
5. Use element coordinates from Appium

## 🐛 Troubleshooting

### Gestures not appearing
- Check console for `autorlgesture` events
- Ensure coordinates are relative to device screen
- Verify element exists in current app

### WebSocket not connecting
- Check backend is running on port 8000
- Verify `/ws/metrics` endpoint exists
- Check browser console for errors

### Attention heatmap not showing
- Pass `attentionData` prop with element IDs
- Ensure elements are registered in `elementsRef`
- Check element IDs match

### App switching not working
- Verify app ID exists in `APPS` object
- Check `mobileRef.current` is not null
- Use correct app ID string

## 📈 Performance

- Animations use CSS transforms for GPU acceleration
- Event log limits to 120 entries
- Heatmap updates throttled to 600ms
- WebSocket reconnects automatically

## 🎯 Next Steps

1. Add more app templates (travel, shopping, social)
2. Integrate with real Appium device stream
3. Add voice command visualization
4. Create animated memory core visualization
5. Add progress rings around brain visualization

## 📞 Support

For issues or questions, check:
- `DEPLOYMENT_GUIDE.md` for backend setup
- `CRITICAL_FIXES_SUMMARY.md` for system overview
- Console logs for debugging information

---

**🎉 Your AutoRL mobile demo is now cinema-ready!**

