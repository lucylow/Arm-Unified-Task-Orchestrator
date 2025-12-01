# AutoRL Windows Startup Script
# Starts backend and frontend in separate PowerShell windows

Write-Host "=====================================" -ForegroundColor Blue
Write-Host "🚀 AutoRL Startup Script" -ForegroundColor Blue
Write-Host "=====================================" -ForegroundColor Blue
Write-Host ""

# Check if .env exists
if (-Not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found!" -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Write-Host "📝 Creating .env from .env.example..." -ForegroundColor Cyan
        Copy-Item ".env.example" ".env"
        Write-Host "✅ .env created! Please edit it with your API keys." -ForegroundColor Green
        Write-Host "   Then run this script again." -ForegroundColor Yellow
        Write-Host ""
        Pause
        exit
    }
}

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Check if node_modules exists
if (-Not (Test-Path "node_modules")) {
    Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Cyan
    npm install
    Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔧 Starting AutoRL services..." -ForegroundColor Cyan
Write-Host ""

# Start Backend in new window
Write-Host "1️⃣  Starting Backend Server (Port 5000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; `$env:AUTORL_MODE='demo'; python backend_server.py"

# Wait for backend to start
Write-Host "   Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start Frontend in new window
Write-Host "2️⃣  Starting Frontend (Port 5173)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; npm run dev"

# Wait for frontend to start
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=====================================" -ForegroundColor Blue
Write-Host "✅ AutoRL Started Successfully!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Blue
Write-Host ""
Write-Host "📊 Access Points:" -ForegroundColor Cyan
Write-Host "   Frontend:  http://localhost:5173" -ForegroundColor White
Write-Host "   API:       http://localhost:5000/api/health" -ForegroundColor White
Write-Host "   WebSocket: ws://localhost:5000/ws" -ForegroundColor White
Write-Host "   API Docs:  http://localhost:5000/docs" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Running in DEMO mode (no real devices needed)" -ForegroundColor White
Write-Host "   - Close the PowerShell windows to stop services" -ForegroundColor White
Write-Host "   - Check .env file to configure API keys" -ForegroundColor White
Write-Host ""
Write-Host "Opening frontend in browser..." -ForegroundColor Cyan

# Wait a bit more for frontend to fully start
Start-Sleep -Seconds 2

# Open browser
Start-Process "http://localhost:5173"

Write-Host ""
Write-Host "✅ Done! AutoRL is running." -ForegroundColor Green
Write-Host ""
Pause

