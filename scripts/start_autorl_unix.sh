#!/bin/bash
# AutoRL Unix/Linux/Mac Startup Script
# Starts backend and frontend in separate terminal tabs/windows

echo "====================================="
echo "🚀 AutoRL Startup Script"
echo "====================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found!${NC}"
    if [ -f ".env.example" ]; then
        echo -e "${CYAN}📝 Creating .env from .env.example...${NC}"
        cp .env.example .env
        echo -e "${GREEN}✅ .env created! Please edit it with your API keys.${NC}"
        echo -e "${YELLOW}   Then run this script again.${NC}"
        echo ""
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${CYAN}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${CYAN}📦 Installing Node.js dependencies...${NC}"
    npm install
    echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
fi

echo ""
echo -e "${CYAN}🔧 Starting AutoRL services...${NC}"
echo ""

# Detect OS and terminal
OS_TYPE="$(uname -s)"

# Function to start backend
start_backend() {
    echo -e "${GREEN}1️⃣  Starting Backend Server (Port 5000)...${NC}"
    
    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS - use osascript to open new Terminal tab
        osascript -e 'tell application "Terminal" to do script "cd '"$PWD"' && source venv/bin/activate && export AUTORL_MODE=demo && python backend_server.py"'
    elif command -v gnome-terminal &> /dev/null; then
        # Linux with gnome-terminal
        gnome-terminal --tab --title="AutoRL Backend" -- bash -c "cd $PWD && source venv/bin/activate && export AUTORL_MODE=demo && python backend_server.py; exec bash"
    elif command -v xterm &> /dev/null; then
        # Fallback to xterm
        xterm -T "AutoRL Backend" -e "cd $PWD && source venv/bin/activate && export AUTORL_MODE=demo && python backend_server.py" &
    else
        # No GUI terminal, run in background
        echo -e "${YELLOW}   Running backend in background...${NC}"
        source venv/bin/activate
        export AUTORL_MODE=demo
        python backend_server.py > backend.log 2>&1 &
        BACKEND_PID=$!
        echo -e "${GREEN}   Backend PID: $BACKEND_PID${NC}"
    fi
}

# Function to start frontend
start_frontend() {
    echo -e "${GREEN}2️⃣  Starting Frontend (Port 5173)...${NC}"
    
    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS
        osascript -e 'tell application "Terminal" to do script "cd '"$PWD"' && npm run dev"'
    elif command -v gnome-terminal &> /dev/null; then
        # Linux with gnome-terminal
        gnome-terminal --tab --title="AutoRL Frontend" -- bash -c "cd $PWD && npm run dev; exec bash"
    elif command -v xterm &> /dev/null; then
        # Fallback to xterm
        xterm -T "AutoRL Frontend" -e "cd $PWD && npm run dev" &
    else
        # No GUI terminal, run in background
        echo -e "${YELLOW}   Running frontend in background...${NC}"
        npm run dev > frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo -e "${GREEN}   Frontend PID: $FRONTEND_PID${NC}"
    fi
}

# Start services
start_backend
sleep 3
start_frontend
sleep 3

echo ""
echo "====================================="
echo -e "${GREEN}✅ AutoRL Started Successfully!${NC}"
echo "====================================="
echo ""
echo -e "${CYAN}📊 Access Points:${NC}"
echo "   Frontend:  http://localhost:5173"
echo "   API:       http://localhost:5000/api/health"
echo "   WebSocket: ws://localhost:5000/ws"
echo "   API Docs:  http://localhost:5000/docs"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "   - Running in DEMO mode (no real devices needed)"
echo "   - Close the terminal tabs to stop services"
echo "   - Check .env file to configure API keys"
echo ""

# Try to open browser (different commands for different OS)
echo -e "${CYAN}Opening frontend in browser...${NC}"
sleep 2

if [ "$OS_TYPE" = "Darwin" ]; then
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173
elif command -v firefox &> /dev/null; then
    firefox http://localhost:5173 &
elif command -v google-chrome &> /dev/null; then
    google-chrome http://localhost:5173 &
fi

echo ""
echo -e "${GREEN}✅ Done! AutoRL is running.${NC}"
echo ""

