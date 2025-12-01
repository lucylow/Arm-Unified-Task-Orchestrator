#!/bin/bash
# Start OMH Mock Server

echo "Starting OMH Mock Server..."
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r omh_mock_requirements.txt
fi

# Start the server
echo ""
echo "Server will be available at:"
echo "  - API: http://localhost:8001"
echo "  - Docs: http://localhost:8001/docs"
echo "  - Mock Tokens: http://localhost:8001/mock/tokens"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

python3 omh_mock_server.py

