#!/bin/bash

# Biopharma M&A Radar - Startup Script
# This script activates the virtual environment and starts the Streamlit app

echo "🧬 Biopharma M&A Radar - Starting Up..."
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "   Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python -c "import streamlit, requests, pandas, plotly, networkx, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

# Start the app
echo "🚀 Starting Biopharma M&A Radar..."
echo "📱 The app will open in your browser at: http://localhost:8501"
echo "🔗 If it doesn't open automatically, copy and paste the URL above"
echo ""
echo "⏹️  To stop the app, press Ctrl+C"
echo "========================================"

streamlit run app.py --server.port 8501
