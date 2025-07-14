#!/bin/bash
# JupyterLab Launcher Script

echo "======================================"
echo "Wine Quality Analysis - JupyterLab"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python..."
fi

# Check if JupyterLab is installed
if ! command -v jupyter &> /dev/null; then
    echo "Jupyter not found. Installing..."
    pip install jupyter jupyterlab
fi

# Install other dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "Starting JupyterLab..."
echo "Opening browser at http://localhost:8888"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start JupyterLab
jupyter lab --port=8888 wine_quality_analysis.ipynb