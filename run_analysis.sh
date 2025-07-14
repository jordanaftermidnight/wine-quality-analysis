#!/bin/bash
# Wine Quality Analysis Launcher Script

echo "======================================"
echo "Wine Quality Analysis - Terminal Mode"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Installing dependencies globally..."
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Run the analysis
echo ""
echo "Starting analysis..."
echo ""

# Default: run all analyses
if [ "$1" == "help" ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    python wine_analysis_cli.py --help
elif [ "$1" == "regression" ]; then
    echo "Running regression analysis only..."
    python wine_analysis_cli.py --task regression
elif [ "$1" == "classification" ]; then
    echo "Running classification analysis only..."
    python wine_analysis_cli.py --task classification
elif [ "$1" == "tune" ]; then
    echo "Running full analysis with hyperparameter tuning..."
    python wine_analysis_cli.py --tune
else
    echo "Running full analysis (regression + classification)..."
    python wine_analysis_cli.py
fi

echo ""
echo "Analysis complete! Check the 'outputs' directory for results."