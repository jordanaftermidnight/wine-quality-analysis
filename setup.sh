#!/bin/bash
# Wine Quality Analysis - Setup and Installation Script

set -e  # Exit on error

echo "======================================"
echo "Wine Quality Analysis Setup"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Check if Python 3 is installed
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python 3 found: $PYTHON_VERSION"
    
    # Check Python version is 3.7 or higher
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.7 or higher is required"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    echo "Please install Python 3.7 or higher from https://www.python.org/"
    exit 1
fi

# Check if pip is installed
echo ""
echo "Checking pip installation..."
if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    print_status "pip is installed"
    PIP_CMD=$(command -v pip3 || command -v pip)
else
    print_error "pip is not installed"
    echo "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi

# Function to clone repository
clone_repository() {
    echo ""
    echo "Cloning Wine Quality Analysis repository..."
    
    if command -v git &> /dev/null; then
        print_status "Git is installed"
        
        # Clone the repository
        git clone https://github.com/jordanaftermidnight/wine-quality-analysis.git
        
        if [ $? -eq 0 ]; then
            print_status "Repository cloned successfully"
            cd wine-quality-analysis
        else
            print_error "Failed to clone repository"
            exit 1
        fi
    else
        print_error "Git is not installed"
        echo "Please install Git from https://git-scm.com/"
        exit 1
    fi
}

# Check if we're already in the project directory
if [ -f "wine_quality_analysis.ipynb" ] && [ -f "requirements.txt" ]; then
    print_status "Already in project directory"
elif [ -d "wine-quality-analysis" ]; then
    print_status "Project directory exists, entering..."
    cd wine-quality-analysis
else
    clone_repository
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

# Install with progress bar
pip install -r requirements.txt --progress-bar on

# Verify installations
echo ""
echo "Verifying installations..."

# Function to check if a Python package is installed
check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        print_status "$1 installed successfully"
        return 0
    else
        print_error "$1 installation failed"
        return 1
    fi
}

# Check critical packages
FAILED=0
for package in numpy pandas sklearn matplotlib seaborn plotly jupyter joblib; do
    check_package $package || FAILED=1
done

if [ $FAILED -eq 1 ]; then
    print_error "Some packages failed to install"
    echo "Please check the error messages above and try again"
    exit 1
fi

# Download dataset if not present
echo ""
echo "Checking dataset..."
if [ ! -f "data/winequality-red.csv" ]; then
    print_warning "Dataset not found"
    echo "Please ensure data/winequality-red.csv exists before running analysis"
else
    print_status "Dataset found"
fi

# Make scripts executable
echo ""
echo "Setting up execution permissions..."
chmod +x run_analysis.sh start_jupyter.sh setup.sh 2>/dev/null || true
print_status "Scripts are now executable"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/images outputs/models images models
print_status "Output directories created"

# Setup complete
echo ""
echo "======================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. To run the analysis in terminal:"
echo "   ./run_analysis.sh"
echo ""
echo "2. To start JupyterLab:"
echo "   ./start_jupyter.sh"
echo ""
echo "3. To see all options:"
echo "   ./run_analysis.sh help"
echo ""
echo "Note: Virtual environment is activated. To deactivate, type: deactivate"