# Wine Quality Analysis

A comprehensive machine learning project that analyzes and predicts wine quality using various chemical properties. This project implements both regression and classification approaches with extensive hyperparameter tuning and model evaluation.

## Features

- **Regression Analysis**: Predicts exact wine quality scores (3-8)
- **Classification Analysis**: Predicts high-quality vs low-quality wines
- **Comprehensive Model Comparison**: Tests 8 different algorithms
- **Hyperparameter Tuning**: Grid search optimization for best models
- **Feature Importance Analysis**: Identifies key chemical properties
- **Dimensionality Reduction**: PCA analysis and comparison
- **Interactive Visualizations**: Plotly charts and matplotlib plots
- **Model Persistence**: Saves trained models for future use

## Project Structure

```
wine-quality-analysis/
├── data/
│   └── winequality-red.csv          # Wine quality dataset
├── images/
│   └── .gitkeep                     # Generated visualizations
├── models/
│   └── .gitkeep                     # Saved trained models
├── outputs/                         # Terminal method output directory
│   ├── images/                      # Generated plots from CLI
│   └── models/                      # Saved models from CLI
├── utils/
│   ├── __init__.py
│   └── visualization.py             # Custom plotting functions
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt                 # Python dependencies
├── wine_quality_analysis.ipynb      # Main analysis notebook
├── wine_analysis_cli.py            # Terminal-based analysis script
├── run_analysis.sh                 # Terminal method launcher
└── start_jupyter.sh                # JupyterLab launcher
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jordanaftermidnight/wine-quality-analysis.git
   cd wine-quality-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### **Method 1: JupyterLab (Interactive)**

Run the provided script to start JupyterLab:
```bash
./start_jupyter.sh
```

Or manually start JupyterLab:
```bash
cd wine-quality-analysis
jupyter lab wine_quality_analysis.ipynb
```

**Alternative Jupyter Options:**
- Classic Jupyter Notebook: `jupyter notebook wine_quality_analysis.ipynb`
- VS Code: Open the project folder and run cells directly in VS Code

### **Method 2: Terminal (Command Line)**

Run the analysis directly from the terminal:

```bash
# Run full analysis (recommended for first time)
./run_analysis.sh

# Run only regression analysis
./run_analysis.sh regression

# Run only classification analysis
./run_analysis.sh classification

# Run with hyperparameter tuning (takes longer)
./run_analysis.sh tune

# See all options
./run_analysis.sh help
```

Or use the Python script directly:
```bash
# Basic usage - runs all analyses
python wine_analysis_cli.py

# Specific task
python wine_analysis_cli.py --task regression
python wine_analysis_cli.py --task classification

# With hyperparameter tuning
python wine_analysis_cli.py --tune

# Custom paths
python wine_analysis_cli.py --data path/to/data.csv --output path/to/output
```

The terminal method will:
- Load and analyze the wine quality dataset
- Perform exploratory data analysis with visualizations
- Train multiple regression and classification models
- Compare model performance
- Save the best models and visualizations to the `outputs/` directory

### **Expected Output**

When you run the analysis (either method), you'll see:
- **Distribution plots** of wine quality scores
- **Correlation heatmaps** between chemical features
- **Box plots** showing feature-quality relationships
- **Model performance charts** (accuracy, ROC curves, confusion matrices)
- **Feature importance visualizations**
- **PCA analysis and dimensionality reduction plots**
- **Learning curves and hyperparameter tuning results**

### **Analysis Contents**
- Data exploration and visualization
- Feature engineering and preprocessing
- Model training and evaluation (8 different algorithms)
- Comprehensive hyperparameter tuning
- Feature importance analysis
- PCA dimensionality reduction
- Results analysis and interpretation

### **Troubleshooting**
- **Kernel won't connect**: Restart Jupyter server
- **No plots visible**: Ensure `%matplotlib inline` is set
- **Import errors**: Run `pip install -r requirements.txt`
- **404 errors**: Make sure you're in the correct directory
- **Permission denied on scripts**: Run `chmod +x *.sh`

## Dataset

The project uses the **Wine Quality Dataset** from the UCI Machine Learning Repository, which contains:
- **1,599 red wine samples**
- **11 chemical properties** (fixed acidity, volatile acidity, citric acid, etc.)
- **Quality ratings** from 3 to 8 (scored by wine experts)

### Key Features Analyzed:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol content

## Results

The analysis reveals that:
- **Random Forest** and **Gradient Boosting** perform best for both tasks
- **Alcohol content**, **volatile acidity**, and **sulphates** are the most important features
- **86.4%** of wines are low quality, **13.6%** are high quality
- Models achieve **~0.65 R²** for regression and **~0.85 F1-score** for classification
- **PCA analysis** shows minimal performance impact, indicating all features are valuable

## Model Performance
- **Regression**: RMSE ~0.65, R² ~0.65
- **Classification**: F1 Score ~0.85, ROC AUC ~0.90
- **Cross-validation**: 5-fold CV with systematic hyperparameter tuning

## Contributors

- Jordan After Midnight - Creator and Main Developer

## License

This project is licensed under the MIT License - see the LICENSE file for details.