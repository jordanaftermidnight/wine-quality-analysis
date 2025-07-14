#!/usr/bin/env python3
"""
Wine Quality Analysis - Command Line Interface
==============================================

This script provides a terminal-based method to run the wine quality analysis
without requiring Jupyter notebook.

Author: Jordan After Midnight
"""

import sys
import os
import argparse
import logging
import yaml
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wine_analysis.log'),
        logging.StreamHandler()
    ]
)

# Import custom utilities
from utils.visualization import save_fig, set_plotting_style, plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# Configuration constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
QUALITY_THRESHOLD = 6  # For binary classification
N_JOBS = -1  # Use all CPU cores

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

class WineQualityAnalyzer:
    """
    Main class for wine quality analysis.
    
    This class handles the complete wine quality analysis pipeline including:
    - Data loading and exploration
    - Feature preprocessing and scaling
    - Training multiple regression and classification models
    - Hyperparameter tuning
    - Feature importance analysis
    - PCA dimensionality reduction analysis
    - Model persistence and report generation
    
    Attributes:
        data_path (str): Path to the wine quality dataset
        output_dir (str): Directory for saving outputs
        df (pd.DataFrame): Loaded dataset
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        scaler (StandardScaler): Feature scaler
        regression_results (dict): Results from regression models
        classification_results (dict): Results from classification models
    """
    
    def __init__(self, data_path='data/winequality-red.csv', output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and display basic information about the dataset.
        
        Returns:
            pd.DataFrame: Loaded wine quality dataset
            
        Raises:
            FileNotFoundError: If dataset file is not found
            pd.errors.EmptyDataError: If dataset is empty
            Exception: For other loading errors
        """
        try:
            logging.info(f"Loading dataset from {self.data_path}")
            print("Loading wine quality dataset...")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
            self.df = pd.read_csv(self.data_path, sep=';')
            
            if self.df.empty:
                raise pd.errors.EmptyDataError("Dataset is empty")
            
            # Validate expected columns
            expected_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 
                           'residual sugar', 'chlorides', 'free sulfur dioxide',
                           'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                           'alcohol', 'quality']
            missing_cols = set(expected_cols) - set(self.df.columns)
            if missing_cols:
                logging.warning(f"Missing expected columns: {missing_cols}")
            
            print(f"Dataset loaded successfully! Shape: {self.df.shape}")
            print("\nDataset Info:")
            print(self.df.info())
            print("\nFirst 5 rows:")
            print(self.df.head())
            print("\nBasic Statistics:")
            print(self.df.describe())
            
            logging.info(f"Dataset loaded successfully with shape {self.df.shape}")
            return self.df
            
        except FileNotFoundError as e:
            logging.error(f"Dataset file not found: {e}")
            print(f"\nError: {e}")
            print(f"Please ensure the dataset exists at: {self.data_path}")
            sys.exit(1)
        except pd.errors.EmptyDataError as e:
            logging.error(f"Empty dataset: {e}")
            print(f"\nError: The dataset file is empty")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            print(f"\nError loading dataset: {e}")
            sys.exit(1)
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Check for missing values
        print("\nMissing values per column:")
        print(self.df.isnull().sum())
        
        # Quality distribution
        print("\nWine Quality Distribution:")
        print(self.df['quality'].value_counts().sort_index())
        
        # Create quality distribution plot
        plt.figure(figsize=(10, 6))
        self.df['quality'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Wine Quality Ratings')
        plt.xlabel('Quality Rating')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'images', 'quality_distribution.png'))
        plt.close()
        print("\nSaved quality distribution plot to outputs/images/quality_distribution.png")
        
        # Correlation analysis
        print("\nCalculating correlations...")
        correlations = self.df.corr()['quality'].sort_values(ascending=False)
        print("\nCorrelations with Quality:")
        print(correlations)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'images', 'correlation_heatmap.png'))
        plt.close()
        print("\nSaved correlation heatmap to outputs/images/correlation_heatmap.png")
        
    def prepare_data(self, task='regression', handle_imbalance=True):
        """
        Prepare data for modeling with optional class imbalance handling.
        
        Args:
            task (str): Type of task ('regression' or 'classification')
            handle_imbalance (bool): Whether to handle class imbalance for classification
        """
        try:
            print(f"\nPreparing data for {task}...")
            logging.info(f"Preparing data for {task} task")
            
            # Separate features and target
            self.X = self.df.drop('quality', axis=1)
            
            if task == 'classification':
                # Convert to binary classification: low quality (3-5) vs high quality (6-8)
                self.y = (self.df['quality'] >= QUALITY_THRESHOLD).astype(int)
                
                # Calculate class distribution
                class_dist = self.y.value_counts()
                imbalance_ratio = class_dist[0] / class_dist[1]
                
                print(f"Binary classification - Low quality (0): {class_dist[0]} ({class_dist[0]/len(self.y)*100:.1f}%)")
                print(f"High quality (1): {class_dist[1]} ({class_dist[1]/len(self.y)*100:.1f}%)")
                print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
                
                logging.info(f"Class distribution - 0: {class_dist[0]}, 1: {class_dist[1]}")
            else:
                self.y = self.df['quality']
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=self.y
            )
            
            # Scale the features
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Handle class imbalance for classification
            if task == 'classification' and handle_imbalance:
                print("\nHandling class imbalance with SMOTE...")
                smote = SMOTE(random_state=RANDOM_SEED)
                self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
                    self.X_train_scaled, self.y_train
                )
                print(f"Balanced training set size: {self.X_train_balanced.shape}")
                print(f"Balanced class distribution: {pd.Series(self.y_train_balanced).value_counts()}")
                
                # Store both balanced and unbalanced versions
                self.use_balanced = True
            else:
                self.use_balanced = False
                self.X_train_balanced = self.X_train_scaled
                self.y_train_balanced = self.y_train
            
            print(f"\nOriginal training set size: {self.X_train.shape}")
            print(f"Test set size: {self.X_test.shape}")
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            print(f"\nError preparing data: {e}")
            raise
        
    def train_regression_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple regression models with progress tracking.
        
        Returns:
            Dict containing model results
        """
        print("\n" + "="*50)
        print("TRAINING REGRESSION MODELS")
        print("="*50)
        logging.info("Starting regression model training")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=RANDOM_SEED),
            'Lasso Regression': Lasso(random_state=RANDOM_SEED),
            'ElasticNet': ElasticNet(random_state=RANDOM_SEED),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=N_JOBS),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor()
        }
        
        results = {}
        
        # Use progress bar for model training
        for name, model in tqdm(models.items(), desc="Training models", unit="model"):
            try:
                print(f"\nTraining {name}...")
                logging.info(f"Training {name}")
                
                # Train the model
                model.fit(self.X_train_scaled, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                # Cross-validation score with progress
                print(f"  Performing cross-validation...")
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=CV_FOLDS, scoring='r2', n_jobs=N_JOBS)
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R²: {r2:.4f}")
                print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                logging.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
                print(f"  Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.regression_results = results
        return results
    
    def train_classification_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple classification models with class imbalance handling.
        
        Returns:
            Dict containing model results
        """
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*50)
        logging.info("Starting classification model training")
        
        # Calculate class weights for models that support it
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(self.y_train), y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Define models with class weight support
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_SEED, n_jobs=N_JOBS, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_SEED
            ),
            'SVC': SVC(
                probability=True, random_state=RANDOM_SEED, class_weight='balanced'
            ),
            'KNN': KNeighborsClassifier()
        }
        
        results = {}
        
        # Decide which training data to use
        X_train = self.X_train_balanced if self.use_balanced else self.X_train_scaled
        y_train = self.y_train_balanced if self.use_balanced else self.y_train
        
        print(f"\nUsing {'balanced' if self.use_balanced else 'original'} training data")
        
        # Use progress bar for model training
        for name, model in tqdm(models.items(), desc="Training models", unit="model"):
            try:
                print(f"\nTraining {name}...")
                logging.info(f"Training {name}")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test_scaled)
                y_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                
                # Cross-validation score with progress
                print(f"  Performing cross-validation...")
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=CV_FOLDS, scoring='f1', n_jobs=N_JOBS)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1': f1,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'classification_report': classification_report(self.y_test, y_pred),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                logging.info(f"{name} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
                print(f"  Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.classification_results = results
        return results
    
    def hyperparameter_tuning(self, model_name='Random Forest', task='regression'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\n" + "="*50)
        print(f"HYPERPARAMETER TUNING - {model_name} ({task})")
        print("="*50)
        
        if task == 'regression':
            if model_name == 'Random Forest':
                model = RandomForestRegressor(random_state=RANDOM_SEED)
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingRegressor(random_state=RANDOM_SEED)
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
        else:  # classification
            if model_name == 'Random Forest':
                model = RandomForestClassifier(random_state=RANDOM_SEED)
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(random_state=RANDOM_SEED)
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
        
        # Perform grid search
        scoring = 'r2' if task == 'regression' else 'f1'
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test_scaled)
        
        if task == 'regression':
            test_score = r2_score(self.y_test, y_pred)
            print(f"Test R² score: {test_score:.4f}")
        else:
            test_score = f1_score(self.y_test, y_pred)
            print(f"Test F1 score: {test_score:.4f}")
        
        return grid_search
    
    def feature_importance_analysis(self, task='regression'):
        """Analyze feature importance"""
        print(f"\n" + "="*50)
        print(f"FEATURE IMPORTANCE ANALYSIS ({task})")
        print("="*50)
        
        if task == 'regression':
            model = self.regression_results['Random Forest']['model']
        else:
            model = self.classification_results['Random Forest']['model']
        
        # Get feature importance
        importance = model.feature_importances_
        features = self.X.columns
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        print("\nFeature Importance Ranking:")
        for i in range(len(features)):
            print(f"{i+1}. {features[indices[i]]}: {importance[indices[i]]:.4f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [features[i] for i in indices], rotation=45, ha='right')
        plt.title(f'Feature Importance - Random Forest ({task.capitalize()})')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'images', f'feature_importance_{task}.png'))
        plt.close()
        print(f"\nSaved feature importance plot to outputs/images/feature_importance_{task}.png")
    
    def pca_analysis(self, task='regression'):
        """Compare model performance with and without PCA"""
        print(f"\n" + "="*50)
        print(f"PCA ANALYSIS ({task})")
        print("="*50)
        
        # Determine target variable
        if task == 'classification':
            y = (self.df['quality'] >= 6).astype(int)
        else:
            y = self.df['quality']
        
        # Get the best model type
        if task == 'regression':
            best_model_name = max(self.regression_results.items(), key=lambda x: x[1]['r2'])[0]
            if 'Random Forest' in best_model_name:
                model_class = RandomForestRegressor
            else:
                model_class = GradientBoostingRegressor
            scoring = 'r2'
        else:
            best_model_name = max(self.classification_results.items(), key=lambda x: x[1]['f1'])[0]
            if 'Random Forest' in best_model_name:
                model_class = RandomForestClassifier
            else:
                model_class = GradientBoostingClassifier
            scoring = 'f1'
        
        # Test different numbers of components
        n_components_list = [2, 3, 5, 7, 9, 11]
        results = []
        
        for n_comp in n_components_list:
            if n_comp == 11:  # Original features
                X_train_pca = self.X_train_scaled
                X_test_pca = self.X_test_scaled
                variance_ratio = 1.0
            else:
                pca = PCA(n_components=n_comp)
                X_train_pca = pca.fit_transform(self.X_train_scaled)
                X_test_pca = pca.transform(self.X_test_scaled)
                variance_ratio = pca.explained_variance_ratio_.sum()
            
            # Train model
            model = model_class(random_state=RANDOM_SEED)
            cv_scores = cross_val_score(model, X_train_pca, self.y_train, cv=5, scoring=scoring)
            model.fit(X_train_pca, self.y_train)
            
            # Test performance
            if task == 'regression':
                test_score = r2_score(self.y_test, model.predict(X_test_pca))
            else:
                test_score = f1_score(self.y_test, model.predict(X_test_pca))
            
            results.append({
                'n_components': n_comp,
                'variance_explained': variance_ratio,
                'cv_score': cv_scores.mean(),
                'test_score': test_score
            })
            
            print(f"\nComponents: {n_comp} (Variance: {variance_ratio:.3f})")
            print(f"  CV {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test {scoring}: {test_score:.4f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        components = [r['n_components'] for r in results]
        cv_scores = [r['cv_score'] for r in results]
        test_scores = [r['test_score'] for r in results]
        
        plt.plot(components, cv_scores, 'b-o', label='CV Score')
        plt.plot(components, test_scores, 'r-s', label='Test Score')
        plt.xlabel('Number of Components')
        plt.ylabel(f'{scoring.upper()} Score')
        plt.title(f'PCA Analysis - {best_model_name} ({task.capitalize()})')
        plt.legend()
        plt.grid(True)
        plt.xticks(components)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'images', f'pca_analysis_{task}.png'))
        plt.close()
        print(f"\nSaved PCA analysis plot to outputs/images/pca_analysis_{task}.png")
        
        # Analysis conclusion
        best_result = max(results, key=lambda x: x['test_score'])
        print(f"\n{'='*50}")
        print(f"PCA ANALYSIS CONCLUSION")
        print(f"{'='*50}")
        print(f"Best performance: {best_result['n_components']} components with {scoring}={best_result['test_score']:.4f}")
        if best_result['n_components'] == 11:
            print("PCA did not improve model performance. All original features are valuable.")
            print("This suggests each chemical property contributes unique information to wine quality.")
        else:
            print(f"PCA improved performance by reducing to {best_result['n_components']} components.")
            print(f"This captures {best_result['variance_explained']:.1%} of variance while improving generalization.")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report of the analysis."""
        print("\n" + "="*50)
        print("GENERATING SUMMARY REPORT")
        print("="*50)
        
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'shape': self.df.shape,
                'features': list(self.X.columns),
                'quality_distribution': self.df['quality'].value_counts().to_dict()
            }
        }
        
        # Regression results
        if hasattr(self, 'regression_results'):
            reg_summary = {}
            for name, result in self.regression_results.items():
                if 'error' not in result:
                    reg_summary[name] = {
                        'r2': result['r2'],
                        'rmse': result['rmse'],
                        'mae': result['mae'],
                        'cv_r2_mean': result['cv_r2_mean'],
                        'cv_r2_std': result['cv_r2_std']
                    }
            report['regression_results'] = reg_summary
            
            # Best regression model
            best_reg = max(reg_summary.items(), key=lambda x: x[1]['r2'])
            report['best_regression_model'] = {
                'name': best_reg[0],
                'r2': best_reg[1]['r2'],
                'rmse': best_reg[1]['rmse']
            }
        
        # Classification results
        if hasattr(self, 'classification_results'):
            clf_summary = {}
            for name, result in self.classification_results.items():
                if 'error' not in result:
                    clf_summary[name] = {
                        'accuracy': result['accuracy'],
                        'f1': result['f1'],
                        'cv_f1_mean': result['cv_f1_mean'],
                        'cv_f1_std': result['cv_f1_std']
                    }
            report['classification_results'] = clf_summary
            
            # Best classification model
            best_clf = max(clf_summary.items(), key=lambda x: x[1]['f1'])
            report['best_classification_model'] = {
                'name': best_clf[0],
                'f1': best_clf[1]['f1'],
                'accuracy': best_clf[1]['accuracy']
            }
            
            # Class imbalance info
            if hasattr(self, 'use_balanced'):
                report['class_imbalance_handled'] = self.use_balanced
        
        # Save report as JSON
        report_path = os.path.join(self.output_dir, 'analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")
        
        # Save report as readable text
        text_report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(text_report_path, 'w') as f:
            f.write("WINE QUALITY ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: {report['analysis_date']}\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"- Shape: {report['dataset_info']['shape']}\n")
            f.write(f"- Features: {len(report['dataset_info']['features'])}\n\n")
            
            if 'best_regression_model' in report:
                f.write("Best Regression Model:\n")
                f.write(f"- Model: {report['best_regression_model']['name']}\n")
                f.write(f"- R² Score: {report['best_regression_model']['r2']:.4f}\n")
                f.write(f"- RMSE: {report['best_regression_model']['rmse']:.4f}\n\n")
            
            if 'best_classification_model' in report:
                f.write("Best Classification Model:\n")
                f.write(f"- Model: {report['best_classification_model']['name']}\n")
                f.write(f"- F1 Score: {report['best_classification_model']['f1']:.4f}\n")
                f.write(f"- Accuracy: {report['best_classification_model']['accuracy']:.4f}\n")
                if 'class_imbalance_handled' in report:
                    f.write(f"- Class Imbalance Handled: {'Yes' if report['class_imbalance_handled'] else 'No'}\n")
        
        print(f"Text report saved to: {text_report_path}")
        logging.info("Summary report generated successfully")
    
    def save_best_models(self):
        """Save the best models"""
        import joblib
        
        print("\n" + "="*50)
        print("SAVING BEST MODELS")
        print("="*50)
        
        # Save best regression model if available
        if hasattr(self, 'regression_results') and self.regression_results:
            valid_results = {k: v for k, v in self.regression_results.items() if 'error' not in v}
            if valid_results:
                best_reg_model = max(valid_results.items(), key=lambda x: x[1]['r2'])
                joblib.dump(best_reg_model[1]['model'], 
                           os.path.join(self.output_dir, 'models', 'best_regression_model.pkl'))
                print(f"Saved best regression model ({best_reg_model[0]}) with R² = {best_reg_model[1]['r2']:.4f}")
        
        # Save best classification model if available
        if hasattr(self, 'classification_results') and self.classification_results:
            valid_results = {k: v for k, v in self.classification_results.items() if 'error' not in v}
            if valid_results:
                best_clf_model = max(valid_results.items(), key=lambda x: x[1]['f1'])
                joblib.dump(best_clf_model[1]['model'], 
                           os.path.join(self.output_dir, 'models', 'best_classification_model.pkl'))
                print(f"Saved best classification model ({best_clf_model[0]}) with F1 = {best_clf_model[1]['f1']:.4f}")
        
        # Save scaler if available
        if hasattr(self, 'scaler') and self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(self.output_dir, 'models', 'scaler.pkl'))
            print("Saved feature scaler")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'data': {'path': 'data/winequality-red.csv', 'quality_threshold': QUALITY_THRESHOLD},
            'training': {'test_size': TEST_SIZE, 'random_seed': RANDOM_SEED, 'cv_folds': CV_FOLDS},
            'imbalance': {'handle_imbalance': True},
            'output': {'dir': 'outputs', 'generate_report': True}
        }

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Wine Quality Analysis CLI')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to wine quality dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--task', type=str, choices=['all', 'regression', 'classification'],
                       default='all', help='Which analysis to perform')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-balance', action='store_true',
                       help='Disable class balancing for classification')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (log to file only)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    data_path = args.data or config['data']['path']
    output_dir = args.output or config['output']['dir']
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = WineQualityAnalyzer(data_path=data_path, output_dir=output_dir)
    
    # Load and explore data
    analyzer.load_data()
    analyzer.explore_data()
    
    if args.task in ['all', 'regression']:
        # Regression analysis
        analyzer.prepare_data(task='regression')
        analyzer.train_regression_models()
        analyzer.feature_importance_analysis(task='regression')
        analyzer.pca_analysis(task='regression')
        
        if args.tune:
            analyzer.hyperparameter_tuning(model_name='Random Forest', task='regression')
    
    if args.task in ['all', 'classification']:
        # Classification analysis
        handle_imbalance = not args.no_balance
        analyzer.prepare_data(task='classification', handle_imbalance=handle_imbalance)
        analyzer.train_classification_models()
        analyzer.feature_importance_analysis(task='classification')
        analyzer.pca_analysis(task='classification')
        
        if args.tune:
            analyzer.hyperparameter_tuning(model_name='Random Forest', task='classification')
    
    # Save models
    analyzer.save_best_models()
    
    # Generate summary report
    if config['output']['generate_report']:
        analyzer.generate_summary_report()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"Results saved to: {output_dir}/")
    print(f"- Images: {output_dir}/images/")
    print(f"- Models: {output_dir}/models/")
    if config['output']['generate_report']:
        print(f"- Report: {output_dir}/analysis_report.txt")
        print(f"- JSON Report: {output_dir}/analysis_report.json")

if __name__ == "__main__":
    main()