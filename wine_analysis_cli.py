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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom utilities
from utils.visualization import save_fig, set_plotting_style, plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class WineQualityAnalyzer:
    """Main class for wine quality analysis"""
    
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
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading wine quality dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nBasic Statistics:")
        print(self.df.describe())
        return self.df
    
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
        
    def prepare_data(self, task='regression'):
        """Prepare data for modeling"""
        print(f"\nPreparing data for {task}...")
        
        # Separate features and target
        self.X = self.df.drop('quality', axis=1)
        
        if task == 'classification':
            # Convert to binary classification: low quality (3-5) vs high quality (6-8)
            self.y = (self.df['quality'] >= 6).astype(int)
            print(f"Binary classification - Low quality (0): {(self.y == 0).sum()}, High quality (1): {(self.y == 1).sum()}")
        else:
            self.y = self.df['quality']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_SEED, stratify=self.y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def train_regression_models(self):
        """Train multiple regression models"""
        print("\n" + "="*50)
        print("TRAINING REGRESSION MODELS")
        print("="*50)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=RANDOM_SEED),
            'Lasso Regression': Lasso(random_state=RANDOM_SEED),
            'ElasticNet': ElasticNet(random_state=RANDOM_SEED),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='r2')
            
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
        
        self.regression_results = results
        return results
    
    def train_classification_models(self):
        """Train multiple classification models"""
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
            'SVC': SVC(probability=True, random_state=RANDOM_SEED),
            'KNN': KNeighborsClassifier()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='f1')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_proba,
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  CV F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
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
    
    def save_best_models(self):
        """Save the best models"""
        import joblib
        
        print("\n" + "="*50)
        print("SAVING BEST MODELS")
        print("="*50)
        
        # Save best regression model
        best_reg_model = max(self.regression_results.items(), key=lambda x: x[1]['r2'])
        joblib.dump(best_reg_model[1]['model'], 
                   os.path.join(self.output_dir, 'models', 'best_regression_model.pkl'))
        print(f"Saved best regression model ({best_reg_model[0]}) with R² = {best_reg_model[1]['r2']:.4f}")
        
        # Save best classification model
        best_clf_model = max(self.classification_results.items(), key=lambda x: x[1]['f1'])
        joblib.dump(best_clf_model[1]['model'], 
                   os.path.join(self.output_dir, 'models', 'best_classification_model.pkl'))
        print(f"Saved best classification model ({best_clf_model[0]}) with F1 = {best_clf_model[1]['f1']:.4f}")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.output_dir, 'models', 'scaler.pkl'))
        print("Saved feature scaler")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Wine Quality Analysis CLI')
    parser.add_argument('--data', type=str, default='data/winequality-red.csv',
                       help='Path to wine quality dataset')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--task', type=str, choices=['all', 'regression', 'classification'],
                       default='all', help='Which analysis to perform')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = WineQualityAnalyzer(data_path=args.data, output_dir=args.output)
    
    # Load and explore data
    analyzer.load_data()
    analyzer.explore_data()
    
    if args.task in ['all', 'regression']:
        # Regression analysis
        analyzer.prepare_data(task='regression')
        analyzer.train_regression_models()
        analyzer.feature_importance_analysis(task='regression')
        
        if args.tune:
            analyzer.hyperparameter_tuning(model_name='Random Forest', task='regression')
    
    if args.task in ['all', 'classification']:
        # Classification analysis
        analyzer.prepare_data(task='classification')
        analyzer.train_classification_models()
        analyzer.feature_importance_analysis(task='classification')
        
        if args.tune:
            analyzer.hyperparameter_tuning(model_name='Random Forest', task='classification')
    
    # Save models
    analyzer.save_best_models()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"Results saved to: {args.output}/")
    print("- Images: outputs/images/")
    print("- Models: outputs/models/")

if __name__ == "__main__":
    main()