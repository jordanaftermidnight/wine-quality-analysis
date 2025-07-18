"""
Visualization utilities for the Wine Quality Analysis project.
These functions help create consistent visualizations across the project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def set_plotting_style():
    """Set consistent plotting style for the project"""
    plt.style.use('default')
    sns.set_style('whitegrid')
    sns.set_palette("deep")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100

def save_fig(fig, filename, folder="images/", dpi=300, bbox_inches="tight"):
    """Save figure for GitHub documentation"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(f"{folder}{filename}", dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot a confusion matrix with annotations"""
    if class_names is None:
        class_names = ['Low Quality', 'High Quality']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], class_names)
    plt.yticks([0.5, 1.5], class_names)

    return plt.gcf()

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve with AUC score"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    return plt.gcf()

def plot_feature_importance(importance_df, title="Feature Importance"):
    """Plot feature importance with error bars"""
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'],
           xerr=importance_df['Std'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()

    return plt.gcf()