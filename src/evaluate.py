import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluation metrics

def evaluate_regression(y_true, y_pred):
    """
    Compute MAE, MSE, RMSE, and R2 for regression.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def print_evaluation_report(metrics_dict, model_name=None):
    """
    Print evaluation metrics in a clean format.
    """
    if model_name:
        print(f"\n===== {model_name} Evaluation =====")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")


def plot_actual_vs_pred(y_true, y_pred, model_name, output_dir):
    """
    Save scatter plot of actual vs predicted values.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted ({model_name})')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'actual_vs_pred_{model_name}.png'))
    plt.close()


def plot_feature_importance(model, feature_names, model_name, output_dir):
    """
    Plot and save feature importances if available.
    """
    importances = None
    # Try to get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    if importances is not None:
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances ({model_name})')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
        plt.close()
    else:
        print(f"Model {model_name} does not support feature importances.")
