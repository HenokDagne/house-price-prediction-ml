import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_dataset

# Set output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(__file__), '../outputs/plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def detect_target_column(df, manual_override=None):
    """
    Try to detect the target column (e.g., price) or use manual override.
    """
    if manual_override and manual_override in df.columns:
        return manual_override
    # Try common target names
    for col in df.columns:
        if col.lower() in ['price', 'saleprice', 'target', 'selling_price', 'sale_price']:
            return col
    # Fallback: last column
    return df.columns[-1]

def plot_target_distribution(df, target_col):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{target_col}_distribution.png"))
    plt.close()

def plot_correlation_heatmap(df, num_cols, target_col):
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
    plt.close()

def plot_scatter_vs_target(df, num_cols, target_col, max_plots=3):
    # Pick top correlated features (excluding target)
    corr = df[num_cols].corr()[target_col].abs().sort_values(ascending=False)
    features = [col for col in corr.index if col != target_col][:max_plots]
    for col in features:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[col], y=df[target_col])
        plt.title(f"{col} vs {target_col}")
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"scatter_{col}_vs_{target_col}.png"))
        plt.close()

def plot_boxplots(df, num_cols, target_col, max_plots=3):
    # Boxplots for outlier detection
    features = [col for col in num_cols if col != target_col][:max_plots]
    for col in features:
        plt.figure(figsize=(7, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"boxplot_{col}.png"))
        plt.close()

def plot_countplots(df, cat_cols, max_plots=2):
    # Count plots for categorical columns
    for col in cat_cols[:max_plots]:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        plt.title(f"Count Plot of {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"countplot_{col}.png"))
        plt.close()

def run_eda(csv_path, target_override=None):
    df = load_dataset(csv_path)
    if df is None:
        print("Failed to load dataset. Exiting EDA.")
        return
    # Basic inspection
    from src.data_loader import inspect_dataset
    num_cols, cat_cols = inspect_dataset(df)
    target_col = detect_target_column(df, target_override)
    print(f"\nTarget column selected: {target_col}\n")
    # EDA Visualizations
    plot_target_distribution(df, target_col)
    plot_correlation_heatmap(df, num_cols, target_col)
    plot_scatter_vs_target(df, num_cols, target_col)
    plot_boxplots(df, num_cols, target_col)
    plot_countplots(df, cat_cols)
    print(f"\nEDA plots saved to: {os.path.abspath(PLOT_DIR)}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA and save plots.")
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--target', type=str, default=None, help='Target column name (optional)')
    args = parser.parse_args()
    run_eda(args.csv, args.target)
