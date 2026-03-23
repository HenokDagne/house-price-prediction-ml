import os
import pandas as pd


def load_dataset(csv_path):
    """
    Loads a CSV dataset using pandas and returns the DataFrame.
    Handles file not found errors gracefully.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded dataset from {csv_path}\n")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def inspect_dataset(df):
    """
    Prints a clean summary report of the dataset.
    """
    if df is None:
        print("No dataset to inspect.")
        return
    print("===== Dataset Shape =====")
    print(df.shape)
    print("\n===== Column Names =====")
    print(list(df.columns))
    print("\n===== Data Types =====")
    print(df.dtypes)
    print("\n===== Summary Statistics =====")
    print(df.describe(include='all').transpose())
    print("\n===== First 5 Rows =====")
    print(df.head())
    print("\n===== Missing Values (per column) =====")
    print(df.isnull().sum())
    print("\n===== Duplicate Rows =====")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    # Separate columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("\n===== Numerical Columns =====")
    print(num_cols)
    print("\n===== Categorical Columns =====")
    print(cat_cols)
    print("\n===== Dataset Inspection Complete =====\n")
    return num_cols, cat_cols

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and inspect a CSV dataset.")
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()
    df = load_dataset(args.csv)
    inspect_dataset(df)
