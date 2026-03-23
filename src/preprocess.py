import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load dataset
def load_dataset(csv_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)

# 2. Infer target column
def infer_target_column(df, manual_override=None):
    """Try to detect the target column, or use manual override if provided."""
    if manual_override and manual_override in df.columns:
        return manual_override
    for col in df.columns:
        if col.lower() in ['price', 'saleprice', 'target', 'selling_price', 'sale_price']:
            return col
    return df.columns[-1]  # fallback: last column

# 3. Split features and target
def split_features_target(df, target_col):
    """Split DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# 4. Detect numerical and categorical features
def detect_feature_types(X):
    """Return lists of numerical and categorical feature names."""
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return num_features, cat_features

# 5. Build preprocessing pipeline
def build_preprocessor(num_features, cat_features, scale_numeric=True):
    """
    Build a ColumnTransformer with separate pipelines for numeric and categorical features.
    - Numeric: impute missing values, optionally scale
    - Categorical: impute missing values, one-hot encode
    """
    num_pipeline = []
    num_pipeline.append(('imputer', SimpleImputer(strategy='mean')))
    if scale_numeric:
        num_pipeline.append(('scaler', StandardScaler()))
    num_pipeline = Pipeline(num_pipeline)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    return preprocessor

# 6. Prepare train/test data
def prepare_train_test_data(df, target_col=None, test_size=0.2, random_state=42, scale_numeric=True):
    """
    - Infers target column if not provided
    - Splits into train/test
    - Builds and fits preprocessor on train only (prevents data leakage)
    - Transforms train and test features
    - Returns: X_train_prep, X_test_prep, y_train, y_test, preprocessor
    """
    if target_col is None:
        target_col = infer_target_column(df)
    X, y = split_features_target(df, target_col)
    num_features, cat_features = detect_feature_types(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = build_preprocessor(num_features, cat_features, scale_numeric=scale_numeric)
    # Return DataFrames, not preprocessed arrays. The pipeline will handle preprocessing.
    return X_train, X_test, y_train, y_test, preprocessor

# Example usage (from train.py):
# from src.preprocess import load_dataset, prepare_train_test_data
# df = load_dataset('data/train.csv')
# X_train_prep, X_test_prep, y_train, y_test, preprocessor = prepare_train_test_data(df)

"""
Function Explanations:
- load_dataset: Loads a CSV file into a DataFrame.
- infer_target_column: Tries to guess the target column (e.g., price).
- split_features_target: Splits DataFrame into features and target.
- detect_feature_types: Finds numerical and categorical columns.
- build_preprocessor: Builds a scikit-learn pipeline for preprocessing.
- prepare_train_test_data: Full pipeline: splits, preprocesses, and returns ready-to-train/test data.

Common Preprocessing Pitfalls to Avoid:
- Fitting transformers (imputer, scaler, encoder) on the whole dataset before splitting (causes data leakage).
- Not handling unknown categories in OneHotEncoder (set handle_unknown='ignore').
- Forgetting to exclude the target column from preprocessing.
- Not using the same preprocessor for both train and test data.
"""
