import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.preprocess import load_dataset, prepare_train_test_data
from src.evaluate import evaluate_regression, print_evaluation_report, plot_actual_vs_pred, plot_feature_importance

# Optional advanced models
try:
    from xgboost import XGBRegressor
    xgb_installed = True
except ImportError:
    xgb_installed = False

try:
    from catboost import CatBoostRegressor
    catboost_installed = True
except ImportError:
    catboost_installed = False

# Paths
MODEL_DIR = 'models'
OUTPUTS_DIR = 'outputs'
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
COMPARISON_CSV = os.path.join(OUTPUTS_DIR, 'model_comparison.csv')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'house_price_pipeline.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_models(random_state=42):
    models = [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(random_state=random_state)),
        ('Gradient Boosting', GradientBoostingRegressor(random_state=random_state))
    ]
    if xgb_installed:
        models.append(('XGBoost', XGBRegressor(random_state=random_state, verbosity=0)))
    if catboost_installed:
        models.append(('CatBoost', CatBoostRegressor(random_state=random_state, verbose=0)))
    return models

def main():
    # Load and preprocess data
    df = load_dataset('data/train.csv')
    # Prepare train/test splits and preprocessor
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df)
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = []
    models = get_models()
    results = []
    best_score = np.inf
    best_model = None
    best_model_name = None
    best_pipeline = None
    for name, model in models:
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        # Fit on training data
        pipeline.fit(X_train, y_train)
        # Predict on train and test
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        # Evaluate on train (for overfitting check)
        train_metrics = evaluate_regression(y_train, y_train_pred)
        print_evaluation_report(train_metrics, model_name=f"{name} (Train)")
        # Evaluate on test
        test_metrics = evaluate_regression(y_test, y_test_pred)
        print_evaluation_report(test_metrics, model_name=f"{name} (Test)")
        # Save results
        results.append({
            'Model': name,
            **test_metrics
        })
        # Save plots
        plot_actual_vs_pred(y_test, y_test_pred, name, PLOTS_DIR)
        # Save feature importance if possible
        if feature_names:
            try:
                plot_feature_importance(model, feature_names, name, PLOTS_DIR)
            except Exception:
                pass
        # Select best model (prefer lowest RMSE, then highest R2)
        if (test_metrics['RMSE'] < best_score) or (best_score == np.inf):
            best_score = test_metrics['RMSE']
            best_model = model
            best_model_name = name
            best_pipeline = pipeline
    # Save comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"\nModel comparison table saved to {COMPARISON_CSV}")
    # Save best model pipeline
    if best_pipeline is not None:
        joblib.dump(best_pipeline, BEST_MODEL_PATH)
        print(f"\nBest model pipeline ({best_model_name}) saved to {BEST_MODEL_PATH}")
        print(f"\nBest model selected: {best_model_name} (lowest RMSE)")
        print(comparison_df.sort_values('RMSE').head(1))
    else:
        print("No model was successfully trained.")

if __name__ == "__main__":
    main()
