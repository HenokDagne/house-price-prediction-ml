import joblib
import pandas as pd
import numpy as np
import os

def load_pipeline(model_path='models/house_price_pipeline.pkl'):
    """Load the trained pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline not found at {model_path}")
    return joblib.load(model_path)

def predict_price(input_data, pipeline=None):
    """
    Predict house price from input data (dict or DataFrame) using the trained pipeline.
    Handles missing optional fields gracefully.
    """
    if pipeline is None:
        pipeline = load_pipeline()
    # Accept dict or DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input must be a dict or pandas DataFrame.")
    # Predict
    try:
        prediction = pipeline.predict(df)
        return prediction[0] if len(prediction) == 1 else prediction
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

if __name__ == "__main__":
    # Example input (customize fields as per your dataset)
    example_input = {
        'area': 1200,
        'bedrooms': 3,
        'bathrooms': 2,
        'location': 'Downtown',
        'furnishing_status': 'Furnished',
        # Add other features as needed
    }
    print("Loading model pipeline...")
    pipeline = load_pipeline()
    print("Predicting house price for example input:")
    print(example_input)
    price = predict_price(example_input, pipeline)
    print(f"\nPredicted House Price: {price:.2f}")
