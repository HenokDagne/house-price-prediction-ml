import joblib
import pandas as pd
import numpy as np
import os
import argparse
import json

def load_pipeline(model_path='models/house_price_pipeline.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline not found at {model_path}")
    return joblib.load(model_path)

def predict_price(input_data, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input must be a dict or pandas DataFrame.")
    try:
        prediction = pipeline.predict(df)
        return prediction[0] if len(prediction) == 1 else prediction
    except Exception as e:
        print(f"Prediction failed: {e}")
        print(f"Expected features: {getattr(pipeline, 'feature_names_in_', 'Unknown')}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict house price using the trained pipeline.")
    parser.add_argument('--input', type=str, help="JSON string or path to JSON file with input features.")
    args = parser.parse_args()

    if args.input:
        # Try to load from file or parse as JSON string
        if os.path.isfile(args.input):
            with open(args.input, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(args.input)
    else:
        # Example input
        input_data = {
            'area': 1200,
            'bedrooms': 3,
            'bathrooms': 2,
            'location': 'Downtown',
            'furnishing_status': 'Furnished',
        }
        print("No input provided. Using example input.")

    pipeline = load_pipeline()
    print("Predicting house price for input:")
    print(input_data)
    price = predict_price(input_data, pipeline)
    print(f"\nPredicted House Price: {price:.2f}" if price is not None else "Prediction failed.")