
import joblib
import pandas as pd
import os

# Define the path to the saved model (relative to input.py)
# Assuming input.py is in 'user/' and models/ is in the parent directory
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'house_price_pipeline.pkl')

# Load the pre-trained model
try:
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please ensure train.py was run successfully.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

def get_valid_input(prompt, valid_options=None, input_type=str, min_val=None, max_val=None):
    while True:
        user_input = input(prompt).strip()
        if valid_options:
            if user_input.upper() in [str(opt).upper() for opt in valid_options]:
                if input_type == int:
                    return int(user_input)
                else:
                    # Return original casing if it matches an option, or the user input itself
                    for opt in valid_options:
                        if user_input.upper() == str(opt).upper():
                            return opt
                    return user_input # Fallback
            else:
                print(f"Invalid input. Please choose from {', '.join(map(str, valid_options))}.")
        else:
            try:
                converted_input = input_type(user_input)
                if min_val is not None and converted_input < min_val:
                    print(f"Input must be at least {min_val}.")
                elif max_val is not None and converted_input > max_val:
                    print(f"Input must be at most {max_val}.")
                else:
                    return converted_input
            except ValueError:
                # Custom error message for invalid numeric input
                print("Invalid input. Please enter a numeric value only (example: 38.7368).")

def predict_house_price():
    print("\nPlease enter the house features for prediction:")

    # BHK_NO.
    bhk_options = [1, 2, 3, 4, 5] # '5+' will be handled as 5
    bhk_no = get_valid_input("Number of BHKs (1, 2, 3, 4, 5+): ", valid_options=[str(x) for x in bhk_options] + ['5+'], input_type=str)
    if bhk_no == '5+':
        bhk_no = 5 # Map 5+ to 5 or handle as desired
    else:
        bhk_no = int(bhk_no)

    # SQUARE_FT
    square_ft = get_valid_input("Square Footage (e.g., 1200): ", input_type=float, min_val=0.0)

    # POSTED_BY
    posted_by_options = ['Owner', 'Dealer', 'Builder']
    posted_by = get_valid_input(f"Posted by ({', '.join(posted_by_options)}): ", valid_options=posted_by_options)

    # UNDER_CONSTRUCTION
    under_construction = get_valid_input("Under Construction (0 for No, 1 for Yes): ", valid_options=[0, 1], input_type=int)

    # RERA
    rera = get_valid_input("RERA (0 for No, 1 for Yes): ", valid_options=[0, 1], input_type=int)

    # BHK_OR_RK
    bhk_or_rk_options = ['BHK', 'RK']
    bhk_or_rk = get_valid_input(f"BHK or RK ({', '.join(bhk_or_rk_options)}): ", valid_options=bhk_or_rk_options)

    # READY_TO_MOVE
    ready_to_move = get_valid_input("Ready to Move (0 for No, 1 for Yes): ", valid_options=[0, 1], input_type=int)

    # RESALE
    resale = get_valid_input("Resale (0 for No, 1 for Yes): ", valid_options=[0, 1], input_type=int)

    # ADDRESS and Coordinates - Manual Input
    address = input("Enter Address (e.g., Bangalore, 123 Main St): ").strip().upper()
    longitude = get_valid_input("Enter Longitude (between -180 and 180): ", input_type=float, min_val=-180.0, max_val=180.0)
    latitude = get_valid_input("Enter Latitude (between -90 and 90): ", input_type=float, min_val=-90.0, max_val=90.0)

    # Create a DataFrame for prediction, mimicking the test_df structure
    input_data = pd.DataFrame([{
        'POSTED_BY': posted_by,
        'UNDER_CONSTRUCTION': under_construction,
        'RERA': rera,
        'BHK_NO.': bhk_no,
        'BHK_OR_RK': bhk_or_rk,
        'SQUARE_FT': square_ft,
        'READY_TO_MOVE': ready_to_move,
        'RESALE': resale,
        'ADDRESS': address,
        'LONGITUDE': longitude,
        'LATITUDE': latitude
    }])

    # Make prediction
    predicted_price_lakhs = loaded_model.predict(input_data)

    # Convert Lakhs to USD (assuming 1 Lakh INR = 1200 USD as an approximation)
    # This conversion rate might need to be updated for current market conditions
    lakh_to_usd_conversion_rate = 1200.0 # Approximate conversion
    predicted_price_usd = predicted_price_lakhs[0] * lakh_to_usd_conversion_rate

    print(f"\nPredicted House Price: ${predicted_price_usd:,.2f} USD")

if __name__ == "__main__":
    predict_house_price()
