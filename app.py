%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

# Categorical features
categorical_features = ['product_category', 'customer_region', 'payment_method']

# Model columns (same order as training)
model_columns = ['price', 'discount_percent', 'quantity_sold', 'review_count',
                'product_category_Books', 'product_category_Electronics', 'product_category_Fashion',
                'product_category_Home & Kitchen', 'product_category_Sports',
                'customer_region_Europe', 'customer_region_Middle East', 'customer_region_North America',
                'payment_method_Credit Card', 'payment_method_Debit Card', 'payment_method_UPI',
                'payment_method_Wallet']

# Home route
@app.route('/')
def home():
    return "Amazon Sales Prediction API is running 🚀"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # One-hot encoding
        df = pd.get_dummies(df, columns=categorical_features)

        # Add missing columns
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct order
        df = df[model_columns]

        # Scale numerical data
        df_scaled = scaler.transform(df)

        # Prediction
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run app
if __name__ == '__main__':
    app.run(debug=True)
