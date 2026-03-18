%%writefile app.py
from flask import Flask
import joblib
import pandas as pd

# Create an instance of the Flask application
app = Flask(__name__)

# Load the trained model and scaler
# These files are expected to be in the same directory as app.py
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

# Define the categorical features for one-hot encoding
categorical_features = ['product_category', 'customer_region', 'payment_method']

# Define the list of feature columns expected by the model
# These columns must be in the same order as during training
model_columns = ['price', 'discount_percent', 'quantity_sold', 'review_count',
                'product_category_Books', 'product_category_Electronics', 'product_category_Fashion',
                'product_category_Home & Kitchen', 'product_category_Sports',
                'customer_region_Europe', 'customer_region_Middle East', 'customer_region_North America',
                'payment_method_Credit Card', 'payment_method_Debit Card', 'payment_method_UPI',
                'payment_method_Wallet']

# Define a simple root route
@app.route('/')
def home():
    return "Welcome to the Amazon Sales Prediction API! Use the /predict endpoint to get predictions."

print("app.py created successfully with basic Flask structure and model_columns.")
