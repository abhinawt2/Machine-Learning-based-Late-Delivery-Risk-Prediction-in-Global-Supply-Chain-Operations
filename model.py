import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def load_and_train_model():
    # Load dataset
 file_id = "1IgsHyQiHbCQde8HRQad5B8pP9AZbpSPw"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
df = pd.read_csv(url, encoding="latin1")

    # Feature Engineering
    df['Shipping_Pressure_Index'] = df['Days for shipment (scheduled)'] / (df['Order Item Quantity'] + 1)
    df['Order_Complexity_Score'] = df['Order Item Quantity'] * df['Product Price']

    features = [
        'Type', 'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer',
        'Customer Segment', 'Department Name', 'Market', 'Order Item Discount',
        'Order Item Product Price', 'Order Item Quantity', 'Sales', 'Order Item Total',
        'Order Profit Per Order', 'Order Region', 'Product Price', 'Shipping Mode',
        'Shipping_Pressure_Index', 'Order_Complexity_Score'
    ]

    X = df[features]
    y = df['Late_delivery_risk']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1))
    ])

    model.fit(X, y)
    return df, model