import streamlit as st
import pandas as pd
import numpy as np
from model import load_and_train_model

st.set_page_config(page_title="APL Logistics Risk Dashboard", layout="wide")

@st.cache_resource
def get_data_and_model():
    return load_and_train_model()

df, model = get_data_and_model()

# Sidebar Filters
st.sidebar.header("Operations Control Panel")
shipping_mode = st.sidebar.selectbox("Filter Shipping Mode", ['All'] + list(df['Shipping Mode'].unique()))
market = st.sidebar.selectbox("Filter Market", ['All'] + list(df['Market'].unique()))
risk_threshold = st.sidebar.slider("High-Risk Probability Threshold", 0.0, 1.0, 0.7)

filtered_df = df.copy()
if shipping_mode != 'All':
    filtered_df = filtered_df[filtered_df['Shipping Mode'] == shipping_mode]
if market != 'All':
    filtered_df = filtered_df[filtered_df['Market'] == market]

# Main Title
st.title("📦 APL Logistics: Late Delivery Risk Prediction Dashboard")
st.markdown("Proactive supply chain risk intelligence platform designed to mitigate operational delays and SLA breaches.")

# Top Metric Cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Monitored Orders", f"{len(filtered_df):,}")
avg_delay_risk = filtered_df['Late_delivery_risk'].mean() * 100
col2.metric("Historical Delay Rate", f"{avg_delay_risk:.1f}%")
col3.metric("Active Shipping Lanes", filtered_df['Order Region'].nunique())

st.markdown("---")

# Module 1: Delay Risk Overview & Region Analysis
st.subheader("📊 Delay Risk Overview & Regional Breakdown")
if not filtered_df.empty:
    region_risk = filtered_df.groupby('Order Region')['Late_delivery_risk'].mean().reset_index()
    region_risk.columns = ['Order Region', 'Delay Rate']
    st.bar_chart(region_risk.set_index('Order Region'))
else:
    st.warning("No data available for the selected filters.")

# Module 2: Operations Action Panel (High-Risk Queue)
st.subheader("🚨 Operations Action Queue (High-Risk Orders)")
# Calculate prediction probabilities on filtered sample
sample_df = filtered_df.head(1000).copy()
if not sample_df.empty:
    features_list = [
        'Type', 'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer',
        'Customer Segment', 'Department Name', 'Market', 'Order Item Discount',
        'Order Item Product Price', 'Order Item Quantity', 'Sales', 'Order Item Total',
        'Order Profit Per Order', 'Order Region', 'Product Price', 'Shipping Mode',
        'Shipping_Pressure_Index', 'Order_Complexity_Score'
    ]
    sample_df['Shipping_Pressure_Index'] = sample_df['Days for shipment (scheduled)'] / (sample_df['Order Item Quantity'] + 1)
    sample_df['Order_Complexity_Score'] = sample_df['Order Item Quantity'] * sample_df['Product Price']
    
    probs = model.predict_proba(sample_df[features_list])[:, 1]
    sample_df['Late_Delivery_Probability'] = probs
    
    high_risk_queue = sample_df[sample_df['Late_Delivery_Probability'] >= risk_threshold]
    
    st.write(f"Showing **{len(high_risk_queue)}** orders exceeding risk threshold of **{risk_threshold}**:")
    st.dataframe(high_risk_queue[['Shipping Mode', 'Market', 'Late_Delivery_Probability', 'Order Region']])
else:
    st.info("Adjust filters to view order queue.")
