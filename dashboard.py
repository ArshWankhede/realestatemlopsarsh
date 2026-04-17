import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="MLOps Monitoring", layout="wide")

st.title("📊 Real Estate Model Monitoring Dashboard")
st.markdown("Live telemetry data from the FastAPI prediction endpoint.")

log_file = "prediction_logs.csv"

# Check if the log file exists (meaning the API has been used)
if os.path.exists(log_file):
    # Read the telemetry data
    df = pd.read_csv(log_file)
    
    # Convert Timestamp to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 1. Top Level Metrics
    st.subheader("Model Usage Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions Made", len(df))
    col2.metric("Average Predicted Price", f"${df['Predicted_Price'].mean() * 100000:,.0f}")
    col3.metric("Highest Predicted Price", f"${df['Predicted_Price'].max() * 100000:,.0f}")
    
    st.divider()

    # 2. Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Predicted Prices Over Time")
        # Line chart showing the predictions in order
        st.line_chart(df.set_index('Timestamp')['Predicted_Price'])
        
    with col_right:
        st.subheader("Distribution of Predictions")
        # Histogram to see if the model favors certain price ranges
        st.bar_chart(df['Predicted_Price'].value_counts(bins=10).sort_index())

    st.divider()
    
    # 3. Raw Data Table
    st.subheader("Raw Telemetry Logs")
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

else:
    st.warning("No data found! Go to your FastAPI docs (http://127.0.0.1:8000/docs) and make some predictions first.")