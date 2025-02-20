import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(page_title="Time Series Forecast Pro", layout="wide")

def main():
    st.title("📈 Time Series Forecasting Application")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV (date, value)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [col.strip().lower() for col in df.columns]  # Clean column names
            
            # Automatically detect date & value columns
            date_col = None
            value_col = None
            for col in df.columns:
                if "date" in col or "ds" in col:
                    date_col = col
                if "value" in col or "y" in col:
                    value_col = col
            
            if not date_col or not value_col:
                st.error("❌ Could not find appropriate date/value columns in CSV.")
                return
            
            # Convert date column safely
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.dropna(subset=[date_col, value_col], inplace=True)  # Remove bad rows
            
            # Rename for Prophet
            df = df.rename(columns={date_col: "ds", value_col: "y"})
            df["y"] = df["y"].astype(float)  # Ensure numeric values
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Prophet Model
            st.subheader("🔮 Forecasting Configuration")
            periods = st.number_input("Forecast Periods", 1, 100, 10)
            freq = st.selectbox("Frequency", ["D", "W", "M"], index=0)

            if st.button("Generate Forecast"):
                with st.spinner("Training model..."):
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                    model.fit(df)

                    future = model.make_future_dataframe(periods=periods, freq=freq)
                    forecast = model.predict(future)

                    st.subheader("📉 Forecast Results")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

                    st.markdown("### 📈 Forecast Visualization")
                    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

                    st.markdown("### 🔍 Forecast Components")
                    st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

if __name__ == "__main__":
    main()
