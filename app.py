import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(page_title="Time Series Forecast Pro", layout="wide")

def main():
    st.title("📈 Time Series Forecasting Application")
    
    # Data Upload
    uploaded_file = st.file_uploader("Upload CSV (date, value)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
    
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        # Ensure correct formatting
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Prophet Model
        st.subheader("🔮 Forecasting Configuration")
        periods = st.number_input("Forecast Periods", 1, 100, 10)
        freq = st.selectbox("Frequency", ["D", "W", "M"], index=0)

        if st.button("Generate Forecast"):
            with st.spinner("Training model..."):
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    interval_width=0.95
                )
                model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
                model.fit(df)

                future = model.make_future_dataframe(periods=periods, freq=freq)
                forecast = model.predict(future)

                st.subheader("📉 Forecast Results")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10), use_container_width=True)

                st.markdown("### 📈 Forecast Visualization")
                st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

                st.markdown("### 🔍 Forecast Components")
                st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

if __name__ == "__main__":
    main()
