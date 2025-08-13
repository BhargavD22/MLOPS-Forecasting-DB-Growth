import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
import logging

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app_logs.log"), logging.StreamHandler()]
)
logging.info("Streamlit app started.")

st.set_page_config(page_title="DB Growth Forecast", layout="wide")

# -------------------------------
# Load CSV data
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("db_growth_data.csv", parse_dates=["Date"])
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'db_growth_data.csv' is in the same folder.")
        st.stop()

df = load_data()

# -------------------------------
# Current Usage Summary
# -------------------------------
st.title("📊 Current Database Usage Summary")

latest_dates = df.groupby('Server')['Date'].max().reset_index()
latest_data = pd.merge(df, latest_dates, on=['Server', 'Date'], how='inner')
server_sizes = latest_data.groupby('Server')['DB_Size_GB'].sum().reset_index()
total_size = server_sizes['DB_Size_GB'].sum()

for _, row in server_sizes.iterrows():
    st.write(f"Server: **{row['Server']}** — Total DB Size: **{row['DB_Size_GB']:.2f} GB**")

st.write("---")
st.write(f"### Overall Total DB Size Across All Servers: **{total_size:.2f} GB**")

# -------------------------------
# Forecasting Inputs
# -------------------------------
st.title("📈 Database Growth Forecasting")

months_to_forecast = st.number_input("Months to Forecast", min_value=1, max_value=48, value=6)
chart_type = st.selectbox("Chart Type", ["Line Chart", "Bar Chart"])

server_list = ["All Servers"] + sorted(df["Server"].unique())
selected_server = st.selectbox("Select Server", server_list)

if selected_server != "All Servers":
    db_list = ["All Databases"] + sorted(df[df["Server"] == selected_server]["Database"].unique())
else:
    db_list = ["All Databases"]
selected_database = st.selectbox("Select Database", db_list)

# -------------------------------
# Forecast Function
# -------------------------------
def forecast_arima(ts, periods):
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    future_dates = pd.date_range(ts.index[-1] + timedelta(days=1), periods=periods, freq="MS")
    return pd.DataFrame({"Date": future_dates, "Forecast_GB": forecast})

# -------------------------------
# Generate Forecast
# -------------------------------
if st.button("Generate Forecast"):
    if selected_server == "All Servers" and selected_database == "All Databases":
        ts = df.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
        forecast_df = forecast_arima(ts, months_to_forecast)
        title = "All Servers - All Databases"

    elif selected_server != "All Servers" and selected_database == "All Databases":
        filtered = df[df["Server"] == selected_server]
        ts = filtered.resample("MS", on="Date")["DB_Size_GB"].sum().asfreq("MS").fillna(method="ffill")
        forecast_df = forecast_arima(ts, months_to_forecast)
        title = f"{selected_server} - All Databases"

    else:
        filtered = df[(df["Server"] == selected_server) & (df["Database"] == selected_database)]
        ts = filtered.resample("MS", on="Date")["DB_Size_GB"].last().asfreq("MS").fillna(method="ffill")
        forecast_df = forecast_arima(ts, months_to_forecast)
        title = f"{selected_server} - {selected_database}"

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    if chart_type == "Line Chart":
        ax.plot(ts.index, ts.values, label="Historical", color="blue")
        ax.plot(forecast_df["Date"], forecast_df["Forecast_GB"], label="Forecast", color="red", linestyle="--")
    else:
        ax.bar(ts.index, ts.values, label="Historical", color="blue", alpha=0.6)
        ax.bar(forecast_df["Date"], forecast_df["Forecast_GB"], label="Forecast", color="red", alpha=0.8)

    ax.set_title(f"Forecast: {title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("DB Size (GB)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Table
    st.write("📅 **Forecasted Values Table**")
    st.dataframe(forecast_df)

