import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Importing required libraries for deployment.
# Streamlit is used for creating a web app.
# Numpy is used for numerical operations.
# Matplotlib is used for plotting for visualization.
# Keras is used for loading the trained LSTM model. 

st.set_page_config(page_title="LSTM Stock Forecasting", layout="wide")
# Adding title to web and description.
st.title(" LSTM Stock Price Forecasting App")
st.write("Upload your historical stock data and generate future predictions using a trained LSTM model.")

# Load Model & Scaler (Cached)
@st.cache_resource
def load_lstm_components():
    model = load_model("models/lstm_model.keras", compile=False)
    scaler = joblib.load("forecastlstm.pkl")
    return model, scaler

try:
    model, scaler = load_lstm_components()
    st.success(" Model and Scaler Loaded Successfully")
except Exception as e:
    st.error(f" Error loading model/scaler: {e}")
    st.stop()

# Loading trained LSTM model 
# Loading the MinMaxScaler used during training.


uploaded_file = st.file_uploader(
    " Upload CSV or Excel File",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Please upload a file to continue.")
    st.stop()

file_name = uploaded_file.name.lower()

try:
    df = pd.read_csv(uploaded_file) if file_name.endswith(".csv") else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f" Error reading file: {e}")
    st.stop()

# Accepting the file given by the user.
# Accepting only CSV or Excel files.
# If the file is not in the above format, then it will give an error.


st.subheader(" Select Columns")

date_column = st.selectbox("Date Column:", df.columns)
close_column = st.selectbox("Close Price Column:", df.columns)
# We need to select the column from the dataset given for the forecast.

# Validate close column
if not np.issubdtype(df[close_column].dtype, np.number):
    st.error(" Close price column must be numeric.")
    st.stop()
# Validating the column given by the user.

df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
df.dropna(subset=[date_column], inplace=True)
df.sort_values(date_column, inplace=True)
df.set_index(date_column, inplace=True)

df.rename(columns={close_column: "Close"}, inplace=True)

st.success(" Data processed successfully!")
st.line_chart(df["Close"])

# Preprocessing the data given by the user.
# Changing the column name.


forecast_days = st.slider(" Days to Forecast:", 5, 180, 30)
seq_len = 50
# Arranging the no of days to forecast.

prices = df[['Close']]
scaled_prices = scaler.transform(prices)

if len(scaled_prices) < seq_len:
    st.error(" Not enough data. Need at least 50 rows.")
    st.stop()

last_sequence = scaled_prices[-seq_len:].reshape(1, seq_len, 1)
future_preds = []

with st.spinner(" Generating forecast..."):
    for _ in range(forecast_days):
        pred = model.predict(last_sequence, verbose=0)[0]
        future_preds.append(pred)
        last_sequence = np.append(last_sequence[:,1:,:], [[pred]], axis=1)

future_preds = np.array(future_preds).reshape(-1, 1)
future_preds = scaler.inverse_transform(future_preds)

# Taking the target column and transforming it.
# User-provided data should be at least 50 rows.
# It understands the data provided and forecasts the data given by the user.

future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted_Price": future_preds.flatten()
}).set_index("Date")


# Display Results
st.subheader(" Forecast Results")
st.dataframe(forecast_df.style.format({"Forecasted_Price": "{:.2f}"}))

# Plot Forecast
st.subheader(" Forecast Visualization")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['Close'], label="Historical Data")
ax.plot(forecast_df.index, forecast_df["Forecasted_Price"], "--", label="Forecast")

ax.set_title("LSTM Stock Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
# Displaying the result forecasted in plotting.
# Both provided data, and forecasted data are plotted.
# Labeled the title for the visualization.

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(forecast_df.index, forecast_df["Forecasted_Price"], "--", label="Forecast")

ax.set_title("LSTM Stock Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Download Button
csv_download = forecast_df.to_csv().encode('utf-8')
st.download_button(
    " Download Forecast CSV",
    csv_download,
    "stock_forecast.csv",
    "text/csv"
)
