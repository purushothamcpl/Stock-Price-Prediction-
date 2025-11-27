LSTM Stock Price Forecasting App

A Streamlit web application that forecasts future stock prices using a trained LSTM deep learning model.
Users can upload their own dataset (CSV/Excel), select date & close columns, and generate multi-day predictions with visualization.

Features
Upload CSV/XLSX stock datasets
Automated data preprocessing
Select custom Date & Close columns
Uses a pretrained LSTM model for forecasting
Choose forecast horizon (5–180 days)
Interactive line charts for:
Historical close prices
Forecasted future prices
Download forecast results as CSV
Fast model loading with @st.cache_resource

Tech Stack
Python
Streamlit
TensorFlow / Keras
NumPy
Pandas
Joblib
Matplotlib

Project Structure
├── models/
│   └── lstm_model.keras
├── forecastlstm.pkl
├── forecast.py
├── README.md
└── requirements.txt

 How to Run Locally
1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run forecast.py

Deploy on Streamlit Cloud
Push your code to GitHub
Go to https://streamlit.io/cloud
Click Deploy App
Select your repository & forecast.py
Add the following in Advanced Settings:
models/lstm_model.keras
forecastlstm.pkl


Deploy — your app will be live at a public URL 

Usage Instructions

Upload a CSV or Excel file containing stock historical data
Select:
Date column
Close price column

Choose number of future days to forecast
View:
Interactive charts
Forecast table

Download predictions as CSv

Requirements
Make sure you include a requirements.txt file such as:
streamlit
pandas
numpy
tensorflow
matplotlib
joblib
openpyxl

Contact
For questions or contributions:
Your Name: P Purushotham Reddy
Gmail: purushothamreddycpl5@gmail.com
