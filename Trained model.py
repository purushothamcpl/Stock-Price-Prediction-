# Stock Price Prediction 

import pandas as pd
import numpy as np

# Pandas is an inbuilt library in Python.  
# Pandas is a Python library for data manipulation and analysis.
# Pandas builds on NumPy to give you tabular data structures with labels, indexes, and more.
# Pandas is renamed as pd.
# NumPy (Numerical Python) is the fundamental library for numerical computing in Python.
# NumPy handles raw numerical arrays and fast maths.

df=pd.read_csv(r"C:\Users\lucky\OneDrive\Desktop\datascience\projects\AAPL.csv")
df

# Pandas supports reading/writing from many formats like CSV, Excel, SQL, JSON, etc.
# Loaded the dataset into a data analysis tool or programming environment.
# Using pandas, we have loaded the dataset into a data analysis tool.
# read_csv('filename.csv'): it is used to read a CSV file.
# read_excel('filename.xlsx'): it is used to read an Excel file.
# read_sql(query, connection): it is used to read an SQL database.
# It will display the columns present in the dataset.

df.shape
# .shape will display the no of rows and columns present in the dataset.

df.columns
# .columns present in the dataset will be displayed

df.head()
# .head will display the top 5 rows in the dataset.
# We can also specify the number of rows to be displayed in the brackets.

df.tail()
# .tail will display the bottom 5 rows in the dataset.
# We can also specify the number of rows to be displayed in the brackets.

df.info()
# It gives brief information about the dataset.
# It displays the number of columns present in the dataset and the non-null values in each column.
# Data types of each column and memory usage also.

df.dtypes
# .dtypes displays the data types of each column present in the dataset.

df.describe()
# Descriptive Statistics for Numerical Columns generated using .describe() Method
# count: sum of non-null entries in each column.
# mean: Average of the values in the column(sum of observations/no of observations).
# std: Standard deviation.
# min: Minimum value in the column.
# 25%: The 25th percentile (Q1), which means 25% of the data points are less than this value.
# 50%: Median value (50th percentile) where half the data points are below it.
# 75%: The 75th percentile (Q3) means 75% of the data points are below this value.
# max: Maximum value in the column.

df.isnull()
# .isnull will display any missing value present in the dataset.
# To find any null value or empty cell in the dataset.
# False means the cell is not empty.
# True means the cell is empty.

df.isnull().sum()
# We cannot see every cell as null or not.
# .sum will display the sum of null values in each column in the dataset.
# We can see any missing or null values present in the dataset clearly.

df.drop_duplicates()
# If any duplicate rows are present in the dataset will be removed.

df["Volume"].min()
# It will display the minimum value in the volume column.

df["Volume"].max()
# It will display the maximum value in the volume column.

df.corr(numeric_only=True)
# Correlation is a relation between two columns. 
# If correlation is near 1, then the relation between two columns is strong.
# If correlation is near 0, then the relation between two columns is weak.
# Strong correlation indicates that an increase in one column will increase the other column.
# Weak correlation indicates that an increase in one column will decrease the other column.

# Data Visualization
# Data visualization is the art and science of turning raw data into visual stories that are easy to understand, analyze, and act upon.
# Python libraries like Matplotlib, Seaborn are used to visualize raw data.
# There are many plotting techniques like bar charts, line charts, histograms, box plots, etc.
# Matplotlib is a Python library for creating static, animated, and interactive visualizations.
# plt.plot(): Draws the graph
# plt.title(): To add a title to the graph.
# plt.xlabel() / plt.ylabel(): it is used to add labels to the x and y axis.
# Labels show which column the axis represents.

import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib is the foundational plotting library in Python.
# A Python library used to make graphs and charts.
# Think of it like a drawing tool for data.
# You can make line plots, bar charts, scatter plots, pie charts, etc.
# Seaborn is built on top of Matplotlib and Pandas, designed for statistical data visualization.
# Another library that sits on top of Matplotlib.
# It makes charts look prettier and easier with less code.
# Great for statistics and data analysis.

plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Open"], label="Open")
plt.plot(df["Date"], df["High"], label="High")
plt.plot(df["Date"], df["Low"], label="Low")
plt.plot(df["Date"], df["Close"], label="Close")
plt.legend()
plt.title("OHLC Price Movement")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Plotting the Open, High, Low, and Close price of the Apple stock.
# For better understanding, we are plotting the price.
# In plotting, we can adjust the figure size.
# Plots the Open price against the Date column.
# Plots the High price over time.
# Plots the Low price over time.
# Plots the Close price over time.
# Displays a legend box that shows which line corresponds to Open, High, Low, and Close.
# Adds a title to the chart.
# Labels the x-axis as Date and the y-axis as Price.
# Renders the plot so you can see it.

# Converting Date column to datetime with dayfirst=True
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Setting as index
df.set_index('Date', inplace=True)

# Plotting the adjective close of the apple price
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Adj Close'], label='AAPL Closing Price', color='blue')
plt.title(f"AAPL Stock Price ({df.index.min().date()} to {df.index.max().date()})")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# The plotting is price vs date.
# We can see the up and down in the plotting below. 
# Price of the apple stock keeps increasing.
# The graph is 9 years of apple stock price.

plt.figure(figsize=(12,6))
plt.plot(df['Volume'], color='orange')
plt.title('Trading Volume Over Time')
plt.show()

# This is the plotting of the volume and date.
# Most highest trading was done in 2012 and 2014.
# Highest volume of trades sold in the year 2012.

df["Year"] = df.index.year

# Group by year and sum the Volume
yearly_volume = df.groupby("Year")["Volume"].sum()

# Plot bar chart
plt.figure(figsize=(10,5))
plt.bar(yearly_volume.index, yearly_volume.values, color="skyblue")
plt.title("Yearly Trading Volume")
plt.xlabel("Year")
plt.ylabel("Total Volume")
plt.xticks(yearly_volume.index, rotation=45)
plt.show()

# The bar plot illustrates the yearly trading volume.
# The volume goes on decreasing yearly till 2017.
# The bar plot shows that most of the stocks were sold in 2012, 2013, and 2014.df["Year"] = df.index.year

# Group by year and sum the Volume
yearly_volume = df.groupby("Year")["Volume"].sum()

# Plot bar chart
plt.figure(figsize=(10,5))
plt.bar(yearly_volume.index, yearly_volume.values, color="skyblue")
plt.title("Yearly Trading Volume")
plt.xlabel("Year")
plt.ylabel("Total Volume")
plt.xticks(yearly_volume.index, rotation=45)
plt.show()

# The bar plot illustrates the yearly trading volume.
# The volume goes on decreasing yearly till 2017.
# The bar plot shows that most of the stocks were sold in 2012, 2013, and 2014.

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# The correlation between open, high, low, and close is strong.
# The correlation between volume and others is neither strong nor weak.

df['ma10'] = df['Close'].rolling(10).mean()
df['ma50'] = df['Close'].rolling(50).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(df['ma10'], label='ma10')
plt.plot(df['ma50'], label='ma50')
plt.legend()
plt.show()

# ma10 is the moving average of 10 items.
# ma50 is the moving average of 50 items.
# Moving average is calculated by averaging 10 rows and adding at 10 row.
# We can see a line chart below of close column, ma10, and ma50.

monthly = df.resample("ME").mean()

plt.figure(figsize=(12,5))
plt.plot(monthly.index, monthly["Close"])
plt.title("Monthly Average Close Price")
plt.xlabel("Month")
plt.ylabel("Close Price")
plt.show()

# The plotting is the monthly average vs the closing price.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# Imports the pandas library and gives it the alias pd. Used for dataframes (loading and manipulating tabular data).
# Imports NumPy and aliases it as np. Used for arrays and numerical operations.
# Imports the plotting interface from matplotlib as plt. Used for creating graphs.
# Imports MinMaxScaler, which scales features to a range (by default 0 to 1). Needed for LSTM stability.
# Imports performance metrics:
# mean_absolute_error -> average absolute difference between actual and predicted.
# mean_squared_error -> average squared difference.
# Imports the Sequential model class from Keras - used to stack layers linearly.
# Imports:
# LSTM -> Long Short-Term Memory layer (for sequence modeling).
# Dense -> fully connected layer.
# Dropout -> regularization layer to reduce overfitting.
# Imports the warnings module to manage warning messages.
# Tells Python to ignore all warnings in the output (cleaner console, but you might miss useful warnings).

# Loading Dataset
df=pd.read_csv(r"C:\Users\lucky\OneDrive\Desktop\datascience\projects\AAPL.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

prices = df[['Close']]

# Uses pandas.read_csv to load the CSV at the given absolute path into a DataFrame named df.
# The r"" makes it a raw string, so backslashes in the path are not treated as escape characters.
# Converts the 'Date' column from string to datetime objects using the format day-month-year.
# Sorts the rows of df by the Date column in ascending order (oldest to newest).
# Sets the 'Date' column as the index of the DataFrame (time series style).
# inplace=True modifies df directly instead of returning a new dataframe.
# Creates a new DataFrame prices containing only the 'Close' price column.
# Note: [['Close']] returns a DataFrame, not a Series.

# Normalize Data
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# Creates an instance of MinMaxScaler, which will scale values to the range [0,1].
# fit_transform first learns the min/max of the data (fit) then applies scaling to all values (transform).
# scaled_prices becomes a NumPy array of normalized close prices.

# Create Sliding Window Dataset
def create_sequences(data, seq_len=50):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

seq_len = 50
X, y = create_sequences(scaled_prices, seq_len)

# Defines a function to convert a 1D time series into sequences for LSTM.
# data: scaled prices.
# seq_len: how many time steps to use as input (default 50).
# Initializes two empty Python lists:
# X -> will store input sequences.
# y -> will store corresponding target values.
# Loops over the data starting from index seq_len up to len(data)-1.
# Each iteration creates:
# A window of seq_len values as input.
# The next value is the output.
# Takes a slice from data:
# From i-seq_len (inclusive) to i (exclusive).
# That is the previous 50 time steps. Appends that sequence to X.
# Takes the value at position i → the next time step after the sequence and appends to y.
# Converts X and y lists into NumPy arrays and returns them (needed for Keras).
# Sets the sequence length to 50 time steps.
# X, y = create_sequences(scaled_prices, seq_len)
# Calls the function using the scaled prices to create:
# X: shape -> (num_samples, 50, 1)
# y: shape -> (num_samples, 1)

# Train-Test Split (last 30 days as test)
train_size = len(X) - 30
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Calculates how many samples will be used for training.
# Last 30 samples are reserved for testing, so we subtract 30.
# X_train, X_test = X[:train_size], X[train_size:]
# X[:train_size] -> from start to train_size-1 -> training inputs.
# X[train_size:] -> last 30 samples -> test inputs.
# Same logic for target values.

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(0.2),
    LSTM(80, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])


# Creates a Sequential model and defines the layers in a list.
# Inside the list:
# Adds an LSTM layer with 100 units (neurons).
# input_shape=(seq_len, 1) → each sample has:
# seq_len time steps (50),
# feature (Close price).
# return_sequences=True → this LSTM outputs the full sequence of hidden states (because another LSTM comes after it).
# Adds a Dropout layer that randomly sets 20% (0.2) of the units to 0 during training to avoid overfitting.
# Adds a second LSTM layer with 80 units.
# return_sequences=False → outputs only the last hidden state, not the full sequence, since the next layer is Dense (expects 2D input).
# Another Dropout layer with 20% dropout.
# Final Dense layer with 1 neuron → predicts a single value: the next day’s closing price (scaled).

model.compile(optimizer='adam', loss='mse')
model.summary()

# optimizer='adam' -> adaptive learning optimizer (very common).
# loss='mse' -> mean squared error loss function.
# Prints a summary of the model architecture: layers, output shapes, and parameter counts.

history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# Trains the model using training data:
# X_train, y_train -> training inputs and targets.
# epochs=20 -> the model sees the full training dataset 20 times.
# batch_size=16 -> model updates weights after every 16 samples.
# validation_split=0.1 -> uses 10% of training data for validation (checks performance during training).
# verbose=1 -> prints progress bar and metrics during training.
# Returns a History object with training & validation loss per epoch.

y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# Uses the trained model to predict outputs for the test inputs:
# outputs are still in scaled [0,1] form.
# Converts the scaled predicted values back to the original price scale using inverse_transform.
# Converts the true test targets from scaled values back to the original price scale too.
# (Assumes y_test is scaled; it is, because it came from scaled_prices.)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n Model Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Computes the Mean Absolute Error between actual and predicted prices.
# Computes Root Mean Squared Error:
# First computes MSE via mean_squared_error.
# Then square root using np.sqrt.
# Prints a blank line (\n) then the heading "Model Evaluation:".
# Prints MAE with 4 decimal places.
# Prints RMSE with 4 decimal places.

# Forecasting Next 30 Days price.
forecast_input = scaled_prices[-seq_len:].reshape(1, seq_len, 1)
future_forecast = []

# Takes the last 50 scaled prices: scaled_prices[-seq_len:].
# reshape(1, seq_len, 1) → shapes it like a single sample with:
# batch size = 1
# time steps = seq_len (50)
# features = 1
# This is the initial input sequence to forecast the next day.
# Initializes an empty list to store the 30 future predictions (still in scaled form).
    
for _ in range(30):
    pred = model.predict(forecast_input)[0]
    future_forecast.append(pred)

    forecast_input = np.append(forecast_input[:,1:,:], [[pred]], axis=1)

future_forecast = scaler.inverse_transform(future_forecast)

# Loops 30 times to generate 30 future days.
# Uses the current forecast_input sequence to predict the next scaled price.
# model. predict(...) returns an array of shape (1, 1) → we take [0] to get the first (and only) sample.
# Appends the predicted value to the future_forecast list.
# forecast_input[:,1:,:] → drops the first time step from the current sequence (sliding window).
# [[pred]] → adds the predicted value as the new last time step (wrapped to match dimensions).
# np.append(..., axis=1) → concatenates along the time-step axis.
# So now forecast_input again has shape (1, seq_len, 1), ready for the next iteration.
# Converts all the 30 future predictions from scaled values back to actual price values using inverse_transform.

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({"Date": future_dates, "LSTM_Forecast": future_forecast.reshape(-1)})

print("\n 30-Day LSTM Forecast:\n")
print(forecast_df)


# df.index[-1] → last date in your dataset.
# pd.Timedelta(days=1) → next day after the last historical date.
# pd.date_range(..., periods=30) → generates 30 consecutive dates starting from that day.
# forecast_df = pd.DataFrame({"Date": future_dates, "LSTM_Forecast": future_forecast.reshape(-1)})
# Creates a DataFrame with:
# 'Date' column from future_dates.
# 'LSTM_Forecast' column from future_forecast.
# reshape(-1) ensures it’s a 1D array suitable for a DataFrame column.
# Prints heading with blank lines before/after.
# Prints the forecast DataFrame to the console

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label="Historical", linewidth=2)
plt.plot(df.index[-30:], y_pred, label="Predicted (Test)", linestyle="--")
plt.plot(future_dates, future_forecast, label="Future Forecast (30 days)", linestyle=":")
plt.title("LSTM Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# Plotting Results of forecasted data along with the historical data.
# Creates a new figure of width 12 inches and height 6 inches.
# Plotting the full historical Close prices:
# x-axis -> df.index (dates)
# y-axis ->df['Close'] (prices)
# label="Historical" -> for the legend.
# linewidth=2 -> thicker line.
# Plotting predicted values for the test period:
# x-axis -> last 30 dates: df.index[-30:]
# y-axis -> y_pred (model predictions for test data)
# label=" Predicted (Test)" -> used in legend.
# linestyle="--" -> dashed line.
# Plots future forecast:
# x-axis -> future_dates
# y-axis -> future_forecast (predicted future prices)
# label="Future Forecast (30 days)."
# linestyle=":" -> dotted line.
# Sets the plot title.
# Labels x-axis as "Date".
# Labels y-axis as "Price".
# Displays the legend using the label arguments from each plot.
# Renders and displays the plot window.