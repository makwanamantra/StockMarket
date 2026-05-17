import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

import ta
from datetime import datetime

# ---------------------------
# Streamlit Page Config
# ---------------------------

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

st.title("📈 AI Stock Market Prediction System")

# ---------------------------
# User Inputs
# ---------------------------

stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "NVIDIA": "NVDA"
}

company = st.selectbox("Select Company", list(stocks.keys()))

investment = st.number_input(
    "Investment Amount ($)",
    min_value=100,
    value=1000
)

days_future = st.slider(
    "Prediction Days",
    1,
    30,
    7
)

ticker = stocks[company]

# ---------------------------
# Fetch Real-Time Data
# ---------------------------

data = yf.download(
    ticker,
    period="2y",
    interval="1d"
)

data.dropna(inplace=True)

# ---------------------------
# Technical Indicators
# ---------------------------

data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)

data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

data['MACD'] = ta.trend.macd(data['Close'])

data.dropna(inplace=True)

# ---------------------------
# Feature Engineering
# ---------------------------

features = [
    'Open',
    'High',
    'Low',
    'Volume',
    'SMA_10',
    'SMA_50',
    'RSI',
    'MACD'
]

X = data[features]
y = data['Close']

# Scale Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    shuffle=False
)

# ---------------------------
# Train Model
# ---------------------------

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

# ---------------------------
# Predictions
# ---------------------------

predictions = model.predict(X_test)

# ---------------------------
# Accuracy Calculation
# ---------------------------

# Direction Accuracy
actual_direction = np.where(
    y_test.diff() > 0,
    1,
    0
)

pred_direction = np.where(
    pd.Series(predictions).diff() > 0,
    1,
    0
)

accuracy = accuracy_score(
    actual_direction[1:],
    pred_direction[1:]
)

mae = mean_absolute_error(y_test, predictions)

# ---------------------------
# Future Prediction
# ---------------------------

last_features = X_scaled[-1].reshape(1, -1)

future_prices = []

future_price = model.predict(last_features)[0]

for i in range(days_future):
    future_prices.append(future_price)

future_avg = np.mean(future_prices)

# ---------------------------
# Risk Analysis
# ---------------------------

volatility = data['Close'].pct_change().std()

if volatility < 0.015:
    risk = "🟢 Low Risk"

elif volatility < 0.03:
    risk = "🟡 Moderate Risk"

else:
    risk = "🔴 High Risk"

# ---------------------------
# Profit/Loss Simulation
# ---------------------------

current_price = data['Close'].iloc[-1]

shares = investment / current_price

future_value = shares * future_avg

profit_loss = future_value - investment

# Reverse Loss Amount
loss_reverse = investment - abs(profit_loss)

# ---------------------------
# Dashboard Metrics
# ---------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model Accuracy",
    f"{accuracy*100:.2f}%"
)

col2.metric(
    "Mean Error",
    f"${mae:.2f}"
)

col3.metric(
    "Risk Level",
    risk
)

# ---------------------------
# Investment Prediction
# ---------------------------

st.subheader("💰 Investment Prediction")

st.write(f"Current Price: ${current_price:.2f}")

st.write(f"Predicted Future Price: ${future_avg:.2f}")

st.write(f"Estimated Future Value: ${future_value:.2f}")

if profit_loss >= 0:
    st.success(
        f"Estimated Profit: ${profit_loss:.2f}"
    )
else:
    st.error(
        f"Estimated Loss: ${abs(profit_loss):.2f}"
    )

st.write(
    f"Reverse Amount After Loss: ${loss_reverse:.2f}"
)

# ---------------------------
# Interactive Graph
# ---------------------------

fig = go.Figure()

# Actual Prices
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=y_test,
    mode='lines',
    name='Actual Price',
    line=dict(color='blue'),
    hovertemplate=
    '<b>Date:</b> %{x}<br>' +
    '<b>Actual:</b> $%{y:.2f}<extra></extra>'
))

# Predicted Prices
fig.add_trace(go.Scatter(
    x=y_test.index,
    y=predictions,
    mode='lines',
    name='Predicted Price',
    line=dict(color='red'),
    hovertemplate=
    '<b>Date:</b> %{x}<br>' +
    '<b>Prediction:</b> $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title=f"{company} Stock Prediction",
    xaxis_title="Date & Time",
    yaxis_title="Price",
    hovermode="x unified",
    template="plotly_dark",
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Daily Accuracy Table
# ---------------------------

results = pd.DataFrame({
    "Date": y_test.index,
    "Actual": y_test.values,
    "Predicted": predictions
})

results['Difference'] = (
    results['Actual'] -
    results['Predicted']
)

results['Accuracy %'] = (
    100 -
    abs(results['Difference']) /
    results['Actual'] * 100
)

st.subheader("📊 Daily Prediction Accuracy")

st.dataframe(results.tail(20))

# ---------------------------
# AI Recommendation
# ---------------------------

st.subheader("🤖 AI Recommendation")

if accuracy > 0.75:
    st.success(
        "Model confidence is HIGH. Small to moderate risk investment may be considered."
    )

elif accuracy > 0.60:
    st.warning(
        "Model confidence is MODERATE. Invest carefully."
    )

else:
    st.error(
        "Model confidence is LOW. High market uncertainty."
    )
