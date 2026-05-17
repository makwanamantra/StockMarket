from streamlit_autorefresh import st_autorefresh
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import json
import os
import bcrypt
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="AI Stock Advisor Pro", layout="wide")

# ============================================
# FILES
# ============================================
USER_FILE = "users.json"
PORTFOLIO_FILE = "portfolio.json"

def safe_load_json(file):
    try:
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump({}, f)
        with open(file, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(file, data):
    try:
        with open(file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Error saving {file}: {e}")

# ============================================
# AUTH
# ============================================
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ============================================
# SESSION
# ============================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ============================================
# UI
# ============================================
st.title("AI Stock Advisor Pro")
st_autorefresh(interval=10 * 1000, key="stock_refresh")

# ============================================
# LOGIN
# ============================================
if not st.session_state.logged_in:
    mode = st.sidebar.radio("Choose", ["Login", "Signup"])
    users = safe_load_json(USER_FILE)

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if mode == "Signup":
        if st.sidebar.button("Create Account"):
            if username in users:
                st.sidebar.error("User already exists")
            elif len(password) < 4:
                st.sidebar.error("Password too short")
            else:
                users[username] = {"password": hash_password(password)}
                save_json(USER_FILE, users)
                st.sidebar.success("Account created")

    else:
        if st.sidebar.button("Login"):
            if username in users and verify_password(password, users[username]["password"]):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()   # ✅ fixed
            else:
                st.sidebar.error("Invalid login")

    st.stop()

st.sidebar.success(f"Logged in as {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()   # ✅ fixed

# ============================================
# STOCKS
# ============================================
stocks = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "NVIDIA": "NVDA",
    "Meta": "META"
}

# ============================================
# STOCK ANALYSIS
# ============================================
@st.cache_data(show_spinner=False)
def analyze_stock(ticker):
    try:
        daily_data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return None

    if daily_data is None or daily_data.empty:
        return None

    if isinstance(daily_data.columns, pd.MultiIndex):
        daily_data.columns = daily_data.columns.get_level_values(0)

    daily_data = daily_data.dropna()
    if "Close" not in daily_data.columns:
        return None

    close = daily_data["Close"].astype(float)

    # Technical indicators
    daily_data["SMA_10"] = ta.trend.sma_indicator(close, window=10)
    daily_data["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    daily_data["RSI"] = ta.momentum.rsi(close, window=14)
    daily_data["MACD"] = ta.trend.macd(close)
    daily_data = daily_data.dropna()

    if len(daily_data) < 50:
        return None

    features = ["Open", "High", "Low", "Volume", "SMA_10", "SMA_50", "RSI", "MACD"]
    X = daily_data[features]
    y = daily_data["Close"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    accuracy = max(0, 100 - (mae / y_test.mean() * 100))

    future_price = float(model.predict(X_scaled[-1].reshape(1, -1))[0])

    # Live intraday price
    try:
        intraday_data = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        if intraday_data is not None and not intraday_data.empty and "Close" in intraday_data:
            live_price = float(intraday_data["Close"].dropna().iloc[-1])
        else:
            live_price = float(close.iloc[-1])
    except Exception:
        live_price = float(close.iloc[-1])

    volatility = float(close.pct_change().std())
    risk = "LOW" if volatility < 0.015 else "MODERATE" if volatility < 0.03 else "HIGH"

    return {
        "data": daily_data,
        "pred": pred,
        "y_test": y_test,
        "future_price": future_price,
        "current_price": live_price,
        "accuracy": accuracy,
        "risk": risk
    }

# ============================================
# LOAD RESULTS
# ============================================
results = {}
with st.spinner("Analyzing market..."):
    for name, ticker in stocks.items():
        r = analyze_stock(ticker)
        if r:
            results[name] = r

# ============================================
# RANKING
# ============================================
if results:
    ranking = pd.DataFrame({
        "Stock": list(results.keys()),
        "Accuracy": [round(results[s]["accuracy"], 2) for s in results],
        "Risk": [results[s]["risk"] for s in results]
    }).sort_values("Accuracy", ascending=False)

    st.subheader("AI Stock Rankings")
    st.dataframe(ranking)

    best_stock = ranking.iloc[0]["Stock"]
    st.success(f"AI Recommended Stock: {best_stock}")
else:
    st.warning("No stock data available")

# ============================================
# INVESTMENT
# ============================================
st.subheader("Buy Stocks")
selected_stock = st.selectbox("Choose Stock", list(stocks.keys()))
investment = st.number_input("Investment Amount", min_value=100, value=1000)

if selected_stock not in results:
    st.warning("Stock data not available")
    st.stop()

stock_data = results[selected_stock]
current_price = stock_data["current_price"]
future_price = stock_data["future_price"]

shares = investment / current_price
future_value = shares * future_price
profit = future_value - investment

c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${current_price:.2f}")
c2.metric("Predicted Price", f"${future_price:.2f}")
c3.metric("Predicted Profit", f"${profit:.2f}")

if st.button("Buy Stock"):
    portfolio = safe_load_json(PORTFOLIO_FILE)
    user = st.session_state.username
    if user not in portfolio:
        portfolio[user] = []
    portfolio[user].append({
        "stock": selected_stock,
        "ticker": stocks[selected_stock],
        "investment": investment,
        "buy_price": current_price,
        "predicted_price": future_price,
        "shares": shares,
        "date": str(datetime.now())
    })
    save_json(PORTFOLIO_FILE, portfolio)
    st.success("Stock purchased!")

# ============================================
# CHART
# ============================================
st.subheader("Prediction Graph")
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data["data"].index, y=stock_data["data"]["Close"], name="Actual Close"))
fig.add_trace(go.Scatter(x=stock_data["y_test"].index, y=stock_data["pred"], name="Predicted (Test)"))
fig.add_trace(go.Scatter(x=[datetime.now()], y=[stock_data["current_price"]],
                         mode="markers+text", text=["Live Price"], name="Live Price",
                         marker=dict(color="red", size=10)))
