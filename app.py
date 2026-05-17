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
# CONFIG
# ============================================
st.set_page_config(page_title="AI Stock Advisor Pro", layout="wide")
st.title("AI Stock Advisor Pro")

st_autorefresh(interval=10 * 1000, key="refresh")

# ============================================
# FILES
# ============================================
USER_FILE = "users.json"
PORTFOLIO_FILE = "portfolio.json"

def safe_load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def safe_save(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

# create files
if not os.path.exists(USER_FILE):
    safe_save(USER_FILE, {})
if not os.path.exists(PORTFOLIO_FILE):
    safe_save(PORTFOLIO_FILE, {})

# ============================================
# AUTH
# ============================================
def hash_password(p):
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()

def verify_password(p, h):
    return bcrypt.checkpw(p.encode(), h.encode())

# ============================================
# SESSION
# ============================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ============================================
# LOGIN / SIGNUP
# ============================================
if not st.session_state.logged_in:

    mode = st.sidebar.radio("Choose", ["Login", "Signup"])
    users = safe_load(USER_FILE)

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if mode == "Signup":
        if st.sidebar.button("Create Account"):
            if username in users:
                st.error("User exists")
            elif len(password) < 4:
                st.error("Password too short")
            else:
                users[username] = {"password": hash_password(password)}
                safe_save(USER_FILE, users)
                st.success("Account created")

    else:
        if st.sidebar.button("Login"):
            if username in users and verify_password(password, users[username]["password"]):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid login")

    st.stop()

st.sidebar.success(f"Logged in: {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ============================================
# STOCK LIST
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
# STOCK ANALYSIS SAFE
# ============================================
@st.cache_data(ttl=300)
def analyze_stock(ticker):

    data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)

    if data is None or data.empty:
        return None

    if "Close" not in data.columns:
        return None

    data = data.dropna()
    close = data["Close"].astype(float)

    if len(data) < 60:
        return None

    data["SMA_10"] = ta.trend.sma_indicator(close, 10)
    data["SMA_50"] = ta.trend.sma_indicator(close, 50)
    data["RSI"] = ta.momentum.rsi(close, 14)
    data["MACD"] = ta.trend.macd(close)

    data = data.dropna()

    if len(data) < 60:
        return None

    features = ["Open","High","Low","Volume","SMA_10","SMA_50","RSI","MACD"]

    X = data[features]
    y = data["Close"]

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    accuracy = max(0, 100 - (mae / y_test.mean() * 100))

    future_price = float(model.predict(Xs[-1].reshape(1, -1))[0])
    current_price = float(close.iloc[-1])

    volatility = float(close.pct_change().std())
    risk = "LOW" if volatility < 0.015 else "MODERATE" if volatility < 0.03 else "HIGH"

    return {
        "data": data,
        "pred": pred,
        "y_test": y_test,
        "future_price": future_price,
        "current_price": current_price,
        "accuracy": accuracy,
        "risk": risk
    }

# ============================================
# RUN ANALYSIS
# ============================================
results = {}

with st.spinner("Analyzing..."):
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

    st.subheader("AI Ranking")
    st.dataframe(ranking)
    st.success(f"Best: {ranking.iloc[0]['Stock']}")
else:
    st.warning("No data")

# ============================================
# BUY STOCK
# ============================================
st.subheader("Buy Stock")

selected = st.selectbox("Stock", list(stocks.keys()))
amount = st.number_input("Investment", 100, 100000, 1000)

if selected not in results:
    st.stop()

stock_data = results[selected]

price = stock_data["current_price"]
future = stock_data["future_price"]

shares = amount / price

c1, c2, c3 = st.columns(3)
c1.metric("Current", price)
c2.metric("Future", future)
c3.metric("Profit", (future - price) * shares)

if st.button("Buy Stock"):

    portfolio = safe_load(PORTFOLIO_FILE)
    user = st.session_state.username

    if user not in portfolio:
        portfolio[user] = []

    portfolio[user].append({
        "stock": selected,
        "ticker": stocks[selected],
        "investment": amount,
        "buy_price": price,
        "predicted_price": future,
        "shares": shares,
        "date": datetime.now().isoformat()
    })

    safe_save(PORTFOLIO_FILE, portfolio)

    st.success("Stock added to portfolio")
    st.rerun()   # 🔥 IMPORTANT FIX

# ============================================
# PORTFOLIO (FIXED)
# ============================================
st.subheader("Portfolio")

portfolio = safe_load(PORTFOLIO_FILE)
user = st.session_state.username

if user in portfolio and portfolio[user]:

    rows = []
    total = 0

    for item in portfolio[user]:

        try:
            data = yf.download(item["ticker"], period="5d", interval="1d")

            if data is None or data.empty or "Close" not in data:
                continue

            close = data["Close"].dropna()

            if len(close) == 0:
                continue

            latest = float(close.iloc[-1])

            value = latest * item["shares"]
            pnl = value - item["investment"]

            total += pnl

            rows.append({
                "Stock": item["stock"],
                "Invested": item["investment"],
                "Current": latest,
                "Value": value,
                "PnL": pnl
            })

        except:
            continue

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df)

        if total >= 0:
            st.success(f"Total Profit: {total:.2f}")
        else:
            st.error(f"Total Loss: {total:.2f}")
    else:
        st.info("No valid portfolio data yet")

else:
    st.info("No stocks purchased yet")
