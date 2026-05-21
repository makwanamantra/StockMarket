import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import json
import os
import bcrypt
from datetime import datetime
import pytz
import uuid
import requests
from bs4 import BeautifulSoup
from yahooquery import Ticker
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
                st.rerun()
            else:
                st.sidebar.error("Invalid login")

    st.stop()

st.sidebar.success(f"Logged in as {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

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
# ============================================
# STOCK ANALYSIS
# ============================================
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
    full_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y_test, pred)
    accuracy = max(0, 100 - (mae / y_test.mean() * 100))

    future_price = float(model.predict(X_scaled[-1].reshape(1, -1))[0])

    fallback = float(close.iloc[-1]) if not close.empty else None
    live_price = get_live_price(ticker, fallback_price=fallback)

    volatility = float(close.pct_change().std())
    risk = "LOW" if volatility < 0.015 else "MODERATE" if volatility < 0.03 else "HIGH"

    return {
        "data": daily_data,
        "pred": pred,
        "full_pred": full_pred,
        "y_test": y_test,
        "future_price": future_price,
        "current_price": live_price,
        "accuracy": accuracy,
        "risk": risk
    }

def get_live_price(ticker: str, fallback_price: float = None) -> float:
    """
    Reliable live price fetcher:
    1. Try yfinance fast info
    2. Fallback to YahooQuery
    3. Fallback to scraping
    4. Fallback to last known price
    """
    try:
        # yfinance fast info (most reliable)
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        if price and price > 0:
            return float(price)
    except Exception:
        pass

    try:
        tq = Ticker(ticker)
        price = tq.price[ticker].get("regularMarketPrice")
        if price and price > 0:
            return float(price)
    except Exception:
        pass

    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        tag = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
        if tag and tag.text:
            return float(tag.text.replace(",", ""))
    except Exception:
        pass

    return float(fallback_price) if fallback_price else None


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
        "id": str(uuid.uuid4()),   # ✅ unique random ID
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
    st.rerun()  # refresh immediately so portfolio updates


# ============================================
# CHARTS
# ============================================

st.subheader("Prediction Graph")
fig = go.Figure()

# Timezones
ist = pytz.timezone("Asia/Kolkata")
us_eastern = pytz.timezone("US/Eastern")

# Ensure index has timezone
if stock_data["data"].index.tz is None:
    stock_data["data"].index = stock_data["data"].index.tz_localize("UTC")

# Add IST + US Eastern columns
stock_data["data"]["IST"] = stock_data["data"].index.tz_convert(ist)
stock_data["data"]["US_Eastern"] = stock_data["data"].index.tz_convert(us_eastern)

# Actual close prices
fig.add_trace(go.Scatter(
    x=stock_data["data"]["IST"],
    y=stock_data["data"]["Close"],
    name="Actual Close",
    line=dict(color="cyan"),
    hovertemplate="IST: %{x|%Y-%m-%d %H:%M:%S}<br>Price: $%{y:.2f}"
))

# Full predicted prices
fig.add_trace(go.Scatter(
    x=stock_data["data"]["IST"],
    y=stock_data["full_pred"],
    name="Predicted (Full)",
    line=dict(dash="dot", color="orange"),
    hovertemplate="IST: %{x|%Y-%m-%d %H:%M:%S}<br>Predicted: $%{y:.2f}"
))

# Difference line
diff = stock_data["data"]["Close"] - stock_data["full_pred"]
fig.add_trace(go.Scatter(
    x=stock_data["data"]["IST"],
    y=diff,
    name="Difference (Actual - Predicted)",
    line=dict(color="red", dash="dash"),
    hovertemplate="IST: %{x|%Y-%m-%d %H:%M:%S}<br>Diff: $%{y:.2f}"
))

# Current live price marker
now_utc = datetime.now(pytz.UTC)
now_ist = now_utc.astimezone(ist)
now_us = now_utc.astimezone(us_eastern)

# ✅ Safe live price fetch with sanity check
fallback = float(stock_data["data"]["Close"].iloc[-1]) if not stock_data["data"]["Close"].empty else None
live_price = get_live_price(stocks[selected_stock], fallback_price=fallback)

# Only plot if the live price is realistic
if live_price and 0 < live_price < 1000:
    fig.add_trace(go.Scatter(
        x=[now_ist],
        y=[live_price],
        mode="markers+text",
        text=[f"Live Price ({now_us.strftime('%Y-%m-%d %I:%M %p')} US)"],
        textposition="top center",
        name="Live Price",
        marker=dict(color="lime", size=12, line=dict(color="black", width=1)),
        hovertemplate=(
            "Live Price<br>"
            f"IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"US Eastern: {now_us.strftime('%Y-%m-%d %I:%M %p')}<br>"
            "Price: $%{y:.2f}"
        )
    ))

# Layout
fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis=dict(
        title="Date/Time (IST)",
        showspikes=True,
        spikemode="across",
        tickformat="%Y-%m-%d %H:%M:%S"
    ),
    yaxis_title="Price (USD)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# PORTFOLIO
# ============================================
st.subheader("My Portfolio")
portfolio = safe_load_json(PORTFOLIO_FILE)
user = st.session_state.username

if user in portfolio and portfolio[user]:
    portfolio_data = []
    total_profit = 0

    # Ensure every item has an ID
    updated = False
    for item in portfolio[user]:
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
            updated = True

    if updated:
        save_json(PORTFOLIO_FILE, portfolio)

    for item in portfolio[user]:
        try:
            intraday_data = yf.download(
                item["ticker"], period="1d", interval="1m",
                auto_adjust=True, progress=False
            )
            if intraday_data is not None and not intraday_data.empty and "Close" in intraday_data:
                latest_price = get_live_price(item["ticker"], fallback_price=item["buy_price"])

            else:
                latest_price = float(item["buy_price"])
        except Exception:
            latest_price = float(item["buy_price"])

        current_value = latest_price * item["shares"]
        profit_loss = current_value - item["investment"]
        total_profit += profit_loss

        portfolio_data.append({
            "ID": item["id"],
            "Stock": item["stock"],
            "Investment": round(item["investment"], 2),
            "Buy Price": round(item["buy_price"], 2),
            "Predicted Price": round(item["predicted_price"], 2),
            "Current Price": round(latest_price, 2),
            "Current Value": round(current_value, 2),
            "Profit/Loss": round(profit_loss, 2),
            "Date Bought": item["date"]
        })

    if portfolio_data:
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True)

        # Delete stock by ID
        delete_id = st.text_input("Enter Stock ID to Delete")
        if st.button("Delete Stock"):
            portfolio[user] = [i for i in portfolio[user] if i.get("id") != delete_id]
            save_json(PORTFOLIO_FILE, portfolio)
            st.success(f"Stock with ID {delete_id} deleted.")
            st.rerun()

        if total_profit > 0:
            st.success(f"Total Profit: ${total_profit:.2f}")
        elif total_profit < 0:
            st.error(f"Total Loss: ${abs(total_profit):.2f}")
        else:
            st.info("Portfolio is at break-even")

        st.download_button("Download Portfolio", df.to_csv(index=False), "portfolio.csv")
    else:
        st.info("Portfolio loaded, but no valid price data.")
else:
    st.info("No stocks purchased yet")
