# AI Stock Advisor Pro
## Hosted Version
You can try the live application here: [AI Stock Advisor Pro](https://stockmarket-av06.onrender.com/)

AI Stock Advisor Pro is a Streamlit web application that analyzes stocks using technical indicators and machine learning. It provides live price tracking, predictions, portfolio management, and investment simulations.

# Features
Secure user authentication (signup/login with bcrypt)
Live stock price fetching (via yfinance, YahooQuery, and scraping fallback)
Technical indicators (SMA, RSI, MACD)
Machine learning predictions with XGBoost
Stock ranking based on accuracy and risk
Investment simulation with profit/loss calculation
Interactive prediction graphs with Plotly
Portfolio management (buy, delete, download CSV)
Auto-refresh for live market updates

# Tech Stack
Frontend/UI: Streamlit
Data: yfinance, YahooQuery, requests, BeautifulSoup
Machine Learning: XGBoost, scikit-learn
Visualization: Plotly
Authentication: bcrypt
Storage: JSON files (users.json, portfolio.json)

## File Structure

The repository is organized as follows:

```
.
├── app.py                # Main Streamlit application
├── users.json            # User authentication data
├── portfolio.json        # Portfolio data
├── requirements.txt      # Python dependencies
├── Procfile              # Render deployment configuration
└── README.md             # Project documentation
```
