import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

tickers = ["AAPL", "TSLA", "MSFT", "AMC", "KO", "GOOG", "NFLX"]

def get_features(ticker):
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "beta": info.get("beta", 1.0),
        "debt_to_equity": info.get("debtToEquity", 50),
        "pe_ratio": info.get("trailingPE", 20),
        "volume": info.get("volume", 1000000),
        "market_cap": info.get("marketCap", 10**9),
        "simulated_sentiment": np.random.uniform(0, 1)  # placeholder
    }

# 1. Collect data
data = [get_features(t) for t in tickers]
df = pd.DataFrame(data)

# 2. Simulate target scores (for now), to train regression models
# Replace this with real labeled data in production
df["market_risk_score"] = df["beta"] * 5 + np.random.rand(len(df)) * 2
df["financial_risk_score"] = df["debt_to_equity"] / 10 + np.random.rand(len(df)) * 2
df["valuation_risk_score"] = df["pe_ratio"] / 5 + np.random.rand(len(df))
df["sentiment_risk_score"] = (1 - df["simulated_sentiment"]) * 10
df["liquidity_risk_score"] = 10 - np.log10(df["volume"] + 1)

# 3. Train separate regression models (mock training, since we simulate targets)
risk_types = {
    "market_risk_score": ["beta"],
    "financial_risk_score": ["debt_to_equity"],
    "valuation_risk_score": ["pe_ratio"],
    "sentiment_risk_score": ["simulated_sentiment"],
    "liquidity_risk_score": ["volume"]
}

models = {}
for risk, features in risk_types.items():
    model = LinearRegression()
    X = df[features].values.reshape(-1, len(features))
    y = df[risk]
    model.fit(X, y)
    models[risk] = model

# 4. Predict risk scores (real inference step)
for risk, features in risk_types.items():
    X = df[features].values.reshape(-1, len(features))
    df[f"predicted_{risk}"] = models[risk].predict(X)

# 5. Normalize predicted scores to range [0, 10]
scaler = MinMaxScaler(feature_range=(0, 10))
for risk in risk_types:
    df[f"{risk}_scaled"] = scaler.fit_transform(df[[f"predicted_{risk}"]])

# 6. Output results
risk_columns = ["ticker"] + [f"{risk}_scaled" for risk in risk_types]
df_output = df[risk_columns].copy()
df_output.columns = ["ticker", "market_risk", "financial_risk", "valuation_risk", "sentiment_risk", "liquidity_risk"]

print("\n📊 Multi-Dimensional Risk Scores:\n")
print(df_output.round(2).to_string(index=False))
