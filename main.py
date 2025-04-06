import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time
from pymongo import MongoClient

# Download VADER lexicon
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Set your free News API key here
NEWS_API_KEY = 'jpdd94WWcSe1ezsepLfjvfuJXhWWoKBGAurAr7eG'  # Replace with your key

# Tickers to analyze
tickers = ["AAPL", "TSLA", "MSFT", "AMC", "KO", "GOOG", "NFLX", "AMZN", "META", "NVDA", "SPY", "BABA"]

# Fetch sentiment score from The News API
def get_sentiment_score(ticker):
    try:
        url = f'https://api.thenewsapi.com/v1/news/all?api_token={NEWS_API_KEY}&search={ticker}&language=en&limit=2'
        response = requests.get(url)
        articles = response.json().get('data', [])

        if not articles:
            return 0.5  # Neutral

        scores = []
        for article in articles:
            text = article.get('title', '') + '. ' + article.get('description', '')
            sentiment = sentiment_analyzer.polarity_scores(text)
            scores.append(sentiment['compound'])

        return np.mean(scores) * 0.5 + 0.5  # Scale to [0, 1]
    except Exception as e:
        print(f"Error fetching sentiment for {ticker}: {e}")
        return 0.5

# Extract features from Yahoo Finance
def get_features(ticker):
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "beta": info.get("beta", 1.0),
        "debt_to_equity": info.get("debtToEquity", 50),
        "pe_ratio": info.get("trailingPE", 20),
        "volume": info.get("volume", 1000000),
        "market_cap": info.get("marketCap", 10**9),
        "sentiment_score": get_sentiment_score(ticker)
    }

# 1. Collect data
data = [get_features(t) for t in tickers]
df = pd.DataFrame(data)

# 2. Create target scores from real data
# (In production, you'd use labeled historical risk assessments)
df["market_risk_score"] = df["beta"] * 5 + np.random.rand(len(df)) * 2

df["financial_risk_score"] = df["debt_to_equity"] / 10 + np.random.rand(len(df)) * 2

df["valuation_risk_score"] = df["pe_ratio"] / 5 + np.random.rand(len(df))

df["sentiment_risk_score"] = (1 - df["sentiment_score"]) * 10

df["liquidity_risk_score"] = 10 - np.log10(df["volume"] + 1)

# 3. Train regression models for each risk type
risk_types = {
    "market_risk_score": ["beta"],
    "financial_risk_score": ["debt_to_equity"],
    "valuation_risk_score": ["pe_ratio"],
    "sentiment_risk_score": ["sentiment_score"],
    "liquidity_risk_score": ["volume"]
}

models = {}
for risk, features in risk_types.items():
    model = LinearRegression()
    X = df[features].values.reshape(-1, len(features))
    y = df[risk]
    model.fit(X, y)
    models[risk] = model

# 4. Predict risk scores
for risk, features in risk_types.items():
    X = df[features].values.reshape(
        -1, len(features))
    df[f"predicted_{risk}"] = models[risk].predict(X)

# 5. Normalize to 0-10 scale
scaler = MinMaxScaler(feature_range=(0, 10))
for risk in risk_types:
    df[f"{risk}_scaled"] = scaler.fit_transform(df[[f"predicted_{risk}"]])

# 6. Prepare data for MongoDB
risk_columns = ["ticker"] + [f"{risk}_scaled" for risk in risk_types]
df_output = df[risk_columns].copy()
df_output.columns = ["ticker", "market_risk", "financial_risk", "valuation_risk", "sentiment_risk", "liquidity_risk"]

# Add the total risk score column
df_output['total_risk_score'] = df_output[["market_risk", "financial_risk", "valuation_risk", "sentiment_risk", "liquidity_risk"]].sum(axis=1)

# Prepare MongoDB data (convert to dict for insertion)
risk_data = df_output.to_dict(orient='records')

# 7. MongoDB Atlas connection
# Replace with your MongoDB Atlas URI
client = MongoClient('mongodb+srv://avibagchi04:QsJdKKFpmmb2oofZ@wehackcluster.hidg3jn.mongodb.net/')
db = client['risk_assessment_db']  # Database name
collection = db['risk_data']  # Collection name
collection.insert_many(risk_data)  # Insert data

print("\n📊 Risk data successfully inserted into MongoDB Atlas!")