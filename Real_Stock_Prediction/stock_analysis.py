#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from yahooquery import Screener

# Function to calculate Simple Moving Average (SMA)
def calculate_sma(data, period=50):
    return data["Close"].rolling(window=period).mean()

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate MACD
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data["Close"].ewm(span=short_period, adjust=False).mean()
    long_ema = data["Close"].ewm(span=long_period, adjust=False).mean()
    
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    return macd, signal

# Fetch Stock Data and Generate Indicators
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")

    if data.empty:
        return None  # Handle cases where stock data is unavailable

    # Calculate Technical Indicators
    data["SMA_50"] = calculate_sma(data, period=50)
    data["SMA_200"] = calculate_sma(data, period=200)
    data["RSI"] = calculate_rsi(data, period=14)
    data["MACD"], data["MACD_signal"] = calculate_macd(data)

    data.dropna(inplace=True)
    return data

# Load Model
def load_model():
    try:
        return joblib.load("stock_model.pkl")
    except FileNotFoundError:
        return None

# Function to fetch top gainers and losers from Yahoo Finance
def get_top_movers(category="day_gainers"):
    try:
        screener = Screener()
        data = screener.get_screeners([category], count=5)  # Fetch top 5 stocks
        stocks = data.get(category, {}).get("quotes", [])

        return [
            {"Symbol": stock["symbol"], "Name": stock["shortName"], "Change %": stock["regularMarketChangePercent"]}
            for stock in stocks
        ]
    except:
        return None  # Handle errors gracefully

# Streamlit UI
st.title("📈 Stock Investment Predictor with Market Movers")
st.write("Predict if a stock is a *good investment* based on historical trends and check market movers.")

# Buttons for showing gainers and losers
gainers_visible = st.button("Show Top Gainers")
losers_visible = st.button("Show Top Losers")

if gainers_visible:
    st.subheader("🔼 Top 5 Gainers")
    top_gainers = get_top_movers("day_gainers")
    if top_gainers:
        st.write(pd.DataFrame(top_gainers))
    else:
        st.error("⚠️ Unable to fetch top gainers.")

if losers_visible:
    st.subheader("🔻 Top 5 Losers")
    top_losers = get_top_movers("day_losers")
    if top_losers:
        st.write(pd.DataFrame(top_losers))
    else:
        st.error("⚠️ Unable to fetch top losers.")

# Stock Prediction Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):").upper()

if st.button("Predict Investment"):
    with st.spinner("Predicting..."):
        model = load_model()
        
        if model is None:
            st.error("❌ No trained model found. Please train the model first.")
        else:
            data = get_stock_data(ticker)
            
            if data is None or data.empty:
                st.error("⚠️ No stock data found. Please check the ticker symbol.")
            else:
                latest_data = data.iloc[-1][["Close", "SMA_50", "SMA_200", "RSI", "MACD"]].values.reshape(1, -1)
                prediction = model.predict(latest_data)[0]

                latest_price = round(data.iloc[-1]["Close"], 2)
                sma_50 = round(data.iloc[-1]["SMA_50"], 2)
                sma_200 = round(data.iloc[-1]["SMA_200"], 2)
                rsi = round(data.iloc[-1]["RSI"], 2)
                macd = round(data.iloc[-1]["MACD"], 2)

                rsi_status = "🔴 Overbought (May drop soon)" if rsi > 70 else "🟢 Oversold (May rise soon)" if rsi < 30 else "🟡 Neutral"
                macd_trend = "🔺 Bullish" if macd > 0 else "🔻 Bearish"
                investment_status = "✅ *GOOD Investment!" if prediction == 1 else "❌ **BAD Investment!*"

                report = f"""
                ## 📊 Stock Prediction Report for {ticker}

                - **Latest Price**: ${latest_price}
                - **SMA 50**: {sma_50}  
                - **SMA 200**: {sma_200}  
                - **RSI**: {rsi} → {rsi_status}  
                - **MACD**: {macd} → {macd_trend}  
                
                ### 🏆 *Investment Decision:* {investment_status}  
                """

                st.markdown(report, unsafe_allow_html=True)

                # Display Graphs
                st.subheader("📈 Stock Price & Moving Averages")
                plt.figure(figsize=(10, 5))
                plt.plot(data.index, data["Close"], label="Stock Price", color="blue")
                plt.plot(data.index, data["SMA_50"], label="SMA 50", linestyle="dashed", color="orange")
                plt.plot(data.index, data["SMA_200"], label="SMA 200", linestyle="dashed", color="red")
                plt.title(f"{ticker} Stock Price & Moving Averages")
                plt.xlabel("Date")
                plt.ylabel("Price (USD)")
                plt.legend()
                st.pyplot(plt)


# In[ ]:




