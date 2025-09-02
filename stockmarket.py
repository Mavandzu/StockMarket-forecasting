#!/usr/bin/env python
# coding: utf-8

# In[1]:


# stockmarket.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pandas_datareader import data as pdr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

st.set_page_config(page_title="Stock Market Forecast Dashboard", layout="wide")
st.title("📊 Stock Market Forecast Dashboard")

# -----------------------------
# 1️⃣ User Input
# -----------------------------
tickers_input = st.text_input("Enter stock tickers (comma-separated, e.g. AAPL,MSFT,GOOGL)", "AAPL,MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(",")]
start = "2021-01-01"
end = "2026-01-01"

# -----------------------------
# 2️⃣ Download Stock Prices Safely
# -----------------------------
data = {}
failed_tickers = []

for ticker in tickers:
    attempts = 0
    success = False
    while attempts < 3 and not success:
        try:
            st.info(f"Downloading {ticker}...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            data[ticker] = df
            success = True
        except Exception as e:
            attempts += 1
            st.warning(f"Attempt {attempts} failed for {ticker}: {e}")
            time.sleep(1)
    if not success:
        failed_tickers.append(ticker)

if failed_tickers:
    st.error(f"Failed to download: {', '.join(failed_tickers)}")

if not data:
    st.stop()  # stop if no data

# -----------------------------
# 3️⃣ Normalize & Plot Close Prices
# -----------------------------
normalized = pd.DataFrame({t: df["Close"] for t, df in data.items()})
normalized = normalized / normalized.iloc[0] * 100
st.subheader("📈 Normalized Stock Prices")
st.line_chart(normalized, use_container_width=True, height=400)

# -----------------------------
# 4️⃣ Risk & Return Metrics
# -----------------------------
returns = pd.DataFrame({t: df["Close"].pct_change() for t, df in data.items()}).dropna()
mean_daily_return = returns.mean()
volatility = returns.std()
annual_return = mean_daily_return * 252
annual_volatility = volatility * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility

risk_return = pd.DataFrame({
    "Annual Return (%)": annual_return * 100,
    "Annual Volatility (%)": annual_volatility * 100,
    "Sharpe Ratio": sharpe_ratio
}).round(2)

st.subheader("💹 Risk & Return Metrics")
st.dataframe(risk_return)

# -----------------------------
# 5️⃣ Cumulative Returns
# -----------------------------
cumulative_returns = (1 + returns).cumprod()
st.subheader("💰 Cumulative Returns")
st.line_chart(cumulative_returns, use_container_width=True, height=400)

# -----------------------------
# 6️⃣ Macroeconomic Indicators (FRED)
# -----------------------------
fred_series = {"CPI": "CPIAUCSL", "Unemployment": "UNRATE", "Interest Rate": "FEDFUNDS"}
macro_data = {}
for name, code in fred_series.items():
    try:
        df = pdr.DataReader(code, "fred", start, end)
        df.rename(columns={code: name}, inplace=True)
        macro_data[name] = df
    except Exception as e:
        st.warning(f"Could not load {name} data: {e}")

if macro_data:
    macro_df = pd.concat(macro_data.values(), axis=1).fillna(method="ffill").dropna()
    stock_monthly = normalized.resample("M").mean()
    macro_monthly = macro_df.resample("M").mean()
    combined_monthly = stock_monthly.join(macro_monthly, how="inner")

    # Correlation Heatmap
    st.subheader("🔗 Correlation Matrix")
    corr = combined_monthly.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# -----------------------------
# 7️⃣ Forecasting (Random Forest)
# -----------------------------
st.subheader("🔮 Machine Learning Forecasts")
ml_forecasts = {}

for ticker in stock_monthly.columns:
    series = stock_monthly[ticker].dropna()
    if len(series) < 24:
        continue
    df_feat = pd.DataFrame({
        "y": series,
        "lag1": series.shift(1),
        "lag2": series.shift(2),
        "lag3": series.shift(3)
    }).dropna()
    X, y = df_feat[["lag1", "lag2", "lag3"]], df_feat["y"]
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        actuals.extend(y_test)

    mae = mean_absolute_error(actuals, preds)
    mape = mean_absolute_percentage_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    model.fit(X, y)
    forecast = model.predict(X.iloc[[-1]])[0]

    ml_forecasts[ticker] = {
        "mae": mae, "mape": mape, "r2": r2, "forecast": forecast,
        "cv_preds": pd.Series(preds, index=y.index[-len(preds):])
    }

# Forecast table
results_table = pd.DataFrame({
    t: {
        "MAE": ml_forecasts[t]["mae"],
        "MAPE": ml_forecasts[t]["mape"],
        "R²": ml_forecasts[t]["r2"],
        "Next Forecast": ml_forecasts[t]["forecast"]
    } for t in ml_forecasts
}).T.round(4)

st.dataframe(results_table)

# Forecast plots
st.subheader("📈 Forecast Plots")
for ticker, info in ml_forecasts.items():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_monthly[ticker], label="Actual", marker='o')
    ax.plot(info["cv_preds"], label="CV Predictions", linestyle="--", marker='x')
    ax.scatter(stock_monthly.index[-1] + pd.DateOffset(months=1),
               info["forecast"], color="red", label="Next Forecast", s=100, zorder=5)
    ax.set_title(f"{ticker} - Monthly Closing Price & Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

