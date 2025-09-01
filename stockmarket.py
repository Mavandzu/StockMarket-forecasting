#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as web
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# -----------------------------
# 1. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stock & Macro Analysis", layout="wide")

st.title("📊 Stock & Macro Forecasting Dashboard")

tickers = st.text_input(
    "Enter stock tickers (comma-separated, e.g. AAPL,MSFT,GOOGL):",
    "AAPL,MSFT"
).upper().split(",")

start = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
end = st.date_input("End Date", pd.to_datetime("2026-01-01"))

if st.button("Run Analysis"):

    # -----------------------------
    # 2. Download Stock Prices
    # -----------------------------
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if data.empty:
        st.error("No data found for these tickers. Try different symbols.")
        st.stop()

    # Normalize Close Prices
    normalized = data['Close'] / data['Close'].iloc[0] * 100

    # Latest Prices
    last_prices = data['Close'].iloc[-1]
    pct_changes = data['Close'].pct_change().iloc[-1] * 100

    st.subheader("📊 Latest Prices and % Daily Change")
    st.write(pd.DataFrame({
        "Last Price": last_prices.round(2),
        "% Change (1D)": pct_changes.round(2)
    }))

    # Plot normalized prices
    st.subheader("📈 Stock Price Comparison (Normalized)")
    st.line_chart(normalized)

    # -----------------------------
    # 3. Risk and Return Analysis
    # -----------------------------
    returns = data['Close'].pct_change().dropna()
    mean_daily_return = returns.mean()
    volatility = returns.std()

    annual_return = mean_daily_return * 252
    annual_volatility = volatility * (252 ** 0.5)
    sharpe_ratio = annual_return / annual_volatility

    risk_return = pd.DataFrame({
        'Annual Return (%)': annual_return * 100,
        'Annual Volatility (%)': annual_volatility * 100,
        'Sharpe Ratio': sharpe_ratio
    }).round(2)

    st.subheader("📈 Risk and Return Metrics")
    st.dataframe(risk_return)

    st.subheader("💹 Cumulative Returns (Growth of $1)")
    cumulative_returns = (1 + returns).cumprod()
    st.line_chart(cumulative_returns)

    # -----------------------------
    # 4. Macroeconomic Indicators
    # -----------------------------
    fred_series = {
        'CPI': 'CPIAUCSL',
        'Unemployment': 'UNRATE',
        'Interest Rate': 'FEDFUNDS'
    }

    macro_data = {}
    for name, code in fred_series.items():
        df = web.DataReader(code, 'fred', start, end)
        df.rename(columns={code: name}, inplace=True)
        macro_data[name] = df

    macro_df = pd.concat(macro_data.values(), axis=1).fillna(method="ffill").dropna()

    stock_monthly = data['Close'].resample('M').mean()
    macro_monthly = macro_df.resample('M').mean()

    combined = pd.concat([stock_monthly, macro_monthly], axis=1).dropna()

    st.subheader("🔗 Correlation Matrix")
    corr = combined.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # 5. Machine Learning Forecasting
    # -----------------------------
    st.subheader("🔮 Machine Learning Forecasts (Random Forest)")
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

        X = df_feat[["lag1", "lag2", "lag3"]]
        y = df_feat["y"]

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
            "MAE": mae,
            "MAPE": mape,
            "R²": r2,
            "Next Forecast": forecast,
        }

    if ml_forecasts:
        results_table = pd.DataFrame(ml_forecasts).T.round(4)
        st.dataframe(results_table)

        forecast_summary = pd.DataFrame({
            "Last Price": stock_monthly.iloc[-1],
            "Forecast Next": [ml_forecasts[t]["Next Forecast"] if t in ml_forecasts else np.nan 
                              for t in stock_monthly.columns]
        })
        forecast_summary["Change (%)"] = (
            (forecast_summary["Forecast Next"] - forecast_summary["Last Price"]) / forecast_summary["Last Price"] * 100
        )
        st.subheader("📈 Forecast Summary (Last vs Next)")
        st.dataframe(forecast_summary.round(2))
    else:
        st.warning("Not enough data for ML forecasts.")

