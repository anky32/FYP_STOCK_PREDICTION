import pandas as pd
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

from models.lstm_model import run_lstm_model
from models.arima_model import run_arima_model
from models.hybrid_model import run_hybrid_model


def run_prediction(stock_name, model_type="lstm"):
    try:
        stock_name = stock_name.strip().upper()

        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        file_path = os.path.join(
            BASE_DIR, "data", "cleaned", f"{stock_name}_data_clean.csv"
        )

        if not os.path.exists(file_path):
            return {"error": f"File not found: {stock_name}_data_clean.csv"}

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.capitalize()

        if "Date" not in df.columns or "Close" not in df.columns:
            return {"error": "Required columns missing"}

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)

        # ── Technical Indicators ──────────────────────────────────────────
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()

        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(window=14).mean()
        loss  = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs    = gain / loss.replace(0, 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["BB_Mid"]   = df["Close"].rolling(window=20).mean()
        df["BB_Std"]   = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

        # ── Run Model ─────────────────────────────────────────────────────
        if model_type == "lstm":
            predictions = run_lstm_model(df, stock_name)
        elif model_type == "arima":
            predictions = run_arima_model(df)
        elif model_type == "hybrid":
            predictions = run_hybrid_model(df)
        else:
            return {"error": "Invalid model type"}

        # ── Load Metrics + R2 from pre-computed CSV (no live computation) ─
        metrics_path = os.path.join(BASE_DIR, "results", "model_comparison_results.csv")
        mae = rmse = r2 = None

        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            metrics_df["Stock"] = metrics_df["Stock"].str.strip().str.upper()
            row = metrics_df[metrics_df["Stock"] == stock_name]

            if not row.empty:
                if model_type == "arima":
                    mae  = float(row["ARIMA_MAE"].values[0])
                    rmse = float(row["ARIMA_RMSE"].values[0])
                    if "ARIMA_R2" in row.columns:
                        r2 = float(row["ARIMA_R2"].values[0])
                elif model_type == "lstm":
                    mae  = float(row["LSTM_MAE"].values[0])
                    rmse = float(row["LSTM_RMSE"].values[0])
                    if "LSTM_R2" in row.columns:
                        r2 = float(row["LSTM_R2"].values[0])
                elif model_type == "hybrid":
                    mae  = float(row["HYBRID_MAE"].values[0])
                    rmse = float(row["HYBRID_RMSE"].values[0])
                    if "HYBRID_R2" in row.columns:
                        r2 = float(row["HYBRID_R2"].values[0])

        # ── Backtest Results ──────────────────────────────────────────────
        backtest_path = os.path.join(BASE_DIR, "results", "backtesting_results.csv")
        backtest_data = {"profit": None, "final_balance": None,
                         "win_rate": None, "total_trades": None}

        if os.path.exists(backtest_path):
            bt_df = pd.read_csv(backtest_path)
            bt_df["Stock"] = bt_df["Stock"].str.strip().str.upper()
            row_bt = bt_df[bt_df["Stock"] == stock_name]
            if not row_bt.empty:
                backtest_data = {
                    "profit":        float(row_bt["Profit"].values[0]),
                    "final_balance": float(row_bt["Final Balance"].values[0]),
                    "win_rate":      float(row_bt["Win Rate (%)"].values[0]),
                    "total_trades":  int(row_bt["Total Trades"].values[0])
                }

        # ── Future Dates (weekdays only) ──────────────────────────────────
        future_dates = []
        current = datetime.today()
        while len(future_dates) < 5:
            current += timedelta(days=1)
            if current.weekday() < 5:
                future_dates.append(current)

        # ── Chart ─────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        facecolor='#0d1117')
        fig.subplots_adjust(hspace=0.1)

        ax1.set_facecolor('#0d1117')
        ax1.plot(df.index, df["Close"],    color='#e2e8f0', linewidth=1.2, label='Close Price')
        ax1.plot(df.index, df["MA_20"],    color='#3b82f6', linewidth=1,   linestyle='--', label='MA20')
        ax1.plot(df.index, df["MA_50"],    color='#a855f7', linewidth=1,   linestyle='--', label='MA50')
        ax1.plot(df.index, df["EMA_12"],   color='#f59e0b', linewidth=0.8, linestyle=':',  label='EMA12')
        ax1.plot(df.index, df["BB_Upper"], color='#6ee7b7', linewidth=0.7, linestyle='--', label='BB Upper', alpha=0.7)
        ax1.plot(df.index, df["BB_Lower"], color='#6ee7b7', linewidth=0.7, linestyle='--', label='BB Lower', alpha=0.7)
        ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.05, color='#14b8a6')
        ax1.plot(future_dates, predictions, color='#14b8a6', linewidth=2,
                 marker='o', markersize=5, linestyle='--', label='Forecast')
        ax1.legend(loc='upper left', fontsize=7, facecolor='#0d1117', labelcolor='white', framealpha=0.5)
        ax1.tick_params(colors='#64748b', labelsize=8)
        ax1.spines[:].set_color('#1e293b')
        ax1.set_ylabel('Price (NPR)', color='#64748b', fontsize=9)

        ax2.set_facecolor('#0d1117')
        ax2.plot(df.index, df["RSI"], color='#f59e0b', linewidth=1, label='RSI(14)')
        ax2.axhline(70, color='#ef4444', linewidth=0.7, linestyle='--', alpha=0.7)
        ax2.axhline(30, color='#22c55e', linewidth=0.7, linestyle='--', alpha=0.7)
        ax2.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.15, color='#ef4444')
        ax2.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.15, color='#22c55e')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI', color='#64748b', fontsize=9)
        ax2.tick_params(colors='#64748b', labelsize=8)
        ax2.spines[:].set_color('#1e293b')
        ax2.legend(loc='upper left', fontsize=7, facecolor='#0d1117', labelcolor='white', framealpha=0.5)

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight', facecolor='#0d1117')
        buffer.seek(0)
        graph = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()

        # ── Final Output ──────────────────────────────────────────────────
        combined = list(zip(
            [d.strftime("%Y-%m-%d") for d in future_dates],
            predictions
        ))

        return {
            "stock":       stock_name,
            "model":       model_type.upper(),
            "predictions": combined,
            "metrics":     {"MAE": mae, "RMSE": rmse, "R2": r2},
            "trend": {
                "MA20":     float(df["MA_20"].iloc[-1]),
                "MA50":     float(df["MA_50"].iloc[-1]),
                "EMA12":    float(df["EMA_12"].iloc[-1]),
                "EMA26":    float(df["EMA_26"].iloc[-1]),
                "RSI":      float(df["RSI"].iloc[-1]),
                "BB_Upper": float(df["BB_Upper"].iloc[-1]),
                "BB_Mid":   float(df["BB_Mid"].iloc[-1]),
                "BB_Lower": float(df["BB_Lower"].iloc[-1]),
            },
            "backtest": backtest_data,
            "graph":    graph
        }

    except Exception as e:
        return {"error": str(e)}
