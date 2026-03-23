import pandas as pd
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import base64
from io import BytesIO

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

        # ============================
        # 🔥 MOVING AVERAGES
        # ============================
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()

        # ============================
        # 🔥 RUN MODEL
        # ============================
        if model_type == "lstm":
            predictions = run_lstm_model(df, stock_name)

        elif model_type == "arima":
            predictions = run_arima_model(df)

        elif model_type == "hybrid":
            predictions = run_hybrid_model(df)

        else:
            return {"error": "Invalid model type"}

        # ============================
        # 🔥 LOAD METRICS
        # ============================
        metrics_path = os.path.join(BASE_DIR, "results", "model_comparison_results.csv")

        mae = None
        rmse = None

        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)

            metrics_df["Stock"] = metrics_df["Stock"].str.strip().str.upper()
            stock_name = stock_name.strip().upper()

            row = metrics_df[metrics_df["Stock"] == stock_name]

            if not row.empty:
                if model_type == "arima":
                    mae = float(row["ARIMA_MAE"].values[0])
                    rmse = float(row["ARIMA_RMSE"].values[0])

                elif model_type == "lstm":
                    mae = float(row["LSTM_MAE"].values[0])
                    rmse = float(row["LSTM_RMSE"].values[0])

                elif model_type == "hybrid":
                    mae = float(row["HYBRID_MAE"].values[0])
                    rmse = float(row["HYBRID_RMSE"].values[0])

        # ============================
        # 🔥 LOAD BACKTEST RESULTS
        # ============================
        backtest_path = os.path.join(BASE_DIR, "results", "backtesting_results.csv")

        backtest_data = {
            "profit": None,
            "final_balance": None,
            "win_rate": None,
            "total_trades": None
        }

        if os.path.exists(backtest_path):
            bt_df = pd.read_csv(backtest_path)

            bt_df["Stock"] = bt_df["Stock"].str.strip().str.upper()

            row_bt = bt_df[bt_df["Stock"] == stock_name]

            if not row_bt.empty:
                backtest_data = {
                    "profit": float(row_bt["Profit"].values[0]),
                    "final_balance": float(row_bt["Final Balance"].values[0]),
                    "win_rate": float(row_bt["Win Rate (%)"].values[0]),
                    "total_trades": int(row_bt["Total Trades"].values[0])
                }

        # ============================
        # 🔥 FUTURE DATES
        # ============================
        future_dates = []
        current = datetime.today()

        while len(future_dates) < 5:
            current += timedelta(days=1)
            if current.weekday() < 5:
                future_dates.append(current)

        # ============================
        # 🔥 GRAPH
        # ============================
        plt.figure(figsize=(10, 5))

        plt.plot(df.index, df["Close"], label="Historical")

        plt.plot(df.index, df["MA_20"], linestyle="--", label="MA 20")
        plt.plot(df.index, df["MA_50"], linestyle="--", label="MA 50")

        plt.plot(
            future_dates,
            predictions,
            marker='o',
            linestyle='--',
            label="Prediction"
        )

        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        graph = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()

        # ============================
        # 🔥 FINAL OUTPUT
        # ============================
        combined = list(zip(
            [d.strftime("%Y-%m-%d") for d in future_dates],
            predictions
        ))

        return {
            "stock": stock_name,
            "model": model_type.upper(),
            "predictions": combined,

            "metrics": {
                "MAE": mae,
                "RMSE": rmse
            },

            "trend": {
                "MA20": float(df["MA_20"].iloc[-1]),
                "MA50": float(df["MA_50"].iloc[-1])
            },

            "backtest": backtest_data,

            "graph": graph
        }

    except Exception as e:
        return {"error": str(e)}