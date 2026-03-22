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

        # clean columns
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.capitalize()

        if "Date" not in df.columns or "Close" not in df.columns:
            return {"error": "Required columns missing"}

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        # ============================
        # 🔥 RUN MODEL
        # ============================
        if model_type == "lstm":
            result = run_lstm_model(df)

        elif model_type == "arima":
            result = run_arima_model(df)

        elif model_type == "hybrid":
            result = run_hybrid_model(df)

        else:
            return {"error": "Invalid model type"}

        # ============================
        # 🔥 FUTURE DATES FROM TODAY
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
        plt.plot(df["Date"], df["Close"], label="Historical")

        plt.plot(
            future_dates,
            result,
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
            result
        ))

        return {
            "combined": combined,
            "model": model_type.upper(),
            "stock": stock_name,
            "graph": graph
        }

    except Exception as e:
        return {"error": str(e)}