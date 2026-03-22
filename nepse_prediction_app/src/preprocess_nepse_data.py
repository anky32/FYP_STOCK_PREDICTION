import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler

# Folder paths
input_folder = "data/raw"
output_folder = "data/cleaned"

os.makedirs(output_folder, exist_ok=True)

# Get all CSV files
files = glob.glob(input_folder + "/*.csv")

for file in files:

    print("\nProcessing:", file)

    # Load dataset
    df = pd.read_csv(file)

    # Replace "-" with NaN
    df.replace("-", pd.NA, inplace=True)

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = ["Open","High","Low","Close","Volume"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by Date
    if "Date" in df.columns:
        df = df.sort_values("Date")

    # Forward fill missing values
    df = df.ffill()

    # Remove useless columns
    drop_cols = ["Symbol", "Volume"]

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Drop rows where Close is missing
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    # Normalize price columns
    price_cols = ["Open","High","Low","Close"]

    available_cols = [col for col in price_cols if col in df.columns]

    if len(available_cols) > 0:

        scaler = MinMaxScaler()

        scaled_values = scaler.fit_transform(df[available_cols])

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=[col + "_scaled" for col in available_cols],
            index=df.index
        )

        df = pd.concat([df, scaled_df], axis=1)

    # Save cleaned dataset
    filename = os.path.basename(file)

    new_file = os.path.join(
        output_folder,
        filename.replace(".csv","_clean.csv")
    )

    df.to_csv(new_file, index=False)

    print("Saved:", new_file)

print("\nAll datasets cleaned and normalized successfully.")