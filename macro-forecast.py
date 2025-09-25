import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import joblib

input_vars = [
    "FEDFUNDS",       # Monetary policy
    "GS10",
    "CE16OV",
    'HOUST',
    "USGOVT",        # Fiscal policy
    "OILPRICEx",     # Commodity prices
    "USTRADE", 
    "TWEXAFEGSMTHx" # Exchange rates

]

output_vars = [
    "CPIAUCSL",   # Inflation
    "UNRATE",     # Unemployment
    "PAYEMS",     # Payroll employment
    "USCONS",     # Consumer spending
    "UMCSENTx",    # Consumer sentiment
    "S&P 500",   # Stock market performance
    "INDPRO",   # Industrial production
    "RPI"   # Real personal income
]


df = pd.read_csv("2025-08-md.csv")
df = df[input_vars + output_vars]
df = df.fillna(method="ffill").fillna(method="bfill")
df = df.dropna()
print("Input vars:", len(input_vars), input_vars)
print("Output vars:", len(output_vars), output_vars)
print("Dataframe columns:", df.columns.tolist())
print("Shape of X:", df[input_vars].shape)
final_models = {}

for target in output_vars:
    X = df[input_vars].values
    y = df[target].values

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    final_models[target] = model
    print(f"Trained final model for {target}")

for target, model in final_models.items():
    joblib.dump(model, f"{target}_model.pkl")
print("All models saved.")



