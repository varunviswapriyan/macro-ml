import streamlit as st
import pandas as pd
import joblib
import numpy as np


input_bounds = {
    "FEDFUNDS": (0.05, 19.1),
    "GS10": (0.62, 15.32),
    "CE16OV": (63684, 163969),
    "HOUST": (478, 2494),
    "USGOVT": (0.8105, 23618),
    "OILPRICEx": (2.92, 133.93),
    "USTRADE": (5350.3, 15872.6),
    "TWEXAFEGSMTHx": (83, 170)
}


output_vars = [
    "CPIAUCSL", "UNRATE", "PAYEMS", "USCONS",
    "UMCSENTx", "S&P 500", "INDPRO", "RPI"
]


models = {}
for target in output_vars:
    models[target] = joblib.load(f"{target}_model.pkl")

st.title("Macroeconomic Forecast Simulator")
st.sidebar.header("Input Variables")


inputs = {}
for var, (min_val, max_val) in input_bounds.items():
    min_val, max_val = float(min_val), float(max_val)
    default = (min_val + max_val) / 2 
    inputs[var] = st.sidebar.slider(var, min_val, max_val, default, step=(max_val-min_val)/1000.0)


X = np.array([list(inputs.values())])

st.header("Predicted Outputs")

for target, model in models.items():
    prediction = model.predict(X)[0]
    st.write(f"**{target}:** {prediction:.2f}")
