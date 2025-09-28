import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize


df = pd.read_csv("2025-08-md.csv")

input_vars = [
    "FEDFUNDS", "GS10", "CE16OV", "HOUST",
    "USGOVT", "OILPRICEx", "USTRADE", "TWEXAFEGSMTHx"
]

output_vars = [
    "CPIAUCSL", "UNRATE", "PAYEMS", "USCONS",
    "UMCSENTx", "S&P 500", "INDPRO", "RPI"
]

df = df[input_vars + output_vars]
df = df.fillna(method="ffill").fillna(method="bfill").dropna()

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
var_descriptions = {
    # Inputs
    "FEDFUNDS": "Effective Federal Funds Rate (Monetary policy, interest rate)",
    "GS10": "10-Year Treasury Rate (Long-term interest rates, yields)",
    "CE16OV": "Civilian Employment (Total number of employed, CPS survey)",
    "HOUST": "Housing Starts: Total New Privately Owned (Construction activity)",
    "USGOVT": "All Employees: Government (Fiscal policy, government employment)",
    "OILPRICEx": "Crude Oil Prices (spliced WTI and Cushing)",
    "USTRADE": "All Employees: Retail Trade (Consumer spending proxy)",
    "TWEXAFEGSMTHx": "Trade Weighted U.S. Dollar Index (Exchange rate competitiveness)",

    # Outputs
    "CPIAUCSL": "Consumer Price Index: All Items (Inflation measure)",
    "UNRATE": "Civilian Unemployment Rate (Labor market health)",
    "PAYEMS": "All Employees: Total Nonfarm Payrolls (Employment level)",
    "USCONS": "All Employees: Construction (Construction labor market)",
    "UMCSENTx": "University of Michigan Consumer Sentiment Index",
    "S&P 500": "S&P 500 Composite Stock Market Index",
    "INDPRO": "Industrial Production Index (Manufacturing and output activity)",
    "RPI": "Real Personal Income (Income adjusted for inflation)"
}



models = {target: joblib.load(f"{target}_model.pkl") for target in output_vars}


output_minmax = {}
for target in output_vars:
    output_minmax[target] = (df[target].min(), df[target].max())


desired_direction = {
    "CPIAUCSL": -1,   # want lower inflation
    "UNRATE": -1,     # want lower unemployment
    "PAYEMS": 1,      # want higher payroll employment
    "USCONS": 1,      # want higher consumer spending
    "UMCSENTx": 1,    # want higher sentiment
    "S&P 500": 1,     # want higher stock market
    "INDPRO": 1,      # want higher production
    "RPI": 1          # want higher income
}


st.title("Macroeconomic Forecast Simulator and Optimizer")


st.markdown("## 📖 Variable Glossary")
with st.expander("Click to expand variable descriptions"):
    for var, desc in var_descriptions.items():
        st.markdown(f"**{var}**: {desc}")

st.sidebar.header("Input Variables")

#Side bar input variables
inputs = {}
for var, (lo, hi) in input_bounds.items():
    default = (lo + hi) / 2
    inputs[var] = st.sidebar.slider(var, float(lo), float(hi), float(default),
                                    step=(hi - lo) / 1000.0)

X = np.array([list(inputs.values())])

#Predictions for input combinations
st.header("Predicted Outputs")
preds = {}
for target, model in models.items():
    prediction = model.predict(X)[0]
    preds[target] = prediction
    st.write(f"**{target}:** {prediction:.2f}")

#2 factor interaction plots
st.header("Two-Factor Interaction Plots")

selected_output = st.selectbox("Select output variable for interaction plots:", output_vars)
pairs = list(combinations(input_bounds.keys(), 2))
pair = st.selectbox("Select two input variables to vary:", pairs)

if pair:
    var1, var2 = pair
    grid_points = 25
    v1 = np.linspace(input_bounds[var1][0], input_bounds[var1][1], grid_points)
    v2 = np.linspace(input_bounds[var2][0], input_bounds[var2][1], grid_points)
    V1, V2 = np.meshgrid(v1, v2)
    Z = np.zeros_like(V1)

    model = models[selected_output]
    base_inputs = list(inputs.values())
    i1, i2 = list(inputs.keys()).index(var1), list(inputs.keys()).index(var2)

    for i in range(grid_points):
        for j in range(grid_points):
            test_in = base_inputs.copy()
            test_in[i1] = V1[i, j]
            test_in[i2] = V2[i, j]
            Z[i, j] = model.predict([test_in])[0]

    fig, ax = plt.subplots()
    contour = ax.contourf(V1, V2, Z, cmap="viridis")
    plt.colorbar(contour, ax=ax, label=selected_output)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f"Interaction: {var1} vs {var2} on {selected_output}")
    st.pyplot(fig)


st.header("Desirability Optimization")
st.markdown("Set weights (0–1) for each output (importance). Larger weights means stronger optimization for that variable")

weights = {}
for target in output_vars:
    weights[target] = st.slider(f"Weight for {target}", 0.0, 1.0, 0.5, step=0.1)

def desirability_func(x):
    score = 1.0
    for target in output_vars:
        val = models[target].predict([x])[0]
        lo, hi = output_minmax[target]
        if desired_direction[target] == 1:
            d = (val - lo) / (hi - lo)
        else:
            d = (hi - val) / (hi - lo)
        d = np.clip(d, 0, 1)
        score *= d ** weights[target]
    return -score  # minimize negative desirability

if st.button("Run Optimization"):
    bounds = list(input_bounds.values())
    x0 = [(lo + hi) / 2 for lo, hi in bounds]
    result = minimize(desirability_func, x0, bounds=bounds, method="L-BFGS-B")
    best_x = result.x
    best_score = -result.fun

    st.subheader("Optimal Settings")
    for var, val in zip(input_bounds.keys(), best_x):
        st.write(f"**{var}:** {val:.2f}")

    st.write(f"**Overall Desirability Score:** {best_score:.3f}")

    st.subheader("Predicted Outputs at Optimum")
    for target, model in models.items():
        st.write(f"**{target}:** {model.predict([best_x])[0]:.2f}")
