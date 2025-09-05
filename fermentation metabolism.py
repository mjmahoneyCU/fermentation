import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config(page_title="Yeast Metabolism Simulator", layout="wide")
st.title("🧬 Yeast Metabolism Simulation")

st.markdown("""
This simulation models biomass growth, glucose consumption, and ethanol production under four common yeast metabolic modes:

- **Respiration**: Aerobic growth with no ethanol production.
- **Crabtree Effect**: Aerobic overflow metabolism — cells produce ethanol even when oxygen is present, due to excess glucose.
- **Anaerobic Fermentation**: Ethanol is produced as the sole byproduct, in the absence of oxygen.
- **Fed-Batch**: Glucose is added continuously, allowing controlled growth and product formation.

The kinetic parameters used for each mode are listed in the table below. These values represent biological assumptions about yields, growth rates, and product formation under different conditions.

🔍 **Note**: For the **aerobic modes** (*respiration*, *Crabtree*, and *fed-batch*), this model assumes oxygen is **not limiting** — that is, oxygen transfer via agitation and aeration is sufficient to meet cellular demand. In real bioreactors, however, oxygen supply becomes a critical constraint as biomass increases. During **Week 2 of your lab**, you’ll need to monitor this carefully and adjust **kₗₐ** (oxygen transfer coefficient) by changing agitation speed or gas flow rate to maintain dissolved oxygen levels and avoid oxygen-limited growth.

In contrast, the **anaerobic condition** assumes that no oxygen is available at all, and cells rely solely on fermentative metabolism to grow and produce ethanol.
""")

st.markdown("""
Simulates glucose consumption, ethanol production and oxidation, and biomass growth under different yeast metabolic modes:
- **Respiration**
- **Crabtree Effect**
- **Anaerobic Fermentation**
- **Fed-Batch (with constant glucose feed)**

Use the sliders to explore how parameters affect the outcomes. Ethanol oxidation activates when glucose is low.
""")

# --- Parameter Guide ---
with st.expander("📘 Parameter Guide", expanded=False):
    st.markdown("""
    | Parameter        | Description                                  |
    |------------------|----------------------------------------------|
    | `μ_max`          | Max specific growth rate on glucose (1/h)    |
    | `Yxs`            | Biomass yield on glucose (g/g)               |
    | `Yps`            | Ethanol yield on glucose (g/g)               |
    | `Yxe`            | Biomass yield on ethanol (g/g)               |
    | `Ks`             | Glucose Monod constant (g/L)                 |
    | `Kp`             | Ethanol inhibition constant (g/L)            |
    | `Ke`             | Ethanol Monod constant (g/L)                 |
    | `Sf_feed`        | Feed glucose concentration (g/L, fed-batch)  |
    | `F_rate`         | Glucose feed rate (g/L/h, fed-batch)         |
    """)

# --- Sidebar Controls ---
st.sidebar.header("Global Parameters")
ethanol_inhibition = st.sidebar.checkbox("Enable Ethanol Inhibition", value=True)
Se_threshold = st.sidebar.slider("Glucose Threshold for Ethanol Oxidation (g/L)", 0.1, 5.0, 1.0)
S0_user = st.sidebar.slider("Initial Glucose (Non-Respiration) (g/L)", 1.0, 50.0, 20.0)
t_end = st.sidebar.slider("Simulation Time (h)", 5, 72, 24)

st.sidebar.header("Fed-Batch Only Parameters")
Sf_feed = st.sidebar.slider("Feed Glucose Concentration (g/L)", 10.0, 100.0, 50.0)
F_rate = st.sidebar.slider("Feed Rate (g/L/h)", 0.0, 2.0, 0.1)

# --- Initial Biomass & Ethanol ---
X0 = 0.1   # Biomass (g/L)
P0 = 0.0   # Ethanol (g/L)

# --- ODE Model ---
def fermentation_model(t, y, mu_max, Yxs, Yps, Yxe, Ks, Kp, Ke, mode, inhibition, Se_thresh, Sf_feed=0, F_rate=0):
    X, S, P = y
    mu_ethanol = 0

    if inhibition:
        mu_glucose = mu_max * (S / (Ks + S)) * (1 / (1 + P / Kp))
    else:
        mu_glucose = mu_max * (S / (Ks + S))

    dXdt = mu_glucose * X
    dSdt = - (1 / Yxs) * dXdt
    dPdt = (Yps / Yxs) * dXdt if mode != "respiration" else 0

    # Ethanol oxidation
    if S < Se_thresh and P > 0:
        mu_ethanol = mu_max * (P / (Ke + P))
        dXdt += mu_ethanol * X
        dPdt -= (1 / Yxe) * mu_ethanol * X

    # Fed-batch feed
    if mode == "fed-batch":
        dSdt += F_rate

    return [dXdt, dSdt, dPdt]

# --- Simulation Runner ---
def run_simulation(mode):
    S0 = 0.5 if mode == "respiration" else S0_user
    init_conditions = [X0, S0, P0]

    if mode == "respiration":
        params = (0.3, 0.5, 0.0, 0.4, 1.0, 50.0, 2.0, mode, ethanol_inhibition, Se_threshold)
    elif mode == "crabtree":
        params = (0.4, 0.45, 0.5, 0.4, 1.0, 50.0, 2.0, mode, ethanol_inhibition, Se_threshold)
    elif mode == "anaerobic":
        params = (0.25, 0.4, 0.6, 0.4, 1.0, 50.0, 2.0, mode, ethanol_inhibition, Se_threshold)
    elif mode == "fed-batch":
        params = (0.25, 0.4, 0.6, 0.4, 1.0, 50.0, 2.0, mode, ethanol_inhibition, Se_threshold, Sf_feed, F_rate)
    else:
        raise ValueError("Unknown mode")

    sol = solve_ivp(
        fermentation_model,
        [0, t_end],
        init_conditions,
        args=params,
        t_eval=np.linspace(0, t_end, 300),
        method='RK45'
    )
    return sol.t, sol.y

# --- Plotting ---
modes = ["respiration", "crabtree", "anaerobic", "fed-batch"]
colors = {"Biomass": "green", "Glucose": "blue", "Ethanol": "red"}

plots = []
for mode in modes:
    t, (X, S, P) = run_simulation(mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=X, mode='lines', name="Biomass", line=dict(color=colors["Biomass"])))
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name="Glucose", line=dict(color=colors["Glucose"])))
    fig.add_trace(go.Scatter(x=t, y=P, mode='lines', name="Ethanol", line=dict(color=colors["Ethanol"])))
    fig.update_layout(
        title=mode.capitalize(),
        xaxis_title="Time (h)",
        yaxis_title="Concentration (g/L)" if mode == "crabtree" else "",
        hovermode="x unified",
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 2]) if mode == "respiration" else None
    )
    plots.append(fig)

# --- Display Plots in 4 Columns ---
cols = st.columns(4)
for col, fig in zip(cols, plots):
    with col:
        st.plotly_chart(fig, use_container_width=True)

# --- Legend in Expander (with units) ---
with st.expander("📌 Legend", expanded=False):
    st.markdown("""
**Line Colors and Units:**
- 🟩 **Biomass** (g/L) – green  
- 🔵 **Glucose** (g/L) – blue  
- 🔴 **Ethanol** (g/L) – red  
""")

# --- Parameter Table ---
st.subheader("📊 Mode-Specific Parameters")
df = pd.DataFrame({
    "Mode": ["Respiration", "Crabtree", "Anaerobic", "Fed-Batch"],
    "μ_max": [0.3, 0.4, 0.25, 0.25],
    "Yxs": [0.5, 0.45, 0.4, 0.4],
    "Yps": [0.0, 0.5, 0.6, 0.6],
    "Sf_feed (if used)": ["-", "-", "-", f"{Sf_feed:.1f}"],
    "F_rate (if used)": ["-", "-", "-", f"{F_rate:.2f}"]
})
numeric_format = {
    "μ_max": "{:.2f}",
    "Yxs": "{:.2f}",
    "Yps": "{:.2f}"
}
styled_df = df.style.format(numeric_format)
st.dataframe(styled_df, use_container_width=True)

# --- Reflection Questions ---
with st.expander("🧠 Reflection Questions"):
    st.markdown("""
1. **Which condition results in the highest final biomass concentration? Why?**
2. **What happens to ethanol levels when glucose becomes limiting?**
3. **Under what conditions does ethanol oxidation become relevant?**
4. **How does enabling ethanol inhibition affect the outcomes?**
5. **Compare how substrate is used in batch vs fed-batch conditions.**
""")
