import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config(page_title="Yeast Metabolism Simulator", layout="wide")
st.title("Yeast Metabolism Simulation")

st.markdown("""
This simulation models biomass growth, glucose consumption, and ethanol production under four common yeast metabolic modes:

- **Respiration**: Aerobic growth with no ethanol production.
- **Crabtree Effect**: Aerobic overflow metabolism — cells produce ethanol even when oxygen is present, due to excess glucose.
- **Anaerobic Fermentation**: Ethanol is produced as the sole byproduct, in the absence of oxygen.
- **Fed-Batch**: Glucose is added continuously, allowing controlled growth and product formation.

🔍 **Note**: Oxygen is assumed to be non-limiting in the aerobic modes (*respiration*, *Crabtree*, and *fed-batch aerobic*). During Week 2 of your lab, you’ll need to monitor dissolved oxygen and adjust agitation or airflow to maintain oxygen availability.
""")

st.markdown("Use the sliders to explore how parameters affect the outcomes. Ethanol oxidation occurs only in aerobic conditions when glucose is low.")

# --- Sidebar Controls ---
st.sidebar.header("Global Parameters")
ethanol_inhibition = st.sidebar.checkbox("Enable Ethanol Inhibition", value=True)
Se_threshold = st.sidebar.slider("Glucose Threshold for Ethanol Oxidation (g/L)", 0.1, 5.0, 1.0)
S0_user = st.sidebar.slider("Initial Glucose (Non-Respiration) (g/L)", 1.0, 50.0, 20.0)
X0_user = st.sidebar.slider("Initial Biomass (g/L)", 0.01, 1.0, 0.1)
t_end = st.sidebar.slider("Simulation Time (h)", 5, 72, 24)

st.sidebar.header("Fed-Batch Settings")
fedbatch_aerobic = st.sidebar.radio("Fed-Batch Mode:", ["Aerobic", "Anaerobic"])
F_rate = st.sidebar.slider("Feed Rate (g/L/h)", 0.0, 2.0, 0.1)

# --- Initial Conditions ---
P0 = 0.0  # Ethanol

# --- ODE Model ---
def fermentation_model(t, y, mu_max, Yxs, Yps, Yxe, Ks, Kp, Ke, aerobic, inhibition, Se_thresh, F_rate=0, mode="batch"):
    X, S, P = y
    mu_ethanol = 0

    if inhibition:
        mu_glucose = mu_max * (S / (Ks + S)) * (1 / (1 + P / Kp))
    else:
        mu_glucose = mu_max * (S / (Ks + S))

    dXdt = mu_glucose * X
    dSdt = - (1 / Yxs) * dXdt
    dPdt = (Yps / Yxs) * dXdt if Yps > 0 else 0

    if aerobic and S < Se_thresh and P > 0:
        mu_ethanol = mu_max * (P / (Ke + P))
        dXdt += mu_ethanol * X
        dPdt -= (1 / Yxe) * mu_ethanol * X

    if mode == "fed-batch":
        dSdt += F_rate

    return [dXdt, dSdt, dPdt]

# --- Simulation Runner ---
def run_simulation(mode):
    if mode == "respiration":
        params = (0.3, 0.5, 0.0, 0.4, 1.0, 50.0, 2.0, True, ethanol_inhibition, Se_threshold)
        S0 = 0.5
        fb_params = {"F_rate": 0}
    elif mode == "crabtree":
        params = (0.4, 0.45, 0.5, 0.4, 1.0, 50.0, 2.0, True, ethanol_inhibition, Se_threshold)
        S0 = S0_user
        fb_params = {"F_rate": 0}
    elif mode == "anaerobic":
        params = (0.25, 0.4, 0.6, 0.4, 1.0, 50.0, 2.0, False, ethanol_inhibition, Se_threshold)
        S0 = S0_user
        fb_params = {"F_rate": 0}
    elif mode == "fed-batch":
        if fedbatch_aerobic == "Aerobic":
            params = (0.4, 0.45, 0.5, 0.4, 1.0, 50.0, 2.0, True, ethanol_inhibition, Se_threshold)
        else:
            params = (0.25, 0.4, 0.6, 0.4, 1.0, 50.0, 2.0, False, ethanol_inhibition, Se_threshold)
        S0 = S0_user
        fb_params = {"F_rate": F_rate}
    else:
        raise ValueError("Invalid mode")

    y0 = [X0_user, S0, P0]
    sol = solve_ivp(
        fermentation_model,
        [0, t_end],
        y0,
        t_eval=np.linspace(0, t_end, 300),
        args=params + tuple(fb_params.values()) + (mode,)
    )
    return sol.t, sol.y

# --- Plotting ---
modes = ["respiration", "crabtree", "anaerobic", "fed-batch"]
colors = {"Biomass": "green", "Glucose": "blue", "Ethanol": "red"}

cols = st.columns(4)
for mode, col in zip(modes, cols):
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
    with col:
        st.plotly_chart(fig, use_container_width=True)

# --- Legend ---
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
    "Mode": ["Respiration", "Crabtree", "Anaerobic", "Fed-Batch (selected)"],
    "μ_max": [0.3, 0.4, 0.25, 0.4 if fedbatch_aerobic == "Aerobic" else 0.25],
    "Yxs": [0.5, 0.45, 0.4, 0.45 if fedbatch_aerobic == "Aerobic" else 0.4],
    "Yps": [0.0, 0.5, 0.6, 0.5 if fedbatch_aerobic == "Aerobic" else 0.6],
    "F_rate": ["-", "-", "-", f"{F_rate:.2f}"]
})
st.dataframe(df.style.format({"μ_max": "{:.2f}", "Yxs": "{:.2f}", "Yps": "{:.2f}"}), use_container_width=True)

# --- Reflection Questions ---
with st.expander("🧠 Reflection Questions"):
    st.markdown("""
1. **Which condition results in the highest final biomass concentration? Why?**
2. **What happens to ethanol levels when glucose becomes limiting?**
3. **Under what conditions does ethanol oxidation become relevant?**
4. **How does enabling ethanol inhibition affect the outcomes?**
5. **Compare how substrate is used in batch vs fed-batch conditions.**
""")
