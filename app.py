import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("ERCOT Battery Storage Arbitrage Dashboard")

uploaded_file = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"])

if uploaded_file:

    # ---------- LOAD DATA ----------
    df = pd.read_csv(uploaded_file)
    df.columns = ["Date","Hour","Node","LMP","DST"]

    node = st.selectbox("Select Node", df["Node"].unique())
    node_df = df[df["Node"] == node].sort_values("Hour").reset_index(drop=True)

    # ---------- FIND LOW & HIGH PRICE ----------
    low_i = node_df["LMP"].idxmin()
    high_i = node_df["LMP"].idxmax()

    low_hr = node_df.loc[low_i, "Hour"]
    high_hr = node_df.loc[high_i, "Hour"]

    # ---------- CREATE STORAGE STATES ----------
    node_df["2hr_state"] = 0
    node_df["4hr_state"] = 0

    def apply_window(col, center, hrs, value):
        for i in range(max(0, center-hrs), min(len(node_df), center+hrs+1)):
            node_df.loc[i, col] = value

    # 2-hour storage (±1 hour)
    apply_window("2hr_state", low_i, 1, -1)    # charge
    apply_window("2hr_state", high_i, 1, 1)    # discharge

    # 4-hour storage (±2 hour)
    apply_window("4hr_state", low_i, 2, -1)
    apply_window("4hr_state", high_i, 2, 1)

    # ---------- CREATE INTERSECTING CURVES ----------
    band2 = 8    # visual width
    band4 = 14

    node_df["2hr_curve"] = node_df["LMP"] + node_df["2hr_state"] * band2
    node_df["4hr_curve"] = node_df["LMP"] + node_df["4hr_state"] * band4

    # ---------- CALCULATE REVENUE ----------
    rev2 = round((node_df["2hr_state"] * node_df["LMP"]).sum(), 2)
    rev4 = round((node_df["4hr_state"] * node_df["LMP"]).sum(), 2)

    spread = node_df["LMP"].max() - node_df["LMP"].min()

    if spread > 80:
        strategy = "Pure Merchant Arbitrage Opportunity"
    elif spread > 40:
        strategy = "Solar + Storage Overbuild Recommended"
    else:
        strategy = "Low Spread → Capacity / Ancillary Market Focus"

    # ---------- PLOT ----------
    fig = go.Figure()

    # LMP price
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["LMP"],
        name="LMP PRICE",
        line=dict(color="deepskyblue", width=3)
    ))

    # 2hr storage band
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["2hr_curve"],
        name="2H STORAGE",
        line=dict(color="limegreen", width=3, shape="hv")
    ))

    # 4hr storage band
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["4hr_curve"],
        name="4H STORAGE",
        line=dict(color="orange", width=3, shape="hv")
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Hour Ending",
        yaxis_title="Price ($/MWh)",
        legend=dict(orientation="h", y=1.05, x=0.3)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- RESULTS ----------
    st.subheader("Market Signal")
    st.write(f"Lowest Price Hour: {low_hr}")
    st.write(f"Highest Price Hour: {high_hr}")

    st.subheader("Daily Arbitrage Revenue ($/MW-day)")
    st.write(f"2-hour Storage: ${rev2}")
    st.write(f"4-hour Storage: ${rev4}")

    st.subheader("Recommended Development Strategy")
    st.success(strategy)
