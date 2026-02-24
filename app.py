import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("ERCOT Battery Storage Analysis Platform")

uploaded_file = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"])

if uploaded_file:

    # ---------------- READ DATA ----------------
    df = pd.read_csv(uploaded_file)
    df.columns = ["Date","Hour","Node","LMP","DST"]

    node = st.selectbox("Select Node", df["Node"].unique())
    node_df = df[df["Node"]==node].sort_values("Hour").reset_index(drop=True)

    # ---------------- FIND LOW & HIGH ----------------
    low_i = node_df["LMP"].idxmin()
    high_i = node_df["LMP"].idxmax()

    low_hr = node_df.loc[low_i,"Hour"]
    high_hr = node_df.loc[high_i,"Hour"]

    # ---------------- CREATE DISPATCH STATES ----------------
    node_df["2hr_state"] = 0
    node_df["4hr_state"] = 0

    def apply_window(col, center, hrs, value):
        for i in range(max(0, center-hrs), min(len(node_df), center+hrs+1)):
            node_df.loc[i, col] = value

    # 2-hour storage
    apply_window("2hr_state", low_i, 1, -1)   # charge
    apply_window("2hr_state", high_i, 1, 1)   # discharge

    # 4-hour storage
    apply_window("4hr_state", low_i, 2, -1)
    apply_window("4hr_state", high_i, 2, 1)

    # ---------------- REVENUE ----------------
    rev2 = round((node_df["2hr_state"] * node_df["LMP"]).sum(),2)
    rev4 = round((node_df["4hr_state"] * node_df["LMP"]).sum(),2)

    # ---------------- PLOT ----------------
    fig = go.Figure()

    # LMP line
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["LMP"],
        name="LMP PRICE",
        line=dict(color="deepskyblue", width=3)
    ))

    # 2hr state
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["2hr_state"]*20,
        name="2H STORAGE",
        line=dict(color="limegreen", width=3, shape="hv")
    ))

    # 4hr state
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["4hr_state"]*45,
        name="4H STORAGE",
        line=dict(color="orange", width=3, shape="hv")
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Hour Ending",
        yaxis_title="Price / Operation Level"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RESULTS ----------------
    st.subheader("Market Signals")
    st.write(f"Lowest Price Hour: {low_hr}")
    st.write(f"Highest Price Hour: {high_hr}")

    st.subheader("Daily Arbitrage Revenue ($/MW-day)")
    st.write(f"2hr Storage: ${rev2}")
    st.write(f"4hr Storage: ${rev4}")
