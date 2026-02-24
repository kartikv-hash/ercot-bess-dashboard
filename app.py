import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("ERCOT Battery Storage Analysis Platform")

uploaded_file = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = ["Date","Hour","Node","LMP","DST"]

    node = st.selectbox("Select Node", df["Node"].unique())
    node_df = df[df["Node"]==node].sort_values("Hour").reset_index(drop=True)

    # ---------- FIND LOW & HIGH ----------
    low_i = node_df["LMP"].idxmin()
    high_i = node_df["LMP"].idxmax()

    low_hr = node_df.loc[low_i,"Hour"]
    high_hr = node_df.loc[high_i,"Hour"]

    # ---------- CREATE DISPATCH ----------
    node_df["2hr_dispatch"] = 0
    node_df["4hr_dispatch"] = 0

    def apply(center,hrs,col,charge=-1,discharge=1):
        for i in range(max(0,center-hrs),min(len(node_df),center+hrs+1)):
            node_df.loc[i,col]=charge
        for i in range(max(0,high_i-hrs),min(len(node_df),high_i+hrs+1)):
            node_df.loc[i,col]=discharge

    apply(low_i,1,"2hr_dispatch")
    apply(low_i,2,"4hr_dispatch")

    # ---------- REVENUE ----------
    node_df["2hr_revenue"]=node_df["2hr_dispatch"]*node_df["LMP"]
    node_df["4hr_revenue"]=node_df["4hr_dispatch"]*node_df["LMP"]

    rev2=round(node_df["2hr_revenue"].sum(),2)
    rev4=round(node_df["4hr_revenue"].sum(),2)

    # ---------- STRATEGY LOGIC ----------
    spread = node_df["LMP"].max()-node_df["LMP"].min()

    if spread > 80:
        strategy="Pure Arbitrage Opportunity"
    elif spread > 40:
        strategy="Arbitrage + Solar Overbuild Synergy"
    else:
        strategy="Limited Arbitrage — Capacity / Ancillary Market Preferred"

    # ---------- PLOT ----------
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=node_df["Hour"],y=node_df["LMP"],name="LMP"))

    fig.add_trace(go.Scatter(x=node_df["Hour"],y=node_df["2hr_dispatch"]*node_df["LMP"],
                             name="2hr Storage Operation"))

    fig.add_trace(go.Scatter(x=node_df["Hour"],y=node_df["4hr_dispatch"]*node_df["LMP"],
                             name="4hr Storage Operation"))

    st.plotly_chart(fig,use_container_width=True)

    # ---------- OUTPUT ----------
    st.subheader("Market Signals")
    st.write(f"Lowest Price Hour: {low_hr}")
    st.write(f"Highest Price Hour: {high_hr}")

    st.subheader("Daily Arbitrage Revenue ($/MW)")
    st.write(f"2hr Storage: ${rev2}")
    st.write(f"4hr Storage: ${rev4}")

    st.subheader("Recommended Strategy")
    st.success(strategy)
