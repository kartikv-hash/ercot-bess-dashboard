import streamlit as st
import pandas as pd
import plotly.express as px

st.title("ERCOT Battery Arbitrage Dashboard")

uploaded_file = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns=["Date","Hour","Node","LMP","DST"]

    node = st.selectbox("Select Node", df["Node"].unique())

    node_df = df[df["Node"]==node].sort_values("Hour")

    fig = px.line(node_df, x="Hour", y="LMP", title=f"LMP Price - {node}")
    st.plotly_chart(fig,use_container_width=True)

    low=node_df["LMP"].idxmin()
    high=node_df["LMP"].idxmax()

    st.write("Lowest price hour:", node_df.loc[low,"Hour"])
    st.write("Highest price hour:", node_df.loc[high,"Hour"])
