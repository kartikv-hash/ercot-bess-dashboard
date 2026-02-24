    # -------- FIND LOW & HIGH PRICE --------
    low_i = node_df["LMP"].idxmin()
    high_i = node_df["LMP"].idxmax()

    low_hr = node_df.loc[low_i,"Hour"]
    high_hr = node_df.loc[high_i,"Hour"]

    # -------- CREATE STATE SIGNALS --------
    node_df["2hr_state"] = 0
    node_df["4hr_state"] = 0

    def apply_window(col, center, hrs, value):
        for i in range(max(0, center-hrs), min(len(node_df), center+hrs+1)):
            node_df.loc[i, col] = value

    # 2-hr battery
    apply_window("2hr_state", low_i, 1, -1)   # charge
    apply_window("2hr_state", high_i, 1, 1)   # discharge

    # 4-hr battery
    apply_window("4hr_state", low_i, 2, -1)
    apply_window("4hr_state", high_i, 2, 1)

    # -------- REVENUE --------
    rev2 = round((node_df["2hr_state"] * node_df["LMP"]).sum(),2)
    rev4 = round((node_df["4hr_state"] * node_df["LMP"]).sum(),2)

    # -------- PLOT --------
    fig = go.Figure()

    # LMP price line
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["LMP"],
        name="LMP PRICE",
        line=dict(color="deepskyblue", width=3)
    ))

    # 2hr storage state
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["2hr_state"]*20,   # scaled for visibility
        name="2H STORAGE",
        line=dict(color="limegreen", width=3, shape="hv")
    ))

    # 4hr storage state
    fig.add_trace(go.Scatter(
        x=node_df["Hour"],
        y=node_df["4hr_state"]*45,
        name="4H STORAGE",
        line=dict(color="orange", width=3, shape="hv")
    ))

    fig.update_layout(
        yaxis_title="Price / Operation",
        xaxis_title="Hour Ending",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- OUTPUT --------
    st.subheader("Market Signals")
    st.write(f"Lowest Price Hour: {low_hr}")
    st.write(f"Highest Price Hour: {high_hr}")

    st.subheader("Daily Arbitrage Revenue ($/MW-day)")
    st.write(f"2hr Storage: ${rev2}")
    st.write(f"4hr Storage: ${rev4}")
