# find min & max
low=node_df["LMP"].idxmin()
high=node_df["LMP"].idxmax()

low_i=node_df.index.get_loc(low)
high_i=node_df.index.get_loc(high)

node_df["2hr"]="Idle"
node_df["4hr"]="Idle"

def apply_window(col,center,hrs,label):
    for i in range(max(0,center-hrs),min(len(node_df),center+hrs+1)):
        node_df.loc[node_df.index[i],col]=label

# -------- 2 HR STORAGE --------
apply_window("2hr",low_i,1,"Charge")
apply_window("2hr",high_i,1,"Discharge")

# -------- 4 HR STORAGE --------
apply_window("4hr",low_i,2,"Charge")
apply_window("4hr",high_i,2,"Discharge")

# -------- PLOT --------
fig = px.line(node_df, x="Hour", y="LMP", title=f"LMP Price - {node}")

# 2hr markers
fig.add_scatter(x=node_df[node_df["2hr"]=="Charge"]["Hour"],
                y=node_df[node_df["2hr"]=="Charge"]["LMP"],
                mode="markers",
                marker=dict(color="green",size=9),
                name="2hr Charge")

fig.add_scatter(x=node_df[node_df["2hr"]=="Discharge"]["Hour"],
                y=node_df[node_df["2hr"]=="Discharge"]["LMP"],
                mode="markers",
                marker=dict(color="red",size=9),
                name="2hr Discharge")

# 4hr markers
fig.add_scatter(x=node_df[node_df["4hr"]=="Charge"]["Hour"],
                y=node_df[node_df["4hr"]=="Charge"]["LMP"],
                mode="markers",
                marker=dict(color="blue",size=11,symbol="diamond"),
                name="4hr Charge")

fig.add_scatter(x=node_df[node_df["4hr"]=="Discharge"]["Hour"],
                y=node_df[node_df["4hr"]=="Discharge"]["LMP"],
                mode="markers",
                marker=dict(color="orange",size=11,symbol="diamond"),
                name="4hr Discharge")

st.plotly_chart(fig,use_container_width=True)
