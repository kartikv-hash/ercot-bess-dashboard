import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ERCOT BESS Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark cards + accent colours)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Sidebar nav buttons */
  div[data-testid="stRadio"] > label { font-size: 15px; }

  /* Metric cards */
  .card {
      background: #1e2130;
      border: 1px solid #2e3250;
      border-radius: 12px;
      padding: 18px 22px;
      margin-bottom: 8px;
  }
  .card-title { color: #8b92b8; font-size: 12px; text-transform: uppercase; letter-spacing: .8px; }
  .card-value { color: #ffffff; font-size: 26px; font-weight: 700; margin-top: 4px; }
  .card-sub   { color: #5de0a5; font-size: 12px; margin-top: 2px; }

  /* Landing page nav cards */
  .nav-card {
      background: linear-gradient(135deg, #1e2130, #252a40);
      border: 1px solid #2e3250;
      border-radius: 16px;
      padding: 28px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s;
  }
  .nav-card:hover { border-color: #4f6ef7; }
  .nav-icon  { font-size: 40px; margin-bottom: 10px; }
  .nav-label { color: #fff; font-size: 17px; font-weight: 600; margin-bottom: 6px; }
  .nav-desc  { color: #8b92b8; font-size: 13px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR – LOGO + NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ERCOT BESS")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠  Home", "🗺️  Node Analysis", "📈  LMP Price Analysis", "🔋  BESS Integration Strategy"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"], label_visibility="collapsed")
    st.caption("CSV columns expected:\nDate, Hour, Node, LMP, DST")

# ─────────────────────────────────────────────
#  LOAD & CACHE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(f):
    df = pd.read_csv(f)
    df.columns = ["Date", "Hour", "Node", "LMP", "DST"]
    return df

df = load_data(uploaded_file) if uploaded_file else None

# ─────────────────────────────────────────────
#  HELPER – metric card
# ─────────────────────────────────────────────
def metric_card(title, value, sub=""):
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
#  PAGE: HOME
# ═════════════════════════════════════════════
if page == "🏠  Home":
    st.title("⚡ ERCOT Battery Storage Arbitrage Dashboard")
    st.markdown("Select a module below or use the sidebar to navigate.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="nav-card">
          <div class="nav-icon">🗺️</div>
          <div class="nav-label">Node Analysis</div>
          <div class="nav-desc">Compare LMP statistics across all settlement point nodes. Filter, rank, and explore spread opportunities.</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="nav-card">
          <div class="nav-icon">📈</div>
          <div class="nav-label">LMP Price Analysis</div>
          <div class="nav-desc">Visualise intraday LMP curves, charging & discharging windows, and daily arbitrage revenue per node.</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="nav-card">
          <div class="nav-icon">🔋</div>
          <div class="nav-label">BESS Integration Strategy</div>
          <div class="nav-desc">Evaluate Overbuild vs Augmentation strategies based on LMP spread, battery duration, and project economics.</div>
        </div>""", unsafe_allow_html=True)

    if df is None:
        st.info("👈  Upload a CSV from the sidebar to unlock all modules.", icon="📂")
    else:
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")
        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card("Total Rows", f"{len(df):,}")
        with k2: metric_card("Nodes", f"{df['Node'].nunique()}", "unique settlement points")
        with k3: metric_card("Avg LMP", f"${df['LMP'].mean():.2f}", "$/MWh across all nodes")
        with k4: metric_card("Max LMP", f"${df['LMP'].max():.2f}", "peak price in dataset")

# ═════════════════════════════════════════════
#  PAGE: NODE ANALYSIS
# ═════════════════════════════════════════════
elif page == "🗺️  Node Analysis":
    st.title("🗺️ Node Analysis")

    if df is None:
        st.warning("Please upload an ERCOT LMP CSV from the sidebar to continue.")
        st.stop()

    # ── Summary table across all nodes ──────
    st.subheader("All-Node Summary")
    node_stats = (
        df.groupby("Node")["LMP"]
        .agg(
            Avg_LMP="mean",
            Min_LMP="min",
            Max_LMP="max",
            Spread=lambda x: x.max() - x.min(),
            Std_Dev="std"
        )
        .round(2)
        .reset_index()
        .sort_values("Spread", ascending=False)
    )

    # Colour-code the spread column
    def colour_spread(val):
        if val > 80:  return "background-color: #1f4b2e; color: #5de0a5"
        if val > 40:  return "background-color: #3d3510; color: #f0c040"
        return "background-color: #2e1a1a; color: #f07070"

    st.dataframe(
        node_stats.style.applymap(colour_spread, subset=["Spread"]),
        use_container_width=True,
        height=280
    )

    st.markdown("---")

    # ── Node comparison bar chart ────────────
    st.subheader("Spread Comparison Across Nodes")
    fig_bar = px.bar(
        node_stats,
        x="Node", y="Spread",
        color="Spread",
        color_continuous_scale=["#f07070", "#f0c040", "#5de0a5"],
        labels={"Spread": "LMP Spread ($/MWh)"},
        template="plotly_dark"
    )
    fig_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Single node deep-dive ────────────────
    st.subheader("Node Deep-Dive")
    sel_node = st.selectbox("Select Node", df["Node"].unique())
    ndf = df[df["Node"] == sel_node].sort_values("Hour").reset_index(drop=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1: metric_card("Avg LMP",  f"${ndf['LMP'].mean():.2f}")
    with d2: metric_card("Min LMP",  f"${ndf['LMP'].min():.2f}", f"Hour {ndf.loc[ndf['LMP'].idxmin(),'Hour']}")
    with d3: metric_card("Max LMP",  f"${ndf['LMP'].max():.2f}", f"Hour {ndf.loc[ndf['LMP'].idxmax(),'Hour']}")
    with d4:
        sp = ndf['LMP'].max() - ndf['LMP'].min()
        tag = "High ✅" if sp > 80 else ("Mid ⚠️" if sp > 40 else "Low ❌")
        metric_card("Spread", f"${sp:.2f}", tag)

    fig_node = go.Figure()
    fig_node.add_trace(go.Scatter(
        x=ndf["Hour"], y=ndf["LMP"],
        fill="tozeroy", fillcolor="rgba(93,224,165,0.1)",
        line=dict(color="#5de0a5", width=2),
        name="LMP"
    ))
    fig_node.update_layout(
        template="plotly_dark",
        xaxis_title="Hour Ending",
        yaxis_title="Price ($/MWh)",
        title=f"Intraday LMP — {sel_node}"
    )
    st.plotly_chart(fig_node, use_container_width=True)

# ═════════════════════════════════════════════
#  PAGE: LMP PRICE ANALYSIS  (original logic)
# ═════════════════════════════════════════════
elif page == "📈  LMP Price Analysis":
    st.title("📈 LMP Price Analysis")

    if df is None:
        st.warning("Please upload an ERCOT LMP CSV from the sidebar to continue.")
        st.stop()

    node = st.selectbox("Select Node", df["Node"].unique())
    node_df = df[df["Node"] == node].sort_values("Hour").reset_index(drop=True)

    # ── Find low / high price hours ──────────
    low_i  = node_df["LMP"].idxmin()
    high_i = node_df["LMP"].idxmax()
    low_hr  = node_df.loc[low_i, "Hour"]
    high_hr = node_df.loc[high_i, "Hour"]

    # ── Storage state windows ────────────────
    node_df["2hr_state"] = 0
    node_df["4hr_state"] = 0

    def apply_window(col, center, hrs, value):
        for i in range(max(0, center - hrs), min(len(node_df), center + hrs + 1)):
            node_df.loc[i, col] = value

    apply_window("2hr_state", low_i,  1, -1)
    apply_window("2hr_state", high_i, 1,  1)
    apply_window("4hr_state", low_i,  2, -1)
    apply_window("4hr_state", high_i, 2,  1)

    band2, band4 = 8, 14
    node_df["2hr_curve"] = node_df["LMP"] + node_df["2hr_state"] * band2
    node_df["4hr_curve"] = node_df["LMP"] + node_df["4hr_state"] * band4

    # ── Revenue ──────────────────────────────
    rev2 = round((node_df["2hr_state"] * node_df["LMP"]).sum(), 2)
    rev4 = round((node_df["4hr_state"] * node_df["LMP"]).sum(), 2)
    spread = node_df["LMP"].max() - node_df["LMP"].min()

    if spread > 80:
        strategy = "Pure Merchant Arbitrage Opportunity"
        strat_colour = "success"
    elif spread > 40:
        strategy = "Solar + Storage Overbuild Recommended"
        strat_colour = "warning"
    else:
        strategy = "Low Spread → Capacity / Ancillary Market Focus"
        strat_colour = "error"

    # ── Metrics row ──────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Lowest Price Hour",  str(low_hr))
    with m2: metric_card("Highest Price Hour", str(high_hr))
    with m3: metric_card("2H Revenue",  f"${rev2}", "$/MW-day")
    with m4: metric_card("4H Revenue",  f"${rev4}", "$/MW-day")

    # ── Chart ────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=node_df["Hour"], y=node_df["LMP"],
        name="LMP Price", line=dict(color="deepskyblue", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=node_df["Hour"], y=node_df["2hr_curve"],
        name="2H Storage", line=dict(color="limegreen", width=3, shape="hv")
    ))
    fig.add_trace(go.Scatter(
        x=node_df["Hour"], y=node_df["4hr_curve"],
        name="4H Storage", line=dict(color="orange", width=3, shape="hv")
    ))
    # Shade charge window
    fig.add_vrect(
        x0=max(0, low_hr - 2), x1=min(24, low_hr + 2),
        fillcolor="limegreen", opacity=0.06,
        annotation_text="Charge", annotation_position="top left"
    )
    fig.add_vrect(
        x0=max(0, high_hr - 2), x1=min(24, high_hr + 2),
        fillcolor="orange", opacity=0.06,
        annotation_text="Discharge", annotation_position="top right"
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Hour Ending",
        yaxis_title="Price ($/MWh)",
        legend=dict(orientation="h", y=1.05, x=0.3),
        title=f"LMP & Storage Curves — {node}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommended Development Strategy")
    if strat_colour == "success":
        st.success(f"✅  {strategy}")
    elif strat_colour == "warning":
        st.warning(f"⚠️  {strategy}")
    else:
        st.error(f"❌  {strategy}")

# ═════════════════════════════════════════════
#  PAGE: BESS INTEGRATION STRATEGY
# ═════════════════════════════════════════════
elif page == "🔋  BESS Integration Strategy":
    st.title("🔋 BESS Integration Strategy")
    st.markdown("Evaluate **Overbuild** vs **Augmentation** pathways based on LMP spread economics and battery project lifecycle.")

    if df is None:
        st.warning("Please upload an ERCOT LMP CSV from the sidebar to continue.")
        st.stop()

    # ── Inputs ───────────────────────────────
    st.markdown("### ⚙️ Project Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        project_mw   = st.number_input("Project Size (MW)", 10, 500, 100, step=10)
        duration_hr  = st.selectbox("Storage Duration (hrs)", [2, 4, 6, 8], index=1)
    with col2:
        capex_ob     = st.number_input("Overbuild CAPEX ($/kWh)", 100, 400, 220, step=10)
        capex_aug    = st.number_input("Augmentation CAPEX ($/kWh)", 80, 300, 160, step=10)
    with col3:
        deg_rate     = st.slider("Annual Degradation (%)", 1.0, 4.0, 2.0, 0.1)
        project_life = st.slider("Project Life (years)", 10, 25, 15)

    st.markdown("---")

    # ── Derive spread from uploaded data ─────
    node_spreads = (
        df.groupby("Node")["LMP"]
        .agg(spread=lambda x: x.max() - x.min())
        .reset_index()
    )
    avg_spread = node_spreads["spread"].mean()
    sel_node_bess = st.selectbox("Reference Node for Revenue Model", df["Node"].unique())
    ndf_b = df[df["Node"] == sel_node_bess]
    spread_b = ndf_b["LMP"].max() - ndf_b["LMP"].min()
    daily_rev_mw = spread_b * duration_hr          # simplified $/MW-day
    annual_rev   = daily_rev_mw * 365 * project_mw # $ / year

    # ── Build year-by-year projection ────────
    years = list(range(1, project_life + 1))

    # Overbuild: starts larger, degrades slower because oversized
    ob_capacity  = [project_mw * duration_hr * (1 - (deg_rate * 0.6 / 100)) ** y for y in years]
    ob_revenue   = [annual_rev * (1 - (deg_rate * 0.6 / 100)) ** y for y in years]
    ob_capex_tot = capex_ob * project_mw * duration_hr * 1000  # $
    ob_cum_rev   = np.cumsum(ob_revenue)
    ob_npv       = [r - ob_capex_tot for r in ob_cum_rev]

    # Augmentation: lower upfront, augmented at midlife, sharper early deg
    aug_capacity = []
    aug_revenue  = []
    midpoint = project_life // 2
    for y in years:
        if y <= midpoint:
            cap = project_mw * duration_hr * (1 - (deg_rate / 100)) ** y
        else:                              # post-augmentation boost
            cap = project_mw * duration_hr * (1 - (deg_rate / 100)) ** (y - midpoint) * 1.05
        aug_capacity.append(cap)
        aug_revenue.append(annual_rev * (cap / (project_mw * duration_hr)))

    aug_capex_tot = capex_aug * project_mw * duration_hr * 1000
    aug_extra_cap = aug_capex_tot * 0.35   # mid-life augmentation cost
    aug_cum_rev   = np.cumsum(aug_revenue)
    aug_npv       = [r - aug_capex_tot - (aug_extra_cap if i + 1 > midpoint else 0)
                     for i, r in enumerate(aug_cum_rev)]

    # ── KPI Row ──────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Node Spread",    f"${spread_b:.1f}", f"$/MWh — {sel_node_bess}")
    with k2: metric_card("Est. Annual Rev", f"${annual_rev/1e6:.2f}M", f"{project_mw} MW · {duration_hr}h")
    with k3: metric_card("Overbuild CAPEX", f"${ob_capex_tot/1e6:.1f}M", f"${capex_ob}/kWh")
    with k4: metric_card("Augment CAPEX",  f"${aug_capex_tot/1e6:.1f}M", f"${capex_aug}/kWh")

    st.markdown("---")

    # ── Chart 1 – Capacity over time ─────────
    st.subheader("📉 Usable Capacity Over Project Life")
    fig_cap = go.Figure()
    fig_cap.add_trace(go.Scatter(
        x=years, y=ob_capacity,
        name="Overbuild", line=dict(color="#7b68ee", width=3)
    ))
    fig_cap.add_trace(go.Scatter(
        x=years, y=aug_capacity,
        name="Augmentation", line=dict(color="#ff8c42", width=3, dash="dash")
    ))
    fig_cap.add_vline(
        x=midpoint, line_dash="dot", line_color="gray",
        annotation_text="Augmentation Event", annotation_position="top right"
    )
    fig_cap.update_layout(
        template="plotly_dark",
        xaxis_title="Project Year",
        yaxis_title="Usable Capacity (MWh)",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_cap, use_container_width=True)

    # ── Chart 2 – Cumulative NPV ──────────────
    st.subheader("💰 Cumulative NPV Comparison")
    fig_npv = go.Figure()
    fig_npv.add_trace(go.Scatter(
        x=years, y=[v / 1e6 for v in ob_npv],
        name="Overbuild NPV", fill="tozeroy",
        fillcolor="rgba(123,104,238,0.1)",
        line=dict(color="#7b68ee", width=3)
    ))
    fig_npv.add_trace(go.Scatter(
        x=years, y=[v / 1e6 for v in aug_npv],
        name="Augmentation NPV", fill="tozeroy",
        fillcolor="rgba(255,140,66,0.1)",
        line=dict(color="#ff8c42", width=3, dash="dash")
    ))
    fig_npv.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig_npv.update_layout(
        template="plotly_dark",
        xaxis_title="Project Year",
        yaxis_title="Cumulative NPV ($M)",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_npv, use_container_width=True)

    # ── Chart 3 – Annual Revenue Bars ─────────
    st.subheader("📊 Annual Revenue by Strategy")
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        x=years, y=[v / 1e6 for v in ob_revenue],
        name="Overbuild", marker_color="#7b68ee", opacity=0.85
    ))
    fig_rev.add_trace(go.Bar(
        x=years, y=[v / 1e6 for v in aug_revenue],
        name="Augmentation", marker_color="#ff8c42", opacity=0.85
    ))
    fig_rev.update_layout(
        template="plotly_dark",
        barmode="group",
        xaxis_title="Project Year",
        yaxis_title="Annual Revenue ($M)",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # ── Strategy Recommendation ───────────────
    st.markdown("---")
    st.subheader("📋 Strategy Recommendation")

    ob_final_npv  = ob_npv[-1]  / 1e6
    aug_final_npv = aug_npv[-1] / 1e6

    if spread_b > 80:
        rec = ("**Pure Overbuild** recommended. High LMP spread justifies premium CAPEX — the larger "
               "nameplate capacity maximises arbitrage capture throughout the project life.")
        st.success(f"✅  {rec}")
    elif spread_b > 40:
        rec = ("**Augmentation** strategy offers better capital efficiency. Moderate spreads make "
               "mid-life augmentation the preferred path, preserving IRR while managing upfront cost.")
        st.warning(f"⚠️  {rec}")
    else:
        rec = ("**Low spread environment** — neither pure arbitrage strategy is compelling. "
               "Consider capacity markets, ancillary services (ECRS/RRS), or paired solar offtake.")
        st.error(f"❌  {rec}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**Overbuild {project_life}-yr NPV:** ${ob_final_npv:.2f}M")
    with col_b:
        st.info(f"**Augmentation {project_life}-yr NPV:** ${aug_final_npv:.2f}M")
