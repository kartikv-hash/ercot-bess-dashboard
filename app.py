
iimport streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
import math

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ERCOT BESS Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Metric cards */
  .card {
      background: #1a1d2e;
      border: 1px solid #2a2d45;
      border-radius: 12px;
      padding: 16px 20px;
  }
  .card-title { color: #7880a8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .card-value { color: #ffffff; font-size: 24px; font-weight: 700; margin-top: 4px; }
  .card-sub   { color: #5de0a5; font-size: 11px; margin-top: 3px; }

  /* Hub / Node badge */
  .badge-hub  { background:#3b1f5e; color:#bf7fff; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; }
  .badge-node { background:#1a3550; color:#4fc3f7; padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; }

  /* Section divider */
  .section-header {
      font-size: 13px; font-weight: 600; color: #7880a8;
      text-transform: uppercase; letter-spacing: 1px;
      margin: 18px 0 10px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR – NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ERCOT BESS")
    st.markdown("---")
    page = st.radio(
        "Module",
        ["🗺️  Node Analyser", "📈  LMP Price Analysis"],
        label_visibility="collapsed"
    )

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def metric_card(title, value, sub=""):
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def haversine_miles(lat1, lon1, lat2, lon2):
    """Return distance in miles between two lat/lon points."""
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 2)


# kV label → nominal volts
KV_MAP = {
    "34.5 kV":  34500,
    "69 kV":    69000,
    "115 kV":  115000,
    "138 kV":  138000,
    "230 kV":  230000,
    "345 kV":  345000,
    "500 kV":  500000,
    "765 kV":  765000,
}

def nearest_kv_label(volts):
    """Snap raw OSM voltage (int) to nearest standard kV label."""
    best, best_diff = None, float("inf")
    for label, nominal in KV_MAP.items():
        diff = abs(volts - nominal) / nominal
        if diff < best_diff:
            best, best_diff = label, diff
    return best if best_diff < 0.20 else f"{volts/1000:.1f} kV"


# ══════════════════════════════════════════════
#  PAGE 1 – NODE ANALYSER
# ══════════════════════════════════════════════
if page == "🗺️  Node Analyser":

    st.title("🗺️  Hub & Node Analyser")
    st.caption(
        "Discovers transmission substations from OpenStreetMap within your chosen radius. "
        "Hubs = 230 kV and above  |  Nodes = below 230 kV"
    )

    # ── Input Panel ─────────────────────────
    st.markdown('<div class="section-header">Search Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 3])

    with c1:
        lat = st.number_input("Latitude",  value=33.7944, format="%.4f",
                              help="Decimal degrees, positive = North")
        lon = st.number_input("Longitude", value=-98.5706, format="%.4f",
                              help="Decimal degrees, negative = West")

    with c2:
        radius_miles = st.selectbox("Search Radius (miles)", [5, 10, 25, 50, 100, 150], index=2)
        hub_threshold_kv = st.selectbox(
            "Hub threshold (kV ≥)",
            [115, 138, 230, 345],
            index=2,
            help="Substations at or above this voltage are classified as Hubs"
        )

    with c3:
        selected_kv_labels = st.multiselect(
            "Filter Voltages",
            list(KV_MAP.keys()),
            default=list(KV_MAP.keys()),
            help="Show only substations matching these voltage levels"
        )
        show_unknown_v = st.checkbox("Include substations with unknown voltage", value=True)

    run_search = st.button("🔍  Search Substations", type="primary", use_container_width=False)

    # ── OpenInfraMap quick link ──────────────
    zoom_guess = max(7, min(14, round(14 - math.log2(max(1, radius_miles)))))
    infra_url = f"https://openinframap.org/#{zoom_guess}/{lat:.4f}/{lon:.4f}"
    st.markdown(
        f'🔗 **[Open this area in OpenInfraMap]({infra_url})**  '
        f'<span style="color:#7880a8;font-size:12px;">(opens in new tab)</span>',
        unsafe_allow_html=True
    )

    if run_search:
        radius_m = radius_miles * 1609.34
        hub_threshold_v = hub_threshold_kv * 1000
        selected_nominal_volts = {KV_MAP[k] for k in selected_kv_labels}

        overpass_query = f"""
[out:json][timeout:40];
(
  node["power"="substation"](around:{radius_m:.0f},{lat},{lon});
  way["power"="substation"](around:{radius_m:.0f},{lat},{lon});
  relation["power"="substation"](around:{radius_m:.0f},{lat},{lon});
);
out center tags;
"""
        with st.spinner("Querying OpenStreetMap Overpass API …"):
            try:
                resp = requests.post(
                    "https://overpass-api.de/api/interpreter",
                    data={"data": overpass_query},
                    timeout=45
                )
                resp.raise_for_status()
                osm_data = resp.json()
            except requests.exceptions.Timeout:
                st.error("⏱  Overpass API timed out. Try a smaller radius or try again shortly.")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        rows = []
        for el in osm_data.get("elements", []):
            tags = el.get("tags", {})

            # ── Coordinates ──
            if el["type"] == "node":
                slat, slon = el.get("lat"), el.get("lon")
            elif "center" in el:
                slat, slon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            if slat is None or slon is None:
                continue

            # ── Voltage parsing ──
            volt_raw = tags.get("voltage", "")
            volts = None
            if volt_raw:
                try:
                    # Some entries: "115000;230000" — take highest
                    volts = max(int(v) for v in volt_raw.split(";") if v.strip().isdigit())
                except Exception:
                    pass

            # ── Voltage filter ──
            if volts is not None:
                nearest = nearest_kv_label(volts)
                nom = KV_MAP.get(nearest)
                if nom and nom not in selected_nominal_volts:
                    continue          # filtered out by user
                kv_label = nearest
            else:
                if not show_unknown_v:
                    continue
                kv_label = "Unknown"

            # ── Hub or Node ──
            sub_type = tags.get("substation", "")
            if sub_type == "transmission" or (volts is not None and volts >= hub_threshold_v):
                classification = "Hub"
            else:
                classification = "Node"

            rows.append({
                "Name":          tags.get("name", f"Substation {el['id']}"),
                "Type":          classification,
                "Voltage":       kv_label,
                "Operator":      tags.get("operator", "—"),
                "Substation Tag": sub_type if sub_type else "—",
                "Lat":           round(slat, 5),
                "Lon":           round(slon, 5),
                "Distance (mi)": haversine_miles(lat, lon, slat, slon),
                "_id":           el["id"],
            })

        if not rows:
            st.warning("No substations found. Try a larger radius or loosen the voltage filters.")
            st.stop()

        sdf = pd.DataFrame(rows).sort_values("Distance (mi)").reset_index(drop=True)
        n_hub  = (sdf["Type"] == "Hub").sum()
        n_node = (sdf["Type"] == "Node").sum()

        # ── KPI row ─────────────────────────
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: metric_card("Total Found",      str(len(sdf)))
        with k2: metric_card("Hubs",             str(n_hub),  f"≥ {hub_threshold_kv} kV")
        with k3: metric_card("Nodes",            str(n_node), f"< {hub_threshold_kv} kV")
        with k4: metric_card("Radius",           f"{radius_miles} mi")
        with k5: metric_card("Centre",           f"{lat:.3f}, {lon:.3f}")

        st.markdown("")

        # ── Map ─────────────────────────────
        fig_map = go.Figure()

        # Search-centre star
        fig_map.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode="markers+text",
            marker=dict(size=16, color="#FFD700"),
            text=["📍 Centre"],
            textposition="top right",
            name="Search Centre",
            hoverinfo="text",
            hovertext=f"Centre: {lat:.4f}, {lon:.4f}"
        ))

        # Approximate radius circle (discrete points)
        circle_lats, circle_lons = [], []
        for deg in range(0, 361, 4):
            rad = math.radians(deg)
            dlat = (radius_miles / 3958.8) * math.cos(rad)
            dlon = (radius_miles / 3958.8) * math.sin(rad) / math.cos(math.radians(lat))
            circle_lats.append(lat + math.degrees(dlat))
            circle_lons.append(lon + math.degrees(dlon))
        fig_map.add_trace(go.Scattermapbox(
            lat=circle_lats, lon=circle_lons,
            mode="lines",
            line=dict(color="rgba(255,215,0,0.3)", width=1),
            name=f"{radius_miles} mi radius",
            hoverinfo="skip"
        ))

        # Hubs
        hubs_df = sdf[sdf["Type"] == "Hub"]
        if len(hubs_df):
            fig_map.add_trace(go.Scattermapbox(
                lat=hubs_df["Lat"], lon=hubs_df["Lon"],
                mode="markers",
                marker=dict(size=13, color="#bf7fff"),
                name="Hub",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Type: Hub<br>"
                    "Voltage: %{customdata[1]}<br>"
                    "Operator: %{customdata[2]}<br>"
                    "Distance: %{customdata[3]} mi<extra></extra>"
                ),
                customdata=hubs_df[["Name","Voltage","Operator","Distance (mi)"]].values
            ))

        # Nodes
        nodes_df = sdf[sdf["Type"] == "Node"]
        if len(nodes_df):
            fig_map.add_trace(go.Scattermapbox(
                lat=nodes_df["Lat"], lon=nodes_df["Lon"],
                mode="markers",
                marker=dict(size=9, color="#4fc3f7"),
                name="Node",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Type: Node<br>"
                    "Voltage: %{customdata[1]}<br>"
                    "Operator: %{customdata[2]}<br>"
                    "Distance: %{customdata[3]} mi<extra></extra>"
                ),
                customdata=nodes_df[["Name","Voltage","Operator","Distance (mi)"]].values
            ))

        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=lat, lon=lon),
                zoom=max(7, min(12, round(12 - math.log2(max(1, radius_miles)))))
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
            legend=dict(
                bgcolor="rgba(20,22,38,0.85)",
                font=dict(color="white"),
                x=0.01, y=0.99
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # ── Results table ────────────────────
        st.markdown('<div class="section-header">Substation List</div>', unsafe_allow_html=True)

        # Voltage filter for table
        v_filter = st.multiselect(
            "Filter table by voltage",
            options=sorted(sdf["Voltage"].unique()),
            default=sorted(sdf["Voltage"].unique()),
            key="table_v_filter"
        )
        display_df = sdf[sdf["Voltage"].isin(v_filter)].drop(columns=["_id"])

        def _style_type(val):
            if val == "Hub":  return "background-color:#3b1f5e;color:#bf7fff;font-weight:600"
            return "background-color:#1a3550;color:#4fc3f7;font-weight:600"

        st.dataframe(
            display_df.style.applymap(_style_type, subset=["Type"]),
            use_container_width=True,
            height=340
        )
        st.caption(f"Showing {len(display_df)} of {len(sdf)} substations")


# ══════════════════════════════════════════════
#  PAGE 2 – LMP PRICE ANALYSIS
# ══════════════════════════════════════════════
elif page == "📈  LMP Price Analysis":

    st.title("📈  LMP Price Analysis")
    st.caption("Upload an ERCOT LMP CSV, select a Bus, and visualise the intraday price curve with BESS strategy overlay.")

    # ── Upload ──────────────────────────────
    uploaded = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"])
    if not uploaded:
        st.info("👆  Upload an ERCOT LMP CSV to get started.")
        st.markdown("""
        **Expected columns:**
        `Date | Hour Ending | Bus Name | LMP | DST`
        *(or similar — first 5 columns are mapped automatically)*
        """)
        st.stop()

    @st.cache_data
    def load_lmp(f):
        raw = pd.read_csv(f)
        # Strip whitespace from column names
        raw.columns = [c.strip() for c in raw.columns]

        # Try to auto-detect columns by name first
        col_map = {}
        for c in raw.columns:
            cl = c.lower().strip()
            if "date" in cl and "Date" not in col_map:
                col_map["Date"] = c
            elif any(x in cl for x in ["hour", "he", "hour_ending"]) and "Hour" not in col_map:
                col_map["Hour"] = c
            elif any(x in cl for x in ["bus", "node", "settlement", "name"]) and "Bus" not in col_map:
                col_map["Bus"] = c
            elif any(x in cl for x in ["lmp", "price", "$/mwh"]) and "LMP" not in col_map:
                col_map["LMP"] = c

        # Fall back to positional mapping if detection incomplete
        pos_names = ["Date", "Hour", "Bus", "LMP", "DST"]
        for i, name in enumerate(pos_names):
            if name not in col_map and i < len(raw.columns):
                col_map[name] = raw.columns[i]

        rename = {v: k for k, v in col_map.items()}
        raw = raw.rename(columns=rename)

        # Ensure required columns exist
        for col in ["Date", "Hour", "Bus", "LMP"]:
            if col not in raw.columns:
                raw[col] = None

        raw["Hour"] = pd.to_numeric(raw["Hour"], errors="coerce")
        raw["LMP"]  = pd.to_numeric(raw["LMP"],  errors="coerce")
        raw["Date"] = raw["Date"].astype(str).str.strip()
        raw["Bus"]  = raw["Bus"].astype(str).str.strip()
        raw = raw.dropna(subset=["Hour", "LMP"]).reset_index(drop=True)
        return raw

    df = load_lmp(uploaded)

    # ── Bus selector ────────────────────────
    st.markdown('<div class="section-header">Bus Selection</div>', unsafe_allow_html=True)
    bus_list = sorted(df["Bus"].unique().tolist())
    bus = st.selectbox(
        "Search & select Bus name",
        bus_list,
        help="Type to search"
    )

    # ── Date selector (if multi-day data) ───
    dates = sorted(df["Date"].unique().tolist())
    if len(dates) > 1:
        selected_date = st.selectbox("Select Date", dates)
        bdf = df[(df["Bus"] == bus) & (df["Date"] == selected_date)].sort_values("Hour").reset_index(drop=True)
    elif len(dates) == 1:
        selected_date = dates[0]
        bdf = df[df["Bus"] == bus].sort_values("Hour").reset_index(drop=True)
    else:
        st.error("Could not read a Date column from your CSV. Check the column names match: Date, Hour, Bus Name, LMP, DST")
        st.stop()

    if bdf.empty:
        st.warning("No data for this Bus / Date combination.")
        st.stop()

    # ── BESS strategy toggles ────────────────
    st.markdown('<div class="section-header">BESS Strategy Overlay</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    show_2hr = bc1.checkbox("⚡ 2-Hour Storage", value=True)
    show_4hr = bc2.checkbox("⚡ 4-Hour Storage", value=True)

    # ── Compute charge/discharge windows ────
    low_i  = bdf["LMP"].idxmin()
    high_i = bdf["LMP"].idxmax()
    low_hr  = bdf.loc[low_i,  "Hour"]
    high_hr = bdf.loc[high_i, "Hour"]
    min_lmp = bdf["LMP"].min()
    max_lmp = bdf["LMP"].max()
    spread  = max_lmp - min_lmp

    def window(center_hr, half_width):
        return (max(bdf["Hour"].min(), center_hr - half_width),
                min(bdf["Hour"].max(), center_hr + half_width))

    charge_2hr_win    = window(low_hr,  1)
    discharge_2hr_win = window(high_hr, 1)
    charge_4hr_win    = window(low_hr,  2)
    discharge_4hr_win = window(high_hr, 2)

    # ── Revenue (simplified) ─────────────────
    def calc_revenue(bdf, charge_win, discharge_win):
        ch_mask  = (bdf["Hour"] >= charge_win[0])    & (bdf["Hour"] <= charge_win[1])
        dis_mask = (bdf["Hour"] >= discharge_win[0]) & (bdf["Hour"] <= discharge_win[1])
        charge_avg    = bdf.loc[ch_mask,  "LMP"].mean()
        discharge_avg = bdf.loc[dis_mask, "LMP"].mean()
        return round(discharge_avg - charge_avg, 2)

    rev2 = calc_revenue(bdf, charge_2hr_win, discharge_2hr_win)
    rev4 = calc_revenue(bdf, charge_4hr_win, discharge_4hr_win)

    # ── Strategy recommendation ──────────────
    if spread > 80:
        strategy_label = "✅  Pure Merchant Arbitrage Opportunity"
        strategy_fn = st.success
    elif spread > 40:
        strategy_label = "⚠️  Solar + Storage Overbuild Recommended"
        strategy_fn = st.warning
    else:
        strategy_label = "❌  Low Spread → Capacity / Ancillary Market Focus"
        strategy_fn = st.error

    # ── KPI Row ──────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("Bus",           bus)
    with k2: metric_card("Date",          str(selected_date))
    with k3: metric_card("Lowest LMP",    f"${min_lmp:.2f}", f"Hour {low_hr}")
    with k4: metric_card("Highest LMP",   f"${max_lmp:.2f}", f"Hour {high_hr}")
    with k5: metric_card("LMP Spread",    f"${spread:.2f}",  "$/MWh")

    st.markdown("")
    r1, r2 = st.columns(2)
    with r1: metric_card("2H BESS Arbitrage", f"${rev2:.2f}", "$/MWh net spread")
    with r2: metric_card("4H BESS Arbitrage", f"${rev4:.2f}", "$/MWh net spread")

    st.markdown("---")

    # ── Chart ────────────────────────────────
    fig = go.Figure()

    # Shade regions — drawn first so they sit behind the line
    def add_shade(fig, x0, x1, color, label, opacity=0.12):
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color, opacity=opacity,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=10, color=color)
        )

    if show_2hr:
        add_shade(fig, *charge_2hr_win,    "#00e676", "CHARGE 2H",    opacity=0.13)
        add_shade(fig, *discharge_2hr_win, "#ff9800", "DISCHARGE 2H", opacity=0.13)

    if show_4hr:
        add_shade(fig, *charge_4hr_win,    "#00bcd4", "CHARGE 4H",    opacity=0.08)
        add_shade(fig, *discharge_4hr_win, "#ff5722", "DISCHARGE 4H", opacity=0.08)

    # LMP line
    fig.add_trace(go.Scatter(
        x=bdf["Hour"], y=bdf["LMP"],
        name="LMP Price",
        line=dict(color="#00d4ff", width=3),
        hovertemplate="Hour %{x}<br>LMP: $%{y:.2f}/MWh<extra></extra>"
    ))

    # Min / Max markers
    fig.add_trace(go.Scatter(
        x=[low_hr],  y=[min_lmp],
        mode="markers+text",
        marker=dict(size=14, color="#00e676", symbol="triangle-up"),
        text=["  Charge"],
        textposition="middle right",
        textfont=dict(color="#00e676", size=11),
        name=f"Lowest LMP  ${min_lmp:.2f}",
        hovertemplate=f"Hour {low_hr} — Min LMP: ${min_lmp:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[high_hr], y=[max_lmp],
        mode="markers+text",
        marker=dict(size=14, color="#ff8c42", symbol="triangle-down"),
        text=["  Discharge"],
        textposition="middle right",
        textfont=dict(color="#ff8c42", size=11),
        name=f"Highest LMP  ${max_lmp:.2f}",
        hovertemplate=f"Hour {high_hr} — Max LMP: ${max_lmp:.2f}<extra></extra>"
    ))

    # 2hr storage curve
    if show_2hr:
        bdf["_2hr_state"] = 0
        for i in bdf.index:
            h = bdf.loc[i, "Hour"]
            if charge_2hr_win[0] <= h <= charge_2hr_win[1]:
                bdf.loc[i, "_2hr_state"] = -1    # charging
            elif discharge_2hr_win[0] <= h <= discharge_2hr_win[1]:
                bdf.loc[i, "_2hr_state"] = 1     # discharging

        band2 = spread * 0.12
        bdf["_2hr_curve"] = bdf["LMP"] + bdf["_2hr_state"] * band2
        fig.add_trace(go.Scatter(
            x=bdf["Hour"], y=bdf["_2hr_curve"],
            name="2H Storage Band",
            line=dict(color="#00e676", width=2, dash="dot", shape="hv"),
            hovertemplate="Hour %{x}<br>2H Band: $%{y:.2f}<extra></extra>"
        ))

    # 4hr storage curve
    if show_4hr:
        bdf["_4hr_state"] = 0
        for i in bdf.index:
            h = bdf.loc[i, "Hour"]
            if charge_4hr_win[0] <= h <= charge_4hr_win[1]:
                bdf.loc[i, "_4hr_state"] = -1
            elif discharge_4hr_win[0] <= h <= discharge_4hr_win[1]:
                bdf.loc[i, "_4hr_state"] = 1

        band4 = spread * 0.20
        bdf["_4hr_curve"] = bdf["LMP"] + bdf["_4hr_state"] * band4
        fig.add_trace(go.Scatter(
            x=bdf["Hour"], y=bdf["_4hr_curve"],
            name="4H Storage Band",
            line=dict(color="#ff8c42", width=2, dash="dash", shape="hv"),
            hovertemplate="Hour %{x}<br>4H Band: $%{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"LMP Intraday Curve — <b>{bus}</b>  |  {selected_date}",
            font=dict(size=15)
        ),
        xaxis=dict(
            title="Hour Ending",
            tickmode="linear", dtick=1,
            showgrid=True, gridcolor="rgba(255,255,255,0.05)"
        ),
        yaxis=dict(
            title="LMP ($/MWh)",
            showgrid=True, gridcolor="rgba(255,255,255,0.05)"
        ),
        legend=dict(
            orientation="h", y=1.08, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Strategy box ────────────────────────
    strategy_fn(strategy_label)

    # ── Raw data expander ────────────────────
    with st.expander("📄  Raw Data for this Bus / Date"):
        show_cols = ["Date", "Hour", "Bus", "LMP"]
        st.dataframe(
            bdf[show_cols].rename(columns={
                "Hour": "Hour Ending",
                "Bus":  "Bus Name",
                "LMP":  "LMP ($/MWh)"
            }),
            use_container_width=True
        )
