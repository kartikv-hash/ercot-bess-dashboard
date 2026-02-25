
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import math
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="ERCOT BESS Dashboard", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
if "selected_bus" not in st.session_state:
    st.session_state.selected_bus = None
if "lmp_df" not in st.session_state:
    st.session_state.lmp_df = None

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .card {
      background:#1a1d2e; border:1px solid #2a2d45;
      border-radius:12px; padding:16px 20px; margin-bottom:4px;
  }
  .card-title { color:#7880a8; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
  .card-value { color:#ffffff; font-size:22px; font-weight:700; margin-top:4px; }
  .card-sub   { color:#5de0a5; font-size:11px; margin-top:3px; }
  .section-header {
      font-size:12px; font-weight:600; color:#7880a8;
      text-transform:uppercase; letter-spacing:1px; margin:18px 0 10px 0;
  }
  .copilot-msg-user { background:#1e2a3a; border-radius:10px; padding:10px 14px; margin:6px 0; }
  .copilot-msg-ai   { background:#1a2e1e; border-radius:10px; padding:10px 14px; margin:6px 0; border-left:3px solid #5de0a5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def metric_card(title, value, sub=""):
    st.markdown(f"""<div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return round(R * 2 * math.asin(math.sqrt(a)), 2)

KV_MAP = {
    "34.5 kV": 34500, "69 kV": 69000, "115 kV": 115000,
    "138 kV": 138000, "230 kV": 230000, "345 kV": 345000,
    "500 kV": 500000, "765 kV": 765000,
}

def nearest_kv_label(volts):
    best, best_diff = None, float("inf")
    for label, nominal in KV_MAP.items():
        diff = abs(volts - nominal) / nominal
        if diff < best_diff:
            best, best_diff = label, diff
    return best if best_diff < 0.20 else f"{volts/1000:.1f} kV"

@st.cache_data
def load_lmp(f):
    raw = pd.read_csv(f, sep=None, engine="python")
    raw.columns = [c.strip() for c in raw.columns]
    rename_rules = {
        "DeliveryDate":"Date","DELIVERYDATE":"Date","Oper Day":"Date","OperDay":"Date","OPERDAY":"Date","SETTLEMENT_DATE":"Date",
        "HourEnding":"Hour","HOURENDING":"Hour","Hour Ending":"Hour","HOUR_ENDING":"Hour","HE":"Hour",
        "BusName":"Bus","BUSNAME":"Bus","Bus Name":"Bus","SETTLEMENT_POINT":"Bus","Settlement Point":"Bus","Node":"Bus",
        "LMP":"LMP","SETTLEMENT_POINT_PRICE":"LMP","Price":"LMP",
        "DSTFlag":"DST","DSTFLAG":"DST","DST Flag":"DST","DST_FLAG":"DST",
    }
    raw = raw.rename(columns={c: rename_rules[c] for c in raw.columns if c in rename_rules})
    for col in ["Date","Hour","Bus","LMP"]:
        if col not in raw.columns:
            raw[col] = None
    if raw["Hour"].dtype == object:
        raw["Hour"] = raw["Hour"].astype(str).str.strip().str.extract(r"(\d+)")[0].astype(float)
    raw["LMP"]  = pd.to_numeric(raw["LMP"],  errors="coerce")
    raw["Date"] = raw["Date"].astype(str).str.strip()
    raw["Bus"]  = raw["Bus"].astype(str).str.strip()
    raw = raw.dropna(subset=["Hour","LMP"]).reset_index(drop=True)
    return raw

def bess_revenue(bdf, half_charge, half_discharge):
    low_i  = bdf["LMP"].idxmin()
    high_i = bdf["LMP"].idxmax()
    low_hr  = bdf.loc[low_i,  "Hour"]
    high_hr = bdf.loc[high_i, "Hour"]
    ch_mask  = (bdf["Hour"] >= low_hr  - half_charge)  & (bdf["Hour"] <= low_hr  + half_charge)
    dis_mask = (bdf["Hour"] >= high_hr - half_discharge) & (bdf["Hour"] <= high_hr + half_discharge)
    ch_avg  = bdf.loc[ch_mask,  "LMP"].mean() if ch_mask.any()  else 0
    dis_avg = bdf.loc[dis_mask, "LMP"].mean() if dis_mask.any() else 0
    return round(dis_avg - ch_avg, 2), low_hr, high_hr

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ERCOT BESS")
    st.markdown("---")
    page = st.radio("Module", [
        "🗺️  Node Analyser",
        "📈  LMP Price Analysis",
        "🤖  AI Copilot"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📂 LMP Data")
    uploaded = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = load_lmp(uploaded)
        st.session_state.lmp_df = df
        st.success(f"✅ {len(df):,} rows loaded")
    elif st.session_state.lmp_df is not None:
        df = st.session_state.lmp_df
    else:
        df = None


# ══════════════════════════════════════════════
#  PAGE 1 – NODE ANALYSER
# ══════════════════════════════════════════════
if page == "🗺️  Node Analyser":
    st.title("🗺️  Hub & Node Analyser")
    st.caption("Discovers transmission substations from OpenStreetMap. Hubs = 230 kV+ | Nodes = below 230 kV")

    st.markdown('<div class="section-header">Search Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        lat = st.number_input("Latitude",  value=33.7944, format="%.4f")
        lon = st.number_input("Longitude", value=-98.5706, format="%.4f")
    with c2:
        radius_miles = st.selectbox("Search Radius (miles)", [5,10,25,50,100,150], index=2)
        hub_threshold_kv = st.selectbox("Hub threshold (kV ≥)", [115,138,230,345], index=2)
    with c3:
        selected_kv_labels = st.multiselect("Filter Voltages", list(KV_MAP.keys()), default=list(KV_MAP.keys()))
        show_unknown_v = st.checkbox("Include unknown voltage substations", value=True)

    zoom_guess = max(7, min(14, round(14 - math.log2(max(1, radius_miles)))))
    infra_url = f"https://openinframap.org/#{zoom_guess}/{lat:.4f}/{lon:.4f}"
    st.markdown(f'🔗 **[Open this area in OpenInfraMap]({infra_url})**', unsafe_allow_html=True)

    run_search = st.button("🔍  Search Substations", type="primary")

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
        with st.spinner("Querying OpenStreetMap …"):
            try:
                resp = requests.post("https://overpass-api.de/api/interpreter",
                                     data={"data": overpass_query}, timeout=45)
                resp.raise_for_status()
                osm_data = resp.json()
            except requests.exceptions.Timeout:
                st.error("⏱ Overpass API timed out. Try a smaller radius.")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        rows = []
        for el in osm_data.get("elements", []):
            tags = el.get("tags", {})
            if el["type"] == "node":
                slat, slon = el.get("lat"), el.get("lon")
            elif "center" in el:
                slat, slon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            if slat is None or slon is None:
                continue
            volt_raw = tags.get("voltage","")
            volts = None
            if volt_raw:
                try:
                    volts = max(int(v) for v in volt_raw.split(";") if v.strip().isdigit())
                except Exception:
                    pass
            if volts is not None:
                nearest = nearest_kv_label(volts)
                nom = KV_MAP.get(nearest)
                if nom and nom not in selected_nominal_volts:
                    continue
                kv_label = nearest
            else:
                if not show_unknown_v:
                    continue
                kv_label = "Unknown"

            sub_type = tags.get("substation","")
            classification = "Hub" if (sub_type == "transmission" or (volts is not None and volts >= hub_threshold_v)) else "Node"

            rows.append({
                "Name": tags.get("name", f"Substation {el['id']}"),
                "Type": classification,
                "Voltage": kv_label,
                "Operator": tags.get("operator","—"),
                "Substation Tag": sub_type if sub_type else "—",
                "Lat": round(slat,5),
                "Lon": round(slon,5),
                "Distance (mi)": haversine_miles(lat, lon, slat, slon),
                "_id": el["id"],
            })

        if not rows:
            st.warning("No substations found. Try a larger radius or loosen the voltage filters.")
            st.stop()

        sdf = pd.DataFrame(rows).sort_values("Distance (mi)").reset_index(drop=True)
        n_hub  = (sdf["Type"] == "Hub").sum()
        n_node = (sdf["Type"] == "Node").sum()

        st.markdown("---")
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1: metric_card("Total Found", str(len(sdf)))
        with k2: metric_card("Hubs",  str(n_hub),  f"≥ {hub_threshold_kv} kV")
        with k3: metric_card("Nodes", str(n_node), f"< {hub_threshold_kv} kV")
        with k4: metric_card("Radius", f"{radius_miles} mi")
        with k5: metric_card("Centre", f"{lat:.3f}, {lon:.3f}")
        st.markdown("")

        # Map
        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon], mode="markers+text",
            marker=dict(size=16, color="#FFD700"),
            text=["📍 Centre"], textposition="top right",
            name="Search Centre",
            hovertext=f"Centre: {lat:.4f}, {lon:.4f}", hoverinfo="text"
        ))
        circle_lats, circle_lons = [], []
        for deg in range(0, 361, 4):
            rad = math.radians(deg)
            dlat = (radius_miles/3958.8) * math.cos(rad)
            dlon = (radius_miles/3958.8) * math.sin(rad) / math.cos(math.radians(lat))
            circle_lats.append(lat + math.degrees(dlat))
            circle_lons.append(lon + math.degrees(dlon))
        fig_map.add_trace(go.Scattermapbox(
            lat=circle_lats, lon=circle_lons, mode="lines",
            line=dict(color="rgba(255,215,0,0.3)", width=1),
            name=f"{radius_miles} mi radius", hoverinfo="skip"
        ))
        hubs_df  = sdf[sdf["Type"]=="Hub"]
        nodes_df = sdf[sdf["Type"]=="Node"]
        if len(hubs_df):
            fig_map.add_trace(go.Scattermapbox(
                lat=hubs_df["Lat"], lon=hubs_df["Lon"], mode="markers",
                marker=dict(size=13, color="#bf7fff"), name="Hub",
                hovertemplate="<b>%{customdata[0]}</b><br>Type: Hub<br>Voltage: %{customdata[1]}<br>Distance: %{customdata[2]} mi<extra></extra>",
                customdata=hubs_df[["Name","Voltage","Distance (mi)"]].values
            ))
        if len(nodes_df):
            fig_map.add_trace(go.Scattermapbox(
                lat=nodes_df["Lat"], lon=nodes_df["Lon"], mode="markers",
                marker=dict(size=9, color="#4fc3f7"), name="Node",
                hovertemplate="<b>%{customdata[0]}</b><br>Type: Node<br>Voltage: %{customdata[1]}<br>Distance: %{customdata[2]} mi<extra></extra>",
                customdata=nodes_df[["Name","Voltage","Distance (mi)"]].values
            ))
        fig_map.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=lat, lon=lon),
                        zoom=max(7, min(12, round(12-math.log2(max(1,radius_miles)))))),
            margin=dict(l=0,r=0,t=0,b=0), height=520,
            legend=dict(bgcolor="rgba(20,22,38,0.85)", font=dict(color="white"), x=0.01, y=0.99)
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Table + LMP link
        st.markdown('<div class="section-header">Substation List</div>', unsafe_allow_html=True)
        display_df = sdf.drop(columns=["_id"])

        def _style_type(val):
            if val == "Hub": return "background-color:#3b1f5e;color:#bf7fff;font-weight:600"
            return "background-color:#1a3550;color:#4fc3f7;font-weight:600"

        st.dataframe(
            display_df.style.applymap(_style_type, subset=["Type"]),
            use_container_width=True, height=300
        )

        # 🔗 Link to LMP page
        if df is not None:
            st.markdown('<div class="section-header">Link Node to LMP Analysis</div>', unsafe_allow_html=True)
            st.caption("Select a substation name to pre-fill the LMP Bus selector (partial match)")
            link_node = st.selectbox("Select substation to analyse in LMP", sdf["Name"].tolist(), key="node_link")
            if st.button("📈  Go to LMP Analysis for this Node", type="secondary"):
                # Try to fuzzy-match to a bus name
                bus_list = df["Bus"].unique().tolist()
                keyword = link_node.split()[0].upper() if link_node else ""
                matched = [b for b in bus_list if keyword in b.upper()]
                st.session_state.selected_bus = matched[0] if matched else bus_list[0]
                st.info(f"Matched to bus: **{st.session_state.selected_bus}** — switch to 📈 LMP Price Analysis in the sidebar.")


# ══════════════════════════════════════════════
#  PAGE 2 – LMP PRICE ANALYSIS
# ══════════════════════════════════════════════
elif page == "📈  LMP Price Analysis":
    st.title("📈  LMP Price Analysis")

    if df is None:
        st.info("👈  Upload an ERCOT LMP CSV from the sidebar to continue.")
        st.stop()

    # ── Tabs ────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Single Bus Analysis", "📊 Top N Buses by Spread", "📥 Export Revenue Table"])

    # ═══ TAB 1 – Single Bus ═══════════════
    with tab1:
        bus_list = sorted(df["Bus"].unique().tolist())
        default_bus_idx = 0
        if st.session_state.selected_bus and st.session_state.selected_bus in bus_list:
            default_bus_idx = bus_list.index(st.session_state.selected_bus)

        bus = st.selectbox("Search & select Bus name", bus_list, index=default_bus_idx)
        st.session_state.selected_bus = bus

        dates = sorted(df["Date"].unique().tolist())

        # Multi-date toggle
        multi_date = st.checkbox("📅 Compare multiple dates on one chart", value=False)

        if multi_date:
            sel_dates = st.multiselect("Select Dates to Compare", dates, default=dates[:min(3,len(dates))])
            if not sel_dates:
                st.warning("Select at least one date.")
                st.stop()
        else:
            if len(dates) > 1:
                sel_date = st.selectbox("Select Date", dates)
            elif len(dates) == 1:
                sel_date = dates[0]
            else:
                st.error("Could not parse dates from CSV.")
                st.stop()
            sel_dates = [sel_date]

        show_2hr = st.checkbox("⚡ 2-Hour Storage", value=True)
        show_4hr = st.checkbox("⚡ 4-Hour Storage", value=True)

        PALETTE = ["#00d4ff","#ff8c42","#5de0a5","#bf7fff","#FFD700","#ff6b6b","#74c0fc"]

        fig = go.Figure()
        summary_rows = []

        for di, date in enumerate(sel_dates):
            bdf = df[(df["Bus"]==bus) & (df["Date"]==date)].sort_values("Hour").reset_index(drop=True)
            if bdf.empty:
                continue

            col = PALETTE[di % len(PALETTE)]
            low_i  = bdf["LMP"].idxmin()
            high_i = bdf["LMP"].idxmax()
            low_hr  = bdf.loc[low_i, "Hour"]
            high_hr = bdf.loc[high_i, "Hour"]
            min_lmp = bdf["LMP"].min()
            max_lmp = bdf["LMP"].max()
            spread  = max_lmp - min_lmp
            rev2, _, _ = bess_revenue(bdf, 1, 1)
            rev4, _, _ = bess_revenue(bdf, 2, 2)

            summary_rows.append({
                "Date": date, "Bus": bus,
                "Min LMP": round(min_lmp,2), "Min Hour": low_hr,
                "Max LMP": round(max_lmp,2), "Max Hour": high_hr,
                "Spread": round(spread,2),
                "2H Revenue ($/MWh)": rev2,
                "4H Revenue ($/MWh)": rev4,
            })

            # LMP line
            fig.add_trace(go.Scatter(
                x=bdf["Hour"], y=bdf["LMP"],
                name=f"LMP {date}", line=dict(color=col, width=2.5),
                hovertemplate=f"Date: {date}<br>Hour %{{x}}<br>LMP: $%{{y:.2f}}<extra></extra>"
            ))

            # Only overlay BESS bands on single-date mode (to avoid clutter)
            if not multi_date:
                def window(center_hr, hw):
                    return max(bdf["Hour"].min(), center_hr - hw), min(bdf["Hour"].max(), center_hr + hw)

                c2a, c2b = window(low_hr, 1)
                d2a, d2b = window(high_hr, 1)
                c4a, c4b = window(low_hr, 2)
                d4a, d4b = window(high_hr, 2)

                if show_2hr:
                    fig.add_vrect(x0=c2a, x1=c2b, fillcolor="#00e676", opacity=0.10, line_width=0,
                                  annotation_text="Charge 2H", annotation_font=dict(size=9,color="#00e676"))
                    fig.add_vrect(x0=d2a, x1=d2b, fillcolor="#ff9800", opacity=0.10, line_width=0,
                                  annotation_text="Discharge 2H", annotation_font=dict(size=9,color="#ff9800"))
                    bdf["_2s"] = bdf["Hour"].apply(
                        lambda h: -1 if c2a<=h<=c2b else (1 if d2a<=h<=d2b else 0))
                    bdf["_2c"] = bdf["LMP"] + bdf["_2s"] * spread * 0.12
                    fig.add_trace(go.Scatter(x=bdf["Hour"], y=bdf["_2c"],
                        name="2H Band", line=dict(color="#00e676",width=2,dash="dot",shape="hv"),
                        hovertemplate="Hour %{x}<br>2H Band: $%{y:.2f}<extra></extra>"))

                if show_4hr:
                    fig.add_vrect(x0=c4a, x1=c4b, fillcolor="#00bcd4", opacity=0.07, line_width=0,
                                  annotation_text="Charge 4H", annotation_font=dict(size=9,color="#00bcd4"))
                    fig.add_vrect(x0=d4a, x1=d4b, fillcolor="#ff5722", opacity=0.07, line_width=0,
                                  annotation_text="Discharge 4H", annotation_font=dict(size=9,color="#ff5722"))
                    bdf["_4s"] = bdf["Hour"].apply(
                        lambda h: -1 if c4a<=h<=c4b else (1 if d4a<=h<=d4b else 0))
                    bdf["_4c"] = bdf["LMP"] + bdf["_4s"] * spread * 0.20
                    fig.add_trace(go.Scatter(x=bdf["Hour"], y=bdf["_4c"],
                        name="4H Band", line=dict(color="#ff8c42",width=2,dash="dash",shape="hv"),
                        hovertemplate="Hour %{x}<br>4H Band: $%{y:.2f}<extra></extra>"))

                # Min/Max markers
                fig.add_trace(go.Scatter(
                    x=[low_hr], y=[min_lmp], mode="markers+text",
                    marker=dict(size=12,color="#00e676",symbol="triangle-up"),
                    text=[f"  ${min_lmp:.0f}"], textposition="middle right",
                    textfont=dict(color="#00e676",size=10), name=f"Min LMP",
                    hovertemplate=f"Hour {low_hr} — Min: ${min_lmp:.2f}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=[high_hr], y=[max_lmp], mode="markers+text",
                    marker=dict(size=12,color="#ff8c42",symbol="triangle-down"),
                    text=[f"  ${max_lmp:.0f}"], textposition="middle right",
                    textfont=dict(color="#ff8c42",size=10), name=f"Max LMP",
                    hovertemplate=f"Hour {high_hr} — Max: ${max_lmp:.2f}<extra></extra>"
                ))

        fig.update_layout(
            template="plotly_dark",
            title=dict(text=f"LMP Intraday Curve — <b>{bus}</b>", font=dict(size=15)),
            xaxis=dict(title="Hour Ending", tickmode="linear", dtick=1, showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="LMP ($/MWh)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary KPIs
        if summary_rows:
            srow = summary_rows[0]
            k1,k2,k3,k4,k5 = st.columns(5)
            with k1: metric_card("Bus", bus)
            with k2: metric_card("Lowest LMP",  f"${srow['Min LMP']:.2f}", f"Hour {srow['Min Hour']}")
            with k3: metric_card("Highest LMP", f"${srow['Max LMP']:.2f}", f"Hour {srow['Max Hour']}")
            with k4: metric_card("2H Arbitrage", f"${srow['2H Revenue ($/MWh)']:.2f}", "$/MWh net")
            with k5: metric_card("4H Arbitrage", f"${srow['4H Revenue ($/MWh)']:.2f}", "$/MWh net")
            st.markdown("")
            spread_val = srow["Spread"]
            if spread_val > 80:   st.success("✅  Pure Merchant Arbitrage Opportunity")
            elif spread_val > 40: st.warning("⚠️  Solar + Storage Overbuild Recommended")
            else:                 st.error("❌  Low Spread → Capacity / Ancillary Market Focus")

        if multi_date and summary_rows:
            st.markdown("---")
            st.markdown('<div class="section-header">Multi-Date Summary</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # ═══ TAB 2 – Top N Buses ═════════════
    with tab2:
        st.markdown('<div class="section-header">Top N Buses by LMP Spread</div>', unsafe_allow_html=True)

        dates_all = sorted(df["Date"].unique().tolist())
        c1, c2 = st.columns(2)
        with c1:
            top_n = st.slider("Show Top N Buses", 5, 50, 15)
        with c2:
            if len(dates_all) > 1:
                top_date = st.selectbox("For Date", dates_all, key="top_date")
            else:
                top_date = dates_all[0]
                st.markdown(f"**Date:** {top_date}")

        filt = df[df["Date"] == top_date]
        bus_stats = (
            filt.groupby("Bus")["LMP"]
            .agg(Min_LMP="min", Max_LMP="max",
                 Avg_LMP="mean", Std_Dev="std",
                 Spread=lambda x: x.max()-x.min())
            .round(2).reset_index()
            .sort_values("Spread", ascending=False)
            .head(top_n)
        )

        # Compute BESS revenue for each top bus
        rev_rows = []
        for _, row in bus_stats.iterrows():
            bdf_t = filt[filt["Bus"]==row["Bus"]].sort_values("Hour").reset_index(drop=True)
            r2, lh2, hh2 = bess_revenue(bdf_t, 1, 1)
            r4, lh4, hh4 = bess_revenue(bdf_t, 2, 2)
            rev_rows.append({"Bus": row["Bus"], "2H Rev":r2, "4H Rev":r4,
                             "Low Hr":lh2, "High Hr":hh2})
        rev_df = pd.DataFrame(rev_rows)
        top_df = bus_stats.merge(rev_df, on="Bus")

        # Bar chart
        fig_top = px.bar(
            top_df, x="Bus", y="Spread",
            color="Spread",
            color_continuous_scale=["#f07070","#f0c040","#5de0a5"],
            labels={"Spread":"LMP Spread ($/MWh)"},
            template="plotly_dark",
            title=f"Top {top_n} Buses by LMP Spread — {top_date}"
        )
        fig_top.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig_top, use_container_width=True)

        # Scatter: Spread vs 4H Revenue
        fig_sc = px.scatter(
            top_df, x="Spread", y="4H Rev", text="Bus", size="Spread",
            color="4H Rev", color_continuous_scale=["#4fc3f7","#bf7fff","#ff8c42"],
            template="plotly_dark", title="Spread vs 4H BESS Revenue",
            labels={"Spread":"LMP Spread ($/MWh)","4H Rev":"4H Revenue ($/MWh)"}
        )
        fig_sc.update_traces(textposition="top center", textfont=dict(size=8))
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

        def _style_spread(val):
            if val > 80: return "background-color:#1f4b2e;color:#5de0a5"
            if val > 40: return "background-color:#3d3510;color:#f0c040"
            return "background-color:#2e1a1a;color:#f07070"

        st.dataframe(
            top_df.style.applymap(_style_spread, subset=["Spread"]),
            use_container_width=True, height=320
        )

        # Quick-select best bus
        if not top_df.empty:
            best_bus = top_df.iloc[0]["Bus"]
            if st.button(f"📈  Analyse #{1} Bus: {best_bus} in Single Bus tab"):
                st.session_state.selected_bus = best_bus
                st.info(f"Switched to **{best_bus}** — click the 'Single Bus Analysis' tab above.")

    # ═══ TAB 3 – Export ══════════════════
    with tab3:
        st.markdown('<div class="section-header">Export BESS Revenue for All Buses</div>', unsafe_allow_html=True)

        dates_exp = sorted(df["Date"].unique().tolist())
        if len(dates_exp) > 1:
            exp_date = st.selectbox("Select Date for Export", dates_exp, key="exp_date")
        else:
            exp_date = dates_exp[0]

        if st.button("⚙️  Compute Revenue for All Buses", type="primary"):
            with st.spinner("Calculating …"):
                exp_rows = []
                for bus_name in df["Bus"].unique():
                    bdf_e = df[(df["Bus"]==bus_name) & (df["Date"]==exp_date)].sort_values("Hour").reset_index(drop=True)
                    if bdf_e.empty or len(bdf_e) < 3:
                        continue
                    r2, lh, hh = bess_revenue(bdf_e, 1, 1)
                    r4, _,  __  = bess_revenue(bdf_e, 2, 2)
                    sp = round(bdf_e["LMP"].max() - bdf_e["LMP"].min(), 2)
                    if sp > 80:   rec = "Merchant Arbitrage"
                    elif sp > 40: rec = "Solar + Storage Overbuild"
                    else:         rec = "Ancillary / Capacity"
                    exp_rows.append({
                        "Date": exp_date, "Bus": bus_name,
                        "Min LMP": round(bdf_e["LMP"].min(),2),
                        "Max LMP": round(bdf_e["LMP"].max(),2),
                        "Avg LMP": round(bdf_e["LMP"].mean(),2),
                        "Spread ($/MWh)": sp,
                        "Low Price Hour": lh,
                        "High Price Hour": hh,
                        "2H Revenue ($/MWh)": r2,
                        "4H Revenue ($/MWh)": r4,
                        "Recommended Strategy": rec,
                    })

            exp_df = pd.DataFrame(exp_rows).sort_values("Spread ($/MWh)", ascending=False).reset_index(drop=True)
            st.success(f"✅  Computed {len(exp_df)} buses")
            st.dataframe(exp_df, use_container_width=True, height=400)

            csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥  Download CSV",
                data=csv_bytes,
                file_name=f"BESS_Revenue_{exp_date}.csv",
                mime="text/csv",
                type="primary"
            )


# ══════════════════════════════════════════════
#  PAGE 3 – AI COPILOT
# ══════════════════════════════════════════════
elif page == "🤖  AI Copilot":
    st.title("🤖  AI Copilot")
    st.caption("Ask questions about your LMP data. Powered by Claude (Anthropic).")

    # API Key input
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔑 Anthropic API Key")
        api_key = st.text_input("Enter API Key", type="password", key="api_key",
                                help="Get your key at console.anthropic.com")

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to activate the AI Copilot.")
        st.markdown("""
        **How to get a key:**
        1. Go to [console.anthropic.com](https://console.anthropic.com)
        2. Sign up / log in
        3. Navigate to **API Keys** → Create new key
        4. Paste it in the sidebar
        """)
        st.stop()

    if df is None:
        st.info("👈  Upload an ERCOT LMP CSV from the sidebar first.")
        st.stop()

    # ── Build data context ───────────────────
    dates_ai = sorted(df["Date"].unique().tolist())
    bus_list_ai = sorted(df["Bus"].unique().tolist())
    top_buses_ai = (
        df.groupby("Bus")["LMP"]
        .agg(spread=lambda x: x.max()-x.min(), avg="mean", max="max", min="min")
        .sort_values("spread", ascending=False)
        .head(10).round(2).reset_index()
    )

    data_context = f"""
You are an expert ERCOT energy market analyst and BESS (Battery Energy Storage System) strategy advisor.

The user has uploaded an ERCOT LMP dataset with the following characteristics:
- Dates: {", ".join(dates_ai)}
- Total buses/nodes: {len(bus_list_ai)}
- Total records: {len(df):,}
- Overall avg LMP: ${df['LMP'].mean():.2f}/MWh
- Overall max LMP: ${df['LMP'].max():.2f}/MWh (Bus: {df.loc[df['LMP'].idxmax(),'Bus']})
- Overall min LMP: ${df['LMP'].min():.2f}/MWh (Bus: {df.loc[df['LMP'].idxmin(),'Bus']})

Top 10 buses by LMP Spread:
{top_buses_ai.to_string(index=False)}

BESS Strategy thresholds used:
- Spread > $80/MWh → Pure Merchant Arbitrage
- Spread $40–$80/MWh → Solar + Storage Overbuild
- Spread < $40/MWh → Capacity / Ancillary Market Focus

2H BESS: charges 1 hour before/after the daily price minimum, discharges 1 hour before/after the daily maximum.
4H BESS: charges 2 hours before/after the daily price minimum, discharges 2 hours before/after the daily maximum.

Answer all questions concisely and in the context of ERCOT energy markets and BESS development strategy.
Always be specific using the numbers from the dataset above.
"""

    # ── Chat history ─────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested prompts
    st.markdown('<div class="section-header">Suggested Questions</div>', unsafe_allow_html=True)
    q_cols = st.columns(3)
    suggestions = [
        "Which buses have the best arbitrage opportunity?",
        "What is the recommended BESS strategy for the top bus?",
        "Summarise the overall LMP price trends in this dataset",
        "Which hours have the highest and lowest average prices?",
        "Compare 2H vs 4H storage economics for top nodes",
        "What does the spread distribution suggest about market volatility?",
    ]
    for i, q in enumerate(suggestions):
        with q_cols[i % 3]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})

    st.markdown("---")

    # Chat input
    user_input = st.chat_input("Ask anything about your ERCOT LMP data …")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})

    # Display history + call API for last unanswered message
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if msg["role"]=="user" else "assistant",
                             avatar="👤" if msg["role"]=="user" else "🤖"):
            st.markdown(msg["content"])

        # If this is the last message and it's from user, generate response
        if msg["role"] == "user" and i == len(st.session_state.chat_history) - 1:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analysing …"):
                    messages_payload = [{"role":"user","content":data_context + "\n\n---\n\nUser question: " + st.session_state.chat_history[0]["content"]}]
                    for h in st.session_state.chat_history[1:]:
                        messages_payload.append({"role": h["role"], "content": h["content"]})
                    try:
                        ai_resp = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": api_key,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json"
                            },
                            json={
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 1024,
                                "messages": messages_payload
                            },
                            timeout=30
                        )
                        ai_resp.raise_for_status()
                        answer = ai_resp.json()["content"][0]["text"]
                    except requests.exceptions.HTTPError as e:
                        answer = f"❌ API error {ai_resp.status_code}: {ai_resp.text}"
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"

                    st.markdown(answer)
                    st.session_state.chat_history.append({"role":"assistant","content":answer})

    if st.session_state.chat_history:
        if st.button("🗑️  Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
