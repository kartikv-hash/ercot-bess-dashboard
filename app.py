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
#  SESSION STATE
# ─────────────────────────────────────────────
if "selected_bus"   not in st.session_state: st.session_state.selected_bus   = None
if "lmp_df"         not in st.session_state: st.session_state.lmp_df         = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []

SECOND_APP_URL = "https://fatal-flaw-o7aks4agtoffgyydbvrguj.streamlit.app/"

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
    "34.5 kV":34500,"69 kV":69000,"115 kV":115000,
    "138 kV":138000,"230 kV":230000,"345 kV":345000,
    "500 kV":500000,"765 kV":765000,
}

def nearest_kv_label(volts):
    best, best_diff = None, float("inf")
    for label, nominal in KV_MAP.items():
        diff = abs(volts - nominal) / nominal
        if diff < best_diff:
            best, best_diff = label, diff
    return best if best_diff < 0.20 else f"{volts/1000:.1f} kV"

# ─────────────────────────────────────────────
#  ROLLING-AVERAGE BESS HELPER
#
#  Uses a 3-hr centred rolling average to find
#  the smoothed low & high price hours, then
#  sets charge/discharge windows of ±half_w hrs
#  around those smoothed peaks.
#  Returns: (net_revenue, roll_series,
#            low_hr, high_hr,
#            charge_window, discharge_window)
# ─────────────────────────────────────────────
def bess_calc(bdf: pd.DataFrame, half_w: int):
    """
    half_w = 1  →  2-hour storage  (±1 hr window)
    half_w = 2  →  4-hour storage  (±2 hr window)
    """
    roll = (bdf["LMP"]
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .reset_index(drop=True))

    low_idx  = roll.idxmin()
    high_idx = roll.idxmax()
    low_hr   = bdf.loc[low_idx,  "Hour"]
    high_hr  = bdf.loc[high_idx, "Hour"]

    hr_min = bdf["Hour"].min()
    hr_max = bdf["Hour"].max()

    charge_win    = (max(hr_min, low_hr  - half_w), min(hr_max, low_hr  + half_w))
    discharge_win = (max(hr_min, high_hr - half_w), min(hr_max, high_hr + half_w))

    ch_mask  = (bdf["Hour"] >= charge_win[0])    & (bdf["Hour"] <= charge_win[1])
    dis_mask = (bdf["Hour"] >= discharge_win[0]) & (bdf["Hour"] <= discharge_win[1])

    ch_avg  = bdf.loc[ch_mask,  "LMP"].mean() if ch_mask.any()  else 0
    dis_avg = bdf.loc[dis_mask, "LMP"].mean() if dis_mask.any() else 0
    revenue = round(dis_avg - ch_avg, 2)

    return revenue, roll, low_hr, high_hr, charge_win, discharge_win


@st.cache_data
def load_lmp(f):
    raw = pd.read_csv(f, sep=None, engine="python")
    raw.columns = [c.strip() for c in raw.columns]
    rename_rules = {
        "DeliveryDate":"Date","DELIVERYDATE":"Date","Oper Day":"Date","OperDay":"Date",
        "OPERDAY":"Date","SETTLEMENT_DATE":"Date",
        "HourEnding":"Hour","HOURENDING":"Hour","Hour Ending":"Hour",
        "HOUR_ENDING":"Hour","HE":"Hour",
        "BusName":"Bus","BUSNAME":"Bus","Bus Name":"Bus",
        "SETTLEMENT_POINT":"Bus","Settlement Point":"Bus","Node":"Bus",
        "LMP":"LMP","SETTLEMENT_POINT_PRICE":"LMP","Price":"LMP",
        "DSTFlag":"DST","DSTFLAG":"DST","DST Flag":"DST","DST_FLAG":"DST",
    }
    raw = raw.rename(columns={c: rename_rules[c] for c in raw.columns if c in rename_rules})
    for col in ["Date","Hour","Bus","LMP"]:
        if col not in raw.columns:
            raw[col] = None
    if raw["Hour"].dtype == object:
        raw["Hour"] = (raw["Hour"].astype(str).str.strip()
                       .str.extract(r"(\d+)")[0].astype(float))
    raw["LMP"]  = pd.to_numeric(raw["LMP"],  errors="coerce")
    raw["Date"] = raw["Date"].astype(str).str.strip()
    raw["Bus"]  = raw["Bus"].astype(str).str.strip()
    raw = raw.dropna(subset=["Hour","LMP"]).reset_index(drop=True)
    return raw


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ERCOT BESS")
    st.markdown("---")
    page = st.radio("Module", [
        "🏠  Home",
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
    st.markdown("---")
    st.markdown("### 🔗 Other Tools")
    st.markdown(f'<a href="{SECOND_APP_URL}" target="_blank"><button style="width:100%;padding:8px;background:#1e2a3a;color:#4fc3f7;border:1px solid #2a3d55;border-radius:8px;cursor:pointer;font-size:13px;">📍 Open Fatal Flaw Analyser — SiteIQ</button></a>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px 0;">
        <div style="font-size:52px;">⚡</div>
        <h1 style="font-size:32px;font-weight:800;margin:8px 0 4px 0;">ERCOT BESS Intelligence Platform</h1>
        <p style="color:#7880a8;font-size:15px;margin:0;">Battery Energy Storage & LMP Analysis Suite</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Select a Tool to Get Started")
    st.markdown("")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1d2e,#1e2540);border:1px solid #2a2d55;
                    border-radius:16px;padding:28px 24px;height:100%;">
            <div style="font-size:36px;margin-bottom:12px;">⚡</div>
            <div style="font-size:18px;font-weight:700;color:#fff;margin-bottom:8px;">
                ERCOT BESS Dashboard
            </div>
            <div style="color:#7880a8;font-size:13px;line-height:1.7;margin-bottom:20px;">
                <b style="color:#5de0a5;">Current App</b><br><br>
                🗺️ &nbsp;Hub & Node Analyser with OSM map<br>
                📈 &nbsp;LMP Price Analysis & BESS strategy<br>
                📊 &nbsp;Top N buses, spread ranking, export<br>
                🤖 &nbsp;AI Copilot powered by Claude
            </div>
            <div style="background:#5de0a520;border:1px solid #5de0a540;border-radius:8px;
                        padding:8px 14px;display:inline-block;color:#5de0a5;font-size:12px;font-weight:600;">
                ✅ You are here
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1d2e,#1e2540);border:1px solid #2a2d55;
                    border-radius:16px;padding:28px 24px;height:100%;">
            <div style="font-size:36px;margin-bottom:12px;">📍</div>
            <div style="font-size:18px;font-weight:700;color:#fff;margin-bottom:8px;">
                Fatal Flaw Analyser — SiteIQ
            </div>
            <div style="color:#7880a8;font-size:13px;line-height:1.7;margin-bottom:20px;">
                Renewable energy site screening & fatal flaw identification tool.<br><br>
                🗺️ &nbsp;Click-on-map site selection anywhere in the US<br>
                ⚠️ &nbsp;Fatal flaw screening — wetlands, flood zones, species<br>
                🌞 &nbsp;Solar & wind resource assessment per parcel<br>
                🏗️ &nbsp;Soil suitability, topography & grid access analysis<br>
                📊 &nbsp;Value & risk index scoring with PDF export
            </div>
            <a href="{SECOND_APP_URL}" target="_blank"
               style="background:#4f6ef7;color:#fff;padding:9px 20px;border-radius:8px;
                      font-size:13px;font-weight:600;text-decoration:none;display:inline-block;">
                🚀 Open SiteIQ →
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # Quick stats if data loaded
    if df is not None:
        st.markdown("### 📊 Loaded Dataset Overview")
        k1,k2,k3,k4 = st.columns(4)
        with k1: metric_card("Total Records",  f"{len(df):,}")
        with k2: metric_card("Unique Buses",   f"{df['Bus'].nunique():,}")
        with k3: metric_card("Avg LMP",        f"${df['LMP'].mean():.2f}", "$/MWh")
        with k4: metric_card("Max LMP",        f"${df['LMP'].max():.2f}", f"{df.loc[df['LMP'].idxmax(),'Bus']}")
        st.markdown("")
        top3 = (df.groupby("Bus")["LMP"]
                .agg(spread=lambda x: x.max()-x.min())
                .sort_values("spread", ascending=False)
                .head(3).reset_index())
        st.markdown("**🏆 Top 3 Buses by Spread**")
        t1, t2, t3 = st.columns(3)
        for col, (_, row) in zip([t1,t2,t3], top3.iterrows()):
            with col:
                metric_card(row["Bus"], f"${row['spread']:.2f}", "spread $/MWh")
    else:
        st.info("👈  Upload an ERCOT LMP CSV from the sidebar to see your dataset overview here.")

    st.markdown("")
    st.markdown("""
    <div style="text-align:center;color:#3a3e55;font-size:12px;padding:20px 0 0 0;">
        Built for ERCOT BESS Development · Powered by Claude AI · Data from OpenStreetMap & ERCOT
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 – NODE ANALYSER
# ══════════════════════════════════════════════
elif page == "🗺️  Node Analyser":
    st.title("🗺️  Hub & Node Analyser")
    st.caption("Discovers transmission substations from OpenStreetMap. Hubs = 230 kV+ | Nodes = below 230 kV")

    st.markdown('<div class="section-header">Search Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,3])
    with c1:
        lat_str = st.text_input("Latitude",  value="", placeholder="e.g. 33.7944")
        lon_str = st.text_input("Longitude", value="", placeholder="e.g. -98.5706")
    with c2:
        radius_miles     = st.selectbox("Search Radius (miles)", [5,10,25,50,100,150], index=2)
        hub_threshold_kv = st.selectbox("Hub threshold (kV ≥)", [115,138,230,345], index=2)
    with c3:
        selected_kv_labels = st.multiselect("Filter Voltages", list(KV_MAP.keys()), default=list(KV_MAP.keys()))
        show_unknown_v     = st.checkbox("Include unknown voltage substations", value=True)

    # Parse & validate
    lat, lon = None, None
    if lat_str.strip() and lon_str.strip():
        try:
            lat = float(lat_str.strip())
            lon = float(lon_str.strip())
        except ValueError:
            st.error("⚠️  Please enter valid numeric values for Latitude and Longitude.")

    if lat is not None and lon is not None:
        zoom_guess = max(7, min(14, round(14 - math.log2(max(1, radius_miles)))))
        infra_url  = f"https://openinframap.org/#{zoom_guess}/{lat:.4f}/{lon:.4f}"
        st.markdown(f'🔗 **[Open this area in OpenInfraMap]({infra_url})**', unsafe_allow_html=True)
    else:
        st.info("👆  Enter a Latitude and Longitude above to search for substations.")

    run_search = st.button("🔍  Search Substations", type="primary",
                           disabled=(lat is None or lon is None))

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
            sub_type       = tags.get("substation","")
            classification = ("Hub" if (sub_type=="transmission" or
                               (volts is not None and volts >= hub_threshold_v))
                              else "Node")
            rows.append({
                "Name":          tags.get("name", f"Substation {el['id']}"),
                "Type":          classification,
                "Voltage":       kv_label,
                "Operator":      tags.get("operator","—"),
                "Substation Tag":sub_type if sub_type else "—",
                "Lat":           round(slat,5),
                "Lon":           round(slon,5),
                "Distance (mi)": haversine_miles(lat, lon, slat, slon),
                "_id":           el["id"],
            })

        if not rows:
            st.warning("No substations found. Try a larger radius or loosen voltage filters.")
            st.stop()

        sdf   = pd.DataFrame(rows).sort_values("Distance (mi)").reset_index(drop=True)
        n_hub  = (sdf["Type"]=="Hub").sum()
        n_node = (sdf["Type"]=="Node").sum()

        st.markdown("---")
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1: metric_card("Total Found", str(len(sdf)))
        with k2: metric_card("Hubs",  str(n_hub),  f"≥ {hub_threshold_kv} kV")
        with k3: metric_card("Nodes", str(n_node), f"< {hub_threshold_kv} kV")
        with k4: metric_card("Radius", f"{radius_miles} mi")
        with k5: metric_card("Centre", f"{lat:.3f}, {lon:.3f}")
        st.markdown("")

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
            dlat = (radius_miles/3958.8)*math.cos(rad)
            dlon = (radius_miles/3958.8)*math.sin(rad)/math.cos(math.radians(lat))
            circle_lats.append(lat+math.degrees(dlat))
            circle_lons.append(lon+math.degrees(dlon))
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

        st.markdown('<div class="section-header">Substation List</div>', unsafe_allow_html=True)
        def _style_type(val):
            if val=="Hub": return "background-color:#3b1f5e;color:#bf7fff;font-weight:600"
            return "background-color:#1a3550;color:#4fc3f7;font-weight:600"
        st.dataframe(
            sdf.drop(columns=["_id"]).style.applymap(_style_type, subset=["Type"]),
            use_container_width=True, height=300
        )

        # ── Real-Time LMP from ERCOT public API ──
        st.markdown("---")
        st.markdown('<div class="section-header">⚡ Real-Time LMP for Selected Substation</div>', unsafe_allow_html=True)

        sel_sub = st.selectbox("Select Substation", sdf["Name"].tolist(), key="rtlmp_sub")

        # Let user override/confirm the ERCOT settlement point name
        st.caption("ERCOT settlement point names may differ from OSM names. Edit below if needed.")
        sp_guess = sel_sub.upper().replace(" ","_").replace("-","_")[:20]
        settlement_point = st.text_input("ERCOT Settlement Point Name", value=sp_guess,
                                         help="Must match an ERCOT bus/hub name exactly e.g. HB_NORTH, LZ_HOUSTON")

        fetch_rt = st.button("📡  Fetch Real-Time LMP", type="primary")

        if fetch_rt and settlement_point.strip():
            from datetime import datetime, timedelta, timezone
            now_utc   = datetime.now(timezone.utc)
            # ERCOT SCED runs every 5 min; pull last 2 hours
            ts_from   = (now_utc - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
            ts_to     = now_utc.strftime("%Y-%m-%dT%H:%M:%S")
            sp_clean  = settlement_point.strip()

            ercot_url = "https://api.ercot.com/api/public-reports/np6-788-er"
            params    = {
                "SCEDTimestampFrom": ts_from,
                "SCEDTimestampTo":   ts_to,
                "settlementPoint":   sp_clean,
                "size": 200,
            }
            headers = {"Ocp-Apim-Subscription-Key": "", "accept": "application/json"}

            with st.spinner(f"Fetching real-time LMP for {sp_clean} from ERCOT …"):
                try:
                    rt_resp = requests.get(ercot_url, params=params, timeout=20)
                    rt_resp.raise_for_status()
                    rt_json = rt_resp.json()

                    # ERCOT API returns data under 'data' key as list of lists
                    fields  = [f["name"] for f in rt_json.get("fields", [])]
                    records = rt_json.get("data", [])

                    if not records:
                        st.warning(f"No real-time data returned for **{sp_clean}**. "
                                   "Check the settlement point name matches an ERCOT bus exactly (e.g. HB_NORTH, LZ_WEST).")
                    else:
                        rt_df = pd.DataFrame(records, columns=fields)

                        # Normalise column names
                        rt_df.columns = [c.strip() for c in rt_df.columns]
                        ts_col  = next((c for c in rt_df.columns if "timestamp" in c.lower() or "time" in c.lower()), rt_df.columns[0])
                        lmp_col = next((c for c in rt_df.columns if "lmp" in c.lower() or "price" in c.lower()), rt_df.columns[-1])

                        rt_df[ts_col]  = pd.to_datetime(rt_df[ts_col], errors="coerce")
                        rt_df[lmp_col] = pd.to_numeric(rt_df[lmp_col], errors="coerce")
                        rt_df = rt_df.dropna(subset=[ts_col, lmp_col]).sort_values(ts_col)

                        latest_lmp  = rt_df[lmp_col].iloc[-1]
                        latest_time = rt_df[ts_col].iloc[-1].strftime("%H:%M UTC")
                        avg_lmp     = round(rt_df[lmp_col].mean(), 2)
                        max_lmp     = round(rt_df[lmp_col].max(), 2)
                        min_lmp     = round(rt_df[lmp_col].min(), 2)

                        # Store for AI summary context
                        st.session_state["rt_summary"] = {
                            "sp": sp_clean, "latest": latest_lmp,
                            "avg": avg_lmp, "max": max_lmp, "min": min_lmp
                        }
                        # Clear old AI summary so it regenerates with new LMP data
                        st.session_state["node_ai_summary"] = ""

                        k1,k2,k3,k4 = st.columns(4)
                        with k1: metric_card("Latest LMP",  f"${latest_lmp:.2f}", f"as of {latest_time}")
                        with k2: metric_card("2H Avg LMP",  f"${avg_lmp:.2f}",    "$/MWh")
                        with k3: metric_card("2H Max LMP",  f"${max_lmp:.2f}",    "$/MWh")
                        with k4: metric_card("2H Min LMP",  f"${min_lmp:.2f}",    "$/MWh")
                        st.markdown("")

                        fig_rt = go.Figure()
                        fig_rt.add_trace(go.Scatter(
                            x=rt_df[ts_col], y=rt_df[lmp_col],
                            mode="lines+markers",
                            line=dict(color="#00d4ff", width=2),
                            marker=dict(size=5),
                            name="Real-Time LMP",
                            hovertemplate="%{x|%H:%M}<br>LMP: $%{y:.2f}/MWh<extra></extra>"
                        ))
                        fig_rt.add_hline(
                            y=avg_lmp, line_dash="dot", line_color="#5de0a5",
                            annotation_text=f"2H Avg ${avg_lmp:.2f}",
                            annotation_font=dict(color="#5de0a5", size=10)
                        )
                        fig_rt.update_layout(
                            template="plotly_dark",
                            title=f"Real-Time LMP — {sp_clean}  (Last 2 Hours)",
                            xaxis_title="Time (UTC)",
                            yaxis_title="LMP ($/MWh)",
                            height=380,
                            margin=dict(t=50)
                        )
                        st.plotly_chart(fig_rt, use_container_width=True)

                        with st.expander("📄 Raw real-time data"):
                            st.dataframe(rt_df[[ts_col, lmp_col]].rename(
                                columns={ts_col:"Timestamp (UTC)", lmp_col:"LMP ($/MWh)"}),
                                use_container_width=True)

                except requests.exceptions.HTTPError as e:
                    st.error(f"ERCOT API returned an error: {rt_resp.status_code}. "
                             "The settlement point name may not exist or the API is temporarily unavailable.")
                    with st.expander("Debug info"):
                        st.code(rt_resp.text[:500])
                except Exception as e:
                    st.error(f"Could not fetch real-time data: {e}")

        st.markdown("---")
        if df is not None:
            st.markdown('<div class="section-header">Link Node to LMP Analysis</div>', unsafe_allow_html=True)
            link_node = st.selectbox("Select substation to analyse in LMP", sdf["Name"].tolist(), key="node_link")
            if st.button("📈  Go to LMP Analysis for this Node", type="secondary"):
                bus_list  = df["Bus"].unique().tolist()
                keyword   = link_node.split()[0].upper() if link_node else ""
                matched   = [b for b in bus_list if keyword in b.upper()]
                st.session_state.selected_bus = matched[0] if matched else bus_list[0]
                st.info(f"Matched to bus: **{st.session_state.selected_bus}** — switch to 📈 LMP Price Analysis in the sidebar.")

        # ── AI Workflow Summary ───────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 AI Node & LMP Summary</div>', unsafe_allow_html=True)

        # Pull API key from sidebar session
        ai_key = st.session_state.get("api_key", "")

        if not ai_key:
            st.info("🔑 Enter your Anthropic API key in the sidebar to enable the AI summary.")
        else:
            # Build a rich context from everything currently on screen
            nearest_5 = sdf.head(5)[["Name","Type","Voltage","Distance (mi)","Operator"]].to_string(index=False)
            hub_list   = sdf[sdf["Type"]=="Hub"][["Name","Voltage","Distance (mi)"]].head(5).to_string(index=False)
            node_list  = sdf[sdf["Type"]=="Node"][["Name","Voltage","Distance (mi)"]].head(5).to_string(index=False)
            volt_dist  = sdf["Voltage"].value_counts().to_string()

            # Include RT LMP if it was fetched this session
            rt_lmp_context = ""
            if "rt_summary" in st.session_state and st.session_state.rt_summary:
                rt_lmp_context = f"""
Real-Time LMP Data (last fetched):
- Settlement Point: {st.session_state.rt_summary.get('sp','')}
- Latest LMP:  ${st.session_state.rt_summary.get('latest',0):.2f}/MWh
- 2H Average:  ${st.session_state.rt_summary.get('avg',0):.2f}/MWh
- 2H Max:      ${st.session_state.rt_summary.get('max',0):.2f}/MWh
- 2H Min:      ${st.session_state.rt_summary.get('min',0):.2f}/MWh
"""

            screen_context = f"""
You are an expert ERCOT energy market analyst and BESS developer advisor.

The user has just run a substation search on the ERCOT BESS Dashboard.
Summarise what is currently on the screen in a clear, professional manner.

=== SEARCH PARAMETERS ===
Centre coordinates: {lat:.4f}, {lon:.4f}
Search radius: {radius_miles} miles
Hub threshold: ≥ {hub_threshold_kv} kV
Total substations found: {len(sdf)}
  - Hubs:  {n_hub}
  - Nodes: {n_node}

=== NEAREST 5 SUBSTATIONS ===
{nearest_5}

=== NEAREST HUBS (≥{hub_threshold_kv} kV) ===
{hub_list if n_hub > 0 else "None found"}

=== NEAREST NODES (<{hub_threshold_kv} kV) ===
{node_list if n_node > 0 else "None found"}

=== VOLTAGE DISTRIBUTION ===
{volt_dist}
{rt_lmp_context}

=== YOUR TASK ===
Write a structured summary (use short sections with emoji headers) covering:
1. What substations were found and their significance
2. The nearest Hub — its name, voltage, distance, and why it matters for BESS interconnection
3. The nearest Node — its name, voltage, distance, and role in the local grid
4. The voltage mix in this area and what it tells us about grid density
5. If real-time LMP data is available — interpret the current price level and what it signals for BESS dispatch right now
6. A brief developer recommendation — is this a good area for BESS development based on grid infrastructure?

Be specific, use the actual numbers from the data, and keep each section to 2–3 sentences.
"""

            auto_run = st.checkbox("⚡ Auto-generate summary after search", value=True, key="auto_ai")

            if auto_run or st.button("🤖  Generate AI Summary", type="primary", key="gen_ai"):
                with st.spinner("AI is analysing the screen …"):
                    try:
                        ai_r = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": ai_key,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json"
                            },
                            json={
                                "model": "claude-sonnet-4-20250514",
                                "max_tokens": 900,
                                "messages": [{"role": "user", "content": screen_context}]
                            },
                            timeout=30
                        )
                        ai_r.raise_for_status()
                        summary = ai_r.json()["content"][0]["text"]
                        st.session_state["node_ai_summary"] = summary
                    except requests.exceptions.HTTPError:
                        st.error(f"API error {ai_r.status_code}: {ai_r.text[:200]}")
                    except Exception as e:
                        st.error(f"AI summary failed: {e}")

            # Display persisted summary
            if "node_ai_summary" in st.session_state and st.session_state.node_ai_summary:
                st.markdown("""
                <div style="background:#0f1a14;border:1px solid #2a4a35;border-radius:12px;
                            padding:20px 24px;margin-top:10px;">
                """, unsafe_allow_html=True)
                st.markdown(st.session_state.node_ai_summary)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("💡 Summary generated by Claude · based on current screen data")

                col_copy, col_clear = st.columns([1,5])
                with col_copy:
                    if st.button("🗑️ Clear", key="clear_node_ai"):
                        st.session_state.node_ai_summary = ""
                        st.rerun()


# ══════════════════════════════════════════════
#  PAGE 2 – LMP PRICE ANALYSIS
# ══════════════════════════════════════════════
elif page == "📈  LMP Price Analysis":
    st.title("📈  LMP Price Analysis")

    if df is None:
        st.info("👈  Upload an ERCOT LMP CSV from the sidebar to continue.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔍 Single Bus Analysis", "📊 Top N Buses by Spread", "📥 Export Revenue Table"])

    # ═══ TAB 1 – Single Bus ═══════════════════
    with tab1:
        bus_list = sorted(df["Bus"].unique().tolist())
        default_idx = 0
        if st.session_state.selected_bus and st.session_state.selected_bus in bus_list:
            default_idx = bus_list.index(st.session_state.selected_bus)

        bus = st.selectbox("Search & select Bus name", bus_list, index=default_idx)
        st.session_state.selected_bus = bus

        dates = sorted(df["Date"].unique().tolist())
        multi_date = st.checkbox("📅 Compare multiple dates on one chart", value=False)
        if multi_date:
            sel_dates = st.multiselect("Select Dates", dates, default=dates[:min(3,len(dates))])
            if not sel_dates:
                st.warning("Select at least one date.")
                st.stop()
        else:
            sel_date  = st.selectbox("Select Date", dates) if len(dates)>1 else dates[0]
            sel_dates = [sel_date]

        show_2hr = st.checkbox("⚡ 2-Hour Storage  (3-hr rolling avg, ±1 hr windows)", value=True)
        show_4hr = st.checkbox("⚡ 4-Hour Storage  (3-hr rolling avg, ±2 hr windows)", value=True)

        PALETTE = ["#00d4ff","#ff8c42","#5de0a5","#bf7fff","#FFD700","#ff6b6b","#74c0fc"]

        fig = go.Figure()
        summary_rows = []

        for di, date in enumerate(sel_dates):
            bdf = (df[(df["Bus"]==bus) & (df["Date"]==date)]
                   .sort_values("Hour").reset_index(drop=True))
            if bdf.empty:
                continue

            col = PALETTE[di % len(PALETTE)]

            # ── Rolling average & BESS calc ──────
            rev2, roll2, low2, high2, cw2, dw2 = bess_calc(bdf, half_w=1)
            rev4, roll4, low4, high4, cw4, dw4 = bess_calc(bdf, half_w=2)

            min_lmp = bdf["LMP"].min()
            max_lmp = bdf["LMP"].max()
            spread  = max_lmp - min_lmp

            summary_rows.append({
                "Date": date, "Bus": bus,
                "Min LMP": round(min_lmp,2), "Min Hour": low2,
                "Max LMP": round(max_lmp,2), "Max Hour": high2,
                "Spread": round(spread,2),
                "2H Revenue ($/MWh)": rev2,
                "4H Revenue ($/MWh)": rev4,
            })

            # Raw LMP line
            fig.add_trace(go.Scatter(
                x=bdf["Hour"], y=bdf["LMP"],
                name=f"LMP {date}",
                line=dict(color=col, width=2.5),
                hovertemplate=f"Date: {date}<br>Hour %{{x}}<br>LMP: $%{{y:.2f}}<extra></extra>"
            ))

            # BESS overlays — single-date mode only
            if not multi_date:

                # ── Rolling average lines ────────
                if show_2hr:
                    fig.add_trace(go.Scatter(
                        x=bdf["Hour"], y=roll2,
                        name="3-hr Roll Avg (2H basis)",
                        line=dict(color="#00e676", width=1.5, dash="dot"),
                        hovertemplate="Hour %{x}<br>3-hr Avg: $%{y:.2f}<extra></extra>"
                    ))
                if show_4hr and not show_2hr:   # avoid duplicate roll line if both on
                    fig.add_trace(go.Scatter(
                        x=bdf["Hour"], y=roll4,
                        name="3-hr Roll Avg (4H basis)",
                        line=dict(color="#ff8c42", width=1.5, dash="dot"),
                        hovertemplate="Hour %{x}<br>3-hr Avg: $%{y:.2f}<extra></extra>"
                    ))

                # ── 2H shaded windows & band ─────
                if show_2hr:
                    fig.add_vrect(x0=cw2[0], x1=cw2[1],
                                  fillcolor="#00e676", opacity=0.10, line_width=0,
                                  annotation_text=f"Charge 2H\n(Avg low hr {low2})",
                                  annotation_font=dict(size=9, color="#00e676"))
                    fig.add_vrect(x0=dw2[0], x1=dw2[1],
                                  fillcolor="#ff9800", opacity=0.10, line_width=0,
                                  annotation_text=f"Discharge 2H\n(Avg high hr {high2})",
                                  annotation_font=dict(size=9, color="#ff9800"))
                    bdf["_2s"] = bdf["Hour"].apply(
                        lambda h: -1 if cw2[0]<=h<=cw2[1] else (1 if dw2[0]<=h<=dw2[1] else 0))
                    bdf["_2c"] = bdf["LMP"] + bdf["_2s"] * spread * 0.12
                    fig.add_trace(go.Scatter(
                        x=bdf["Hour"], y=bdf["_2c"],
                        name="2H Storage Band",
                        line=dict(color="#00e676", width=2, dash="dashdot", shape="hv"),
                        hovertemplate="Hour %{x}<br>2H Band: $%{y:.2f}<extra></extra>"
                    ))

                # ── 4H shaded windows & band ─────
                if show_4hr:
                    fig.add_vrect(x0=cw4[0], x1=cw4[1],
                                  fillcolor="#00bcd4", opacity=0.07, line_width=0,
                                  annotation_text=f"Charge 4H\n(Avg low hr {low4})",
                                  annotation_font=dict(size=9, color="#00bcd4"))
                    fig.add_vrect(x0=dw4[0], x1=dw4[1],
                                  fillcolor="#ff5722", opacity=0.07, line_width=0,
                                  annotation_text=f"Discharge 4H\n(Avg high hr {high4})",
                                  annotation_font=dict(size=9, color="#ff5722"))
                    bdf["_4s"] = bdf["Hour"].apply(
                        lambda h: -1 if cw4[0]<=h<=cw4[1] else (1 if dw4[0]<=h<=dw4[1] else 0))
                    bdf["_4c"] = bdf["LMP"] + bdf["_4s"] * spread * 0.20
                    fig.add_trace(go.Scatter(
                        x=bdf["Hour"], y=bdf["_4c"],
                        name="4H Storage Band",
                        line=dict(color="#ff8c42", width=2, dash="dash", shape="hv"),
                        hovertemplate="Hour %{x}<br>4H Band: $%{y:.2f}<extra></extra>"
                    ))

                # ── Markers at rolling-avg peaks ─
                if show_2hr:
                    fig.add_trace(go.Scatter(
                        x=[low2], y=[bdf.loc[bdf["Hour"]==low2, "LMP"].values[0]],
                        mode="markers+text",
                        marker=dict(size=13, color="#00e676", symbol="triangle-up"),
                        text=[f"  Charge\n${roll2.min():.1f} avg"],
                        textposition="middle right",
                        textfont=dict(color="#00e676", size=9),
                        name=f"Avg Low hr {low2}",
                        hovertemplate=f"Hour {low2} — Roll Avg Min: ${roll2.min():.2f}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[high2], y=[bdf.loc[bdf["Hour"]==high2, "LMP"].values[0]],
                        mode="markers+text",
                        marker=dict(size=13, color="#ff9800", symbol="triangle-down"),
                        text=[f"  Discharge\n${roll2.max():.1f} avg"],
                        textposition="middle right",
                        textfont=dict(color="#ff9800", size=9),
                        name=f"Avg High hr {high2}",
                        hovertemplate=f"Hour {high2} — Roll Avg Max: ${roll2.max():.2f}<extra></extra>"
                    ))

        fig.update_layout(
            template="plotly_dark",
            title=dict(text=f"LMP & Rolling-Average BESS Strategy — <b>{bus}</b>", font=dict(size=15)),
            xaxis=dict(title="Hour Ending", tickmode="linear", dtick=1,
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="LMP ($/MWh)",
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", y=1.10, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", height=530
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Legend explainer ────────────────────
        with st.expander("ℹ️  How the rolling-average BESS strategy works"):
            st.markdown("""
**Step 1 — Smooth the price curve**
A **3-hour centred rolling average** is calculated across all 24 hours:
`Roll[h] = mean(LMP[h-1], LMP[h], LMP[h+1])`
This removes single-hour price spikes that would be impractical to capture.

**Step 2 — Find the smoothed low & high**
- The hour with the **lowest** rolling average → optimal charge centre
- The hour with the **highest** rolling average → optimal discharge centre

**Step 3 — Set the windows**
| Storage | Charge window | Discharge window |
|---------|--------------|-----------------|
| 2H BESS | ±1 hr around avg-low | ±1 hr around avg-high |
| 4H BESS | ±2 hr around avg-low | ±2 hr around avg-high |

**Revenue** = average LMP during discharge window − average LMP during charge window
            """)

        # KPIs
        if summary_rows:
            sr = summary_rows[0]
            k1,k2,k3,k4,k5 = st.columns(5)
            with k1: metric_card("Bus", bus)
            with k2: metric_card("Lowest LMP",   f"${sr['Min LMP']:.2f}",  f"Hour {sr['Min Hour']}")
            with k3: metric_card("Highest LMP",  f"${sr['Max LMP']:.2f}",  f"Hour {sr['Max Hour']}")
            with k4: metric_card("2H Arbitrage", f"${sr['2H Revenue ($/MWh)']:.2f}", "$/MWh net")
            with k5: metric_card("4H Arbitrage", f"${sr['4H Revenue ($/MWh)']:.2f}", "$/MWh net")
            st.markdown("")
            sp = sr["Spread"]
            if sp > 80:   st.success("✅  Pure Merchant Arbitrage Opportunity")
            elif sp > 40: st.warning("⚠️  Solar + Storage Overbuild Recommended")
            else:         st.error("❌  Low Spread → Capacity / Ancillary Market Focus")

        if multi_date and summary_rows:
            st.markdown("---")
            st.markdown('<div class="section-header">Multi-Date Summary</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # ═══ TAB 2 – Top N Buses ══════════════════
    with tab2:
        st.markdown('<div class="section-header">Top N Buses by LMP Spread</div>', unsafe_allow_html=True)
        dates_all = sorted(df["Date"].unique().tolist())
        c1, c2 = st.columns(2)
        with c1: top_n = st.slider("Show Top N Buses", 5, 50, 15)
        with c2:
            top_date = st.selectbox("For Date", dates_all, key="top_date") if len(dates_all)>1 else dates_all[0]

        filt = df[df["Date"]==top_date]
        bus_stats = (
            filt.groupby("Bus")["LMP"]
            .agg(Min_LMP="min", Max_LMP="max", Avg_LMP="mean",
                 Std_Dev="std", Spread=lambda x: x.max()-x.min())
            .round(2).reset_index()
            .sort_values("Spread", ascending=False)
            .head(top_n)
        )
        rev_rows = []
        for _, row in bus_stats.iterrows():
            bdf_t = filt[filt["Bus"]==row["Bus"]].sort_values("Hour").reset_index(drop=True)
            r2, _, lh2, hh2, _, _ = bess_calc(bdf_t, 1)
            r4, _, lh4, hh4, _, _ = bess_calc(bdf_t, 2)
            rev_rows.append({"Bus":row["Bus"],"2H Rev":r2,"4H Rev":r4,
                             "Low Hr":lh2,"High Hr":hh2})
        top_df = bus_stats.merge(pd.DataFrame(rev_rows), on="Bus")

        fig_top = px.bar(
            top_df, x="Bus", y="Spread", color="Spread",
            color_continuous_scale=["#f07070","#f0c040","#5de0a5"],
            labels={"Spread":"LMP Spread ($/MWh)"}, template="plotly_dark",
            title=f"Top {top_n} Buses by LMP Spread — {top_date}"
        )
        fig_top.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig_top, use_container_width=True)

        fig_sc = px.scatter(
            top_df, x="Spread", y="4H Rev", text="Bus", size="Spread",
            color="4H Rev", color_continuous_scale=["#4fc3f7","#bf7fff","#ff8c42"],
            template="plotly_dark", title="Spread vs 4H BESS Revenue (Rolling-Avg Strategy)",
            labels={"Spread":"LMP Spread ($/MWh)","4H Rev":"4H Revenue ($/MWh)"}
        )
        fig_sc.update_traces(textposition="top center", textfont=dict(size=8))
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

        def _style_spread(val):
            if val>80:  return "background-color:#1f4b2e;color:#5de0a5"
            if val>40:  return "background-color:#3d3510;color:#f0c040"
            return "background-color:#2e1a1a;color:#f07070"
        st.dataframe(
            top_df.style.applymap(_style_spread, subset=["Spread"]),
            use_container_width=True, height=320
        )
        if not top_df.empty:
            best_bus = top_df.iloc[0]["Bus"]
            if st.button(f"📈  Analyse top bus: {best_bus}"):
                st.session_state.selected_bus = best_bus
                st.info(f"Switched to **{best_bus}** — click the 'Single Bus Analysis' tab.")

    # ═══ TAB 3 – Export ═══════════════════════
    with tab3:
        st.markdown('<div class="section-header">Export BESS Revenue for All Buses</div>', unsafe_allow_html=True)
        dates_exp = sorted(df["Date"].unique().tolist())
        exp_date  = st.selectbox("Select Date for Export", dates_exp, key="exp_date") if len(dates_exp)>1 else dates_exp[0]

        if st.button("⚙️  Compute Revenue for All Buses", type="primary"):
            with st.spinner("Calculating rolling-average BESS revenue for all buses …"):
                exp_rows = []
                for bus_name in df["Bus"].unique():
                    bdf_e = (df[(df["Bus"]==bus_name) & (df["Date"]==exp_date)]
                             .sort_values("Hour").reset_index(drop=True))
                    if len(bdf_e) < 3:
                        continue
                    r2, _, lh, hh, cw2, dw2 = bess_calc(bdf_e, 1)
                    r4, _, _,  _,  cw4, dw4 = bess_calc(bdf_e, 2)
                    sp = round(bdf_e["LMP"].max() - bdf_e["LMP"].min(), 2)
                    rec = ("Merchant Arbitrage" if sp>80
                           else ("Solar + Storage Overbuild" if sp>40 else "Ancillary / Capacity"))
                    exp_rows.append({
                        "Date": exp_date, "Bus": bus_name,
                        "Min LMP": round(bdf_e["LMP"].min(),2),
                        "Max LMP": round(bdf_e["LMP"].max(),2),
                        "Avg LMP": round(bdf_e["LMP"].mean(),2),
                        "Spread ($/MWh)": sp,
                        "2H Charge Window":    f"Hr {cw2[0]:.0f}–{cw2[1]:.0f}",
                        "2H Discharge Window": f"Hr {dw2[0]:.0f}–{dw2[1]:.0f}",
                        "4H Charge Window":    f"Hr {cw4[0]:.0f}–{cw4[1]:.0f}",
                        "4H Discharge Window": f"Hr {dw4[0]:.0f}–{dw4[1]:.0f}",
                        "2H Revenue ($/MWh)": r2,
                        "4H Revenue ($/MWh)": r4,
                        "Recommended Strategy": rec,
                    })

            exp_df = (pd.DataFrame(exp_rows)
                      .sort_values("Spread ($/MWh)", ascending=False)
                      .reset_index(drop=True))
            st.success(f"✅  Computed {len(exp_df)} buses")
            st.dataframe(exp_df, use_container_width=True, height=400)
            st.download_button(
                label="📥  Download CSV",
                data=exp_df.to_csv(index=False).encode("utf-8"),
                file_name=f"BESS_Revenue_{exp_date}.csv",
                mime="text/csv", type="primary"
            )


# ══════════════════════════════════════════════
#  PAGE 3 – AI COPILOT
# ══════════════════════════════════════════════
elif page == "🤖  AI Copilot":
    st.title("🤖  AI Copilot")
    st.caption("Ask questions about your LMP data. Powered by Claude (Anthropic).")

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔑 Anthropic API Key")
        api_key = st.text_input("Enter API Key", type="password", key="api_key",
                                help="Get your key at console.anthropic.com")
        st.markdown("*Used for AI Copilot & Node AI Summary*")

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to activate the AI Copilot.")
        st.markdown("""
        **How to get a key:**
        1. Go to [console.anthropic.com](https://console.anthropic.com)
        2. Sign up / log in → **API Keys** → Create new key
        3. Paste it in the sidebar
        """)
        st.stop()

    if df is None:
        st.info("👈  Upload an ERCOT LMP CSV from the sidebar first.")
        st.stop()

    # Build data context
    dates_ai   = sorted(df["Date"].unique().tolist())
    bus_list_ai = sorted(df["Bus"].unique().tolist())
    top_buses_ai = (
        df.groupby("Bus")["LMP"]
        .agg(spread=lambda x: x.max()-x.min(), avg="mean", max="max", min="min")
        .sort_values("spread", ascending=False)
        .head(10).round(2).reset_index()
    )
    data_context = f"""
You are an expert ERCOT energy market analyst and BESS (Battery Energy Storage System) strategy advisor.

The user has uploaded an ERCOT LMP dataset:
- Dates: {", ".join(dates_ai)}
- Total buses/nodes: {len(bus_list_ai)}
- Total records: {len(df):,}
- Overall avg LMP: ${df['LMP'].mean():.2f}/MWh
- Overall max LMP: ${df['LMP'].max():.2f}/MWh  (Bus: {df.loc[df['LMP'].idxmax(),'Bus']})
- Overall min LMP: ${df['LMP'].min():.2f}/MWh  (Bus: {df.loc[df['LMP'].idxmin(),'Bus']})

Top 10 buses by LMP Spread:
{top_buses_ai.to_string(index=False)}

BESS Strategy used in this dashboard:
- A 3-hour centred rolling average smooths the price curve before finding optimal charge/discharge hours.
- 2H BESS: charges ±1 hr around smoothed price minimum; discharges ±1 hr around smoothed maximum.
- 4H BESS: charges ±2 hr around smoothed price minimum; discharges ±2 hr around smoothed maximum.
- Revenue = avg discharge LMP − avg charge LMP.

Strategy thresholds:
- Spread > $80/MWh  → Pure Merchant Arbitrage
- Spread $40–80/MWh → Solar + Storage Overbuild
- Spread < $40/MWh  → Capacity / Ancillary Market Focus

Answer concisely and specifically using numbers from the dataset. Focus on actionable BESS development insights.
"""

    st.markdown('<div class="section-header">Suggested Questions</div>', unsafe_allow_html=True)
    suggestions = [
        "Which buses have the best arbitrage opportunity?",
        "What is the recommended BESS strategy for the top bus?",
        "Summarise the overall LMP price trends in this dataset",
        "Which hours have the highest and lowest average prices?",
        "Compare 2H vs 4H storage economics for the top nodes",
        "What does the spread distribution suggest about market volatility?",
    ]
    q_cols = st.columns(3)
    for i, q in enumerate(suggestions):
        with q_cols[i % 3]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})

    st.markdown("---")

    user_input = st.chat_input("Ask anything about your ERCOT LMP data …")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})

    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if msg["role"]=="user" else "assistant",
                             avatar="👤" if msg["role"]=="user" else "🤖"):
            st.markdown(msg["content"])

        if msg["role"] == "user" and i == len(st.session_state.chat_history)-1:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analysing …"):
                    msgs_payload = [{"role":"user","content": data_context+"\n\n---\n\nUser question: "+st.session_state.chat_history[0]["content"]}]
                    for h in st.session_state.chat_history[1:]:
                        msgs_payload.append({"role":h["role"],"content":h["content"]})
                    try:
                        ai_resp = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"x-api-key":api_key,"anthropic-version":"2023-06-01","content-type":"application/json"},
                            json={"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":msgs_payload},
                            timeout=30
                        )
                        ai_resp.raise_for_status()
                        answer = ai_resp.json()["content"][0]["text"]
                    except requests.exceptions.HTTPError:
                        answer = f"❌ API error {ai_resp.status_code}: {ai_resp.text}"
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role":"assistant","content":answer})

    if st.session_state.chat_history:
        if st.button("🗑️  Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
