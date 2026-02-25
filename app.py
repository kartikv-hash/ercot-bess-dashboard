
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import math
import io
import google.generativeai as genai

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="ERCOT BESS Dashboard", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
for k, v in {"active_page": "🗺️  Node Analyser", "linked_bus": None, "lmp_df": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .card { background:#1a1d2e; border:1px solid #2a2d45; border-radius:12px; padding:16px 20px; }
  .card-title { color:#7880a8; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
  .card-value { color:#ffffff; font-size:22px; font-weight:700; margin-top:4px; }
  .card-sub   { color:#5de0a5; font-size:11px; margin-top:3px; }
  .section-header { font-size:12px; font-weight:600; color:#7880a8;
      text-transform:uppercase; letter-spacing:1px; margin:18px 0 10px 0; }
  .copilot-box { background:linear-gradient(135deg,#0d1117,#1a1d2e);
      border:1px solid #4f6ef7; border-radius:16px; padding:22px 26px; margin-top:20px; }
  .copilot-title { color:#4f6ef7; font-size:14px; font-weight:700; margin-bottom:10px; }
  .copilot-text  { color:#c8cfe8; font-size:13px; line-height:1.7; }
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
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return round(R*2*math.asin(math.sqrt(a)), 2)

KV_MAP = {"34.5 kV":34500,"69 kV":69000,"115 kV":115000,"138 kV":138000,
          "230 kV":230000,"345 kV":345000,"500 kV":500000,"765 kV":765000}

def nearest_kv_label(volts):
    best, best_diff = None, float("inf")
    for label, nominal in KV_MAP.items():
        diff = abs(volts-nominal)/nominal
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
        if col not in raw.columns: raw[col] = None
    if raw["Hour"].dtype == object:
        raw["Hour"] = raw["Hour"].astype(str).str.strip().str.extract(r"(\d+)")[0].astype(float)
    raw["LMP"]  = pd.to_numeric(raw["LMP"],  errors="coerce")
    raw["Date"] = raw["Date"].astype(str).str.strip()
    raw["Bus"]  = raw["Bus"].astype(str).str.strip()
    return raw.dropna(subset=["Hour","LMP"]).reset_index(drop=True)

def gemini_copilot(context: str, question: str = "") -> str:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return "⚠️ Add `GEMINI_API_KEY` to your Streamlit secrets to enable the Copilot."
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are an expert ERCOT energy market analyst embedded in a BESS (Battery Energy Storage System) developer dashboard.

Dashboard context:
{context}

{f"User question: {question}" if question else "Provide a concise, actionable analysis of what the data shows. Cover: market conditions, BESS opportunity quality, key risks, and one concrete recommendation. Be direct and professional. Under 220 words."}"""
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini error: {e}"

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ERCOT BESS")
    st.markdown("---")
    page = st.radio("Module", ["🗺️  Node Analyser", "📈  LMP Price Analysis"],
                    index=["🗺️  Node Analyser","📈  LMP Price Analysis"].index(st.session_state.active_page),
                    label_visibility="collapsed")
    st.session_state.active_page = page
    st.markdown("---")
    st.markdown("### 📂 LMP Data")
    uploaded = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        st.session_state.lmp_df = load_lmp(uploaded)
        st.success(f"✅ {len(st.session_state.lmp_df):,} rows loaded")


# ══════════════════════════════════════════════
#  PAGE 1 – NODE ANALYSER
# ══════════════════════════════════════════════
if page == "🗺️  Node Analyser":
    st.title("🗺️  Hub & Node Analyser")
    st.caption("Hubs = substations at or above your chosen threshold kV  |  Nodes = below threshold  |  Data: OpenStreetMap")

    st.markdown('<div class="section-header">Search Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2,2,3])
    with c1:
        lat = st.number_input("Latitude",  value=33.7944, format="%.4f")
        lon = st.number_input("Longitude", value=-98.5706, format="%.4f")
    with c2:
        radius_miles     = st.selectbox("Radius (miles)", [5,10,25,50,100,150], index=2)
        hub_threshold_kv = st.selectbox("Hub threshold (kV ≥)", [115,138,230,345], index=2)
    with c3:
        selected_kv_labels = st.multiselect("Filter Voltages", list(KV_MAP.keys()), default=list(KV_MAP.keys()))
        show_unknown_v = st.checkbox("Include unknown voltage", value=True)

    zoom_guess = max(7, min(14, round(14 - math.log2(max(1,radius_miles)))))
    st.markdown(f'🔗 **[Open in OpenInfraMap](https://openinframap.org/#{zoom_guess}/{lat:.4f}/{lon:.4f})**', unsafe_allow_html=True)

    run_search = st.button("🔍  Search Substations", type="primary")

    if run_search:
        radius_m = radius_miles * 1609.34
        hub_threshold_v = hub_threshold_kv * 1000
        selected_nominal_volts = {KV_MAP[k] for k in selected_kv_labels}
        overpass_query = f"""[out:json][timeout:40];
(node["power"="substation"](around:{radius_m:.0f},{lat},{lon});
 way["power"="substation"](around:{radius_m:.0f},{lat},{lon});
 relation["power"="substation"](around:{radius_m:.0f},{lat},{lon}););
out center tags;"""

        with st.spinner("Querying OpenStreetMap …"):
            try:
                resp = requests.post("https://overpass-api.de/api/interpreter",
                                     data={"data": overpass_query}, timeout=45)
                resp.raise_for_status()
                osm_data = resp.json()
            except requests.exceptions.Timeout:
                st.error("Overpass API timed out. Try a smaller radius."); st.stop()
            except Exception as e:
                st.error(f"API error: {e}"); st.stop()

        rows = []
        for el in osm_data.get("elements", []):
            tags = el.get("tags", {})
            if el["type"] == "node": slat, slon = el.get("lat"), el.get("lon")
            elif "center" in el:     slat, slon = el["center"]["lat"], el["center"]["lon"]
            else: continue
            if slat is None or slon is None: continue

            volt_raw = tags.get("voltage","")
            volts = None
            if volt_raw:
                try: volts = max(int(v) for v in volt_raw.split(";") if v.strip().isdigit())
                except: pass

            if volts is not None:
                nearest = nearest_kv_label(volts)
                nom = KV_MAP.get(nearest)
                if nom and nom not in selected_nominal_volts: continue
                kv_label = nearest
            else:
                if not show_unknown_v: continue
                kv_label = "Unknown"

            sub_type = tags.get("substation","")
            classification = "Hub" if (sub_type == "transmission" or (volts and volts >= hub_threshold_v)) else "Node"

            rows.append({"Name": tags.get("name", f"Substation {el['id']}"),
                         "Type": classification, "Voltage": kv_label,
                         "Operator": tags.get("operator","—"),
                         "Substation Tag": sub_type if sub_type else "—",
                         "Lat": round(slat,5), "Lon": round(slon,5),
                         "Distance (mi)": haversine_miles(lat,lon,slat,slon)})

        if not rows:
            st.warning("No substations found. Try a larger radius or loosen voltage filters."); st.stop()

        sdf = pd.DataFrame(rows).sort_values("Distance (mi)").reset_index(drop=True)
        n_hub  = (sdf["Type"]=="Hub").sum()
        n_node = (sdf["Type"]=="Node").sum()

        # ── KPIs ────────────────────────────
        st.markdown("---")
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1: metric_card("Total Found", str(len(sdf)))
        with k2: metric_card("Hubs",   str(n_hub),  f"≥ {hub_threshold_kv} kV")
        with k3: metric_card("Nodes",  str(n_node), f"< {hub_threshold_kv} kV")
        with k4: metric_card("Radius", f"{radius_miles} mi")
        with k5: metric_card("Centre", f"{lat:.3f}, {lon:.3f}")
        st.markdown("")

        # ── Map ─────────────────────────────
        fig_map = go.Figure()
        circle_lats, circle_lons = [], []
        for deg in range(0,361,4):
            rad = math.radians(deg)
            dlat = (radius_miles/3958.8)*math.cos(rad)
            dlon = (radius_miles/3958.8)*math.sin(rad)/math.cos(math.radians(lat))
            circle_lats.append(lat+math.degrees(dlat))
            circle_lons.append(lon+math.degrees(dlon))
        fig_map.add_trace(go.Scattermapbox(lat=circle_lats, lon=circle_lons,
            mode="lines", line=dict(color="rgba(255,215,0,0.35)",width=1),
            name=f"{radius_miles} mi radius", hoverinfo="skip"))
        fig_map.add_trace(go.Scattermapbox(lat=[lat], lon=[lon], mode="markers",
            marker=dict(size=16,color="#FFD700"), name="Centre",
            hovertext=f"Centre: {lat:.4f},{lon:.4f}", hoverinfo="text"))
        hubs_df  = sdf[sdf["Type"]=="Hub"]
        nodes_df = sdf[sdf["Type"]=="Node"]
        if len(hubs_df):
            fig_map.add_trace(go.Scattermapbox(lat=hubs_df["Lat"], lon=hubs_df["Lon"],
                mode="markers", marker=dict(size=13,color="#bf7fff"), name="Hub",
                hovertemplate="<b>%{customdata[0]}</b><br>Hub<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]} mi<extra></extra>",
                customdata=hubs_df[["Name","Voltage","Operator","Distance (mi)"]].values))
        if len(nodes_df):
            fig_map.add_trace(go.Scattermapbox(lat=nodes_df["Lat"], lon=nodes_df["Lon"],
                mode="markers", marker=dict(size=9,color="#4fc3f7"), name="Node",
                hovertemplate="<b>%{customdata[0]}</b><br>Node<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]} mi<extra></extra>",
                customdata=nodes_df[["Name","Voltage","Operator","Distance (mi)"]].values))
        fig_map.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=lat,lon=lon),
                        zoom=max(7,min(12,round(12-math.log2(max(1,radius_miles)))))),
            margin=dict(l=0,r=0,t=0,b=0), height=500,
            legend=dict(bgcolor="rgba(20,22,38,0.85)",font=dict(color="white"),x=0.01,y=0.99))
        st.plotly_chart(fig_map, use_container_width=True)

        # ── Voltage Distribution Bar ─────────
        st.markdown('<div class="section-header">Voltage Distribution</div>', unsafe_allow_html=True)
        v_counts = sdf.groupby(["Voltage","Type"]).size().reset_index(name="Count")
        fig_vbar = px.bar(v_counts, x="Voltage", y="Count", color="Type",
                          color_discrete_map={"Hub":"#bf7fff","Node":"#4fc3f7"},
                          template="plotly_dark", barmode="stack")
        fig_vbar.update_layout(height=280, margin=dict(t=20,b=20))
        st.plotly_chart(fig_vbar, use_container_width=True)

        # ── Results Table ────────────────────
        st.markdown('<div class="section-header">Substation List</div>', unsafe_allow_html=True)
        t1, t2 = st.columns([3,1])
        with t1:
            v_filter = st.multiselect("Filter by voltage", sorted(sdf["Voltage"].unique()),
                                      default=sorted(sdf["Voltage"].unique()), key="node_v_filter")
        with t2:
            type_filter = st.selectbox("Type", ["All","Hub","Node"])

        disp = sdf[sdf["Voltage"].isin(v_filter)]
        if type_filter != "All": disp = disp[disp["Type"]==type_filter]

        def _style_type(val):
            return ("background-color:#3b1f5e;color:#bf7fff;font-weight:600" if val=="Hub"
                    else "background-color:#1a3550;color:#4fc3f7;font-weight:600")

        st.dataframe(disp.drop(columns=[c for c in disp.columns if c.startswith("_")])
                     .style.applymap(_style_type, subset=["Type"]),
                     use_container_width=True, height=320)
        st.caption(f"Showing {len(disp)} of {len(sdf)} substations")

        # ── Link to LMP ──────────────────────
        if st.session_state.lmp_df is not None:
            st.markdown('<div class="section-header">🔗 Jump to LMP Analysis</div>', unsafe_allow_html=True)
            named = disp[disp["Name"] != disp["Name"].str.extract(r"(Substation \d+)", expand=False).fillna("")]["Name"].tolist()
            lmp_buses = st.session_state.lmp_df["Bus"].unique().tolist()
            jump_bus = st.selectbox("Select a substation to analyse in LMP →", ["— select —"] + lmp_buses)
            if jump_bus != "— select —":
                if st.button(f"📈  Open LMP Analysis for {jump_bus}", type="primary"):
                    st.session_state.linked_bus = jump_bus
                    st.session_state.active_page = "📈  LMP Price Analysis"
                    st.rerun()

        # ── Gemini Copilot ───────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 Gemini Copilot</div>', unsafe_allow_html=True)
        kv_breakdown = sdf.groupby("Voltage").size().to_dict()
        node_context = f"""
Location: {lat:.4f}, {lon:.4f}
Search radius: {radius_miles} miles
Total substations found: {len(sdf)}
Hubs (≥{hub_threshold_kv} kV): {n_hub}
Nodes (<{hub_threshold_kv} kV): {n_node}
Voltage breakdown: {kv_breakdown}
Closest substation: {sdf.iloc[0]['Name']} ({sdf.iloc[0]['Type']}, {sdf.iloc[0]['Voltage']}, {sdf.iloc[0]['Distance (mi)']} mi)
Operators present: {', '.join(sdf['Operator'].unique()[:5].tolist())}
"""
        with st.expander("💬 Ask the Copilot about this area", expanded=True):
            user_q = st.text_input("Ask a question (or leave blank for auto-summary)", key="node_q",
                                   placeholder="e.g. Is this a good area for a BESS project? What voltage should I interconnect at?")
            if st.button("✨ Generate Insight", key="node_gen"):
                with st.spinner("Gemini is analysing …"):
                    answer = gemini_copilot(node_context, user_q)
                st.markdown(f'<div class="copilot-box"><div class="copilot-title">⚡ Copilot Analysis</div><div class="copilot-text">{answer}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 2 – LMP PRICE ANALYSIS
# ══════════════════════════════════════════════
elif page == "📈  LMP Price Analysis":
    st.title("📈  LMP Price Analysis")
    st.caption("Upload an ERCOT LMP CSV via the sidebar, then explore bus-level price curves and BESS arbitrage strategy.")

    df = st.session_state.lmp_df
    if df is None:
        st.info("👈  Upload an ERCOT LMP CSV from the sidebar to get started."); st.stop()

    # ── TABS ────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊  Bus Analysis", "🏆  Top Buses by Spread", "💾  Revenue Export"])

    # ════════════════════════════════
    #  TAB 1 – BUS ANALYSIS
    # ════════════════════════════════
    with tab1:
        bus_list = sorted(df["Bus"].unique().tolist())
        default_bus = st.session_state.linked_bus if st.session_state.linked_bus in bus_list else bus_list[0]
        if st.session_state.linked_bus:
            st.session_state.linked_bus = None  # clear after use

        bus = st.selectbox("Search & select Bus", bus_list,
                           index=bus_list.index(default_bus), help="Type to search")

        dates = sorted(df["Date"].unique().tolist())
        if len(dates) > 1:
            st.markdown('<div class="section-header">Date Selection</div>', unsafe_allow_html=True)
            d1, d2 = st.columns([2,3])
            with d1:
                date_mode = st.radio("Mode", ["Single date","Multi-date overlay"], horizontal=True)
            with d2:
                if date_mode == "Single date":
                    selected_dates = [st.selectbox("Date", dates)]
                else:
                    selected_dates = st.multiselect("Dates to overlay", dates, default=dates[:min(4,len(dates))])
        else:
            date_mode = "Single date"
            selected_dates = dates

        if not selected_dates:
            st.warning("Select at least one date."); st.stop()

        # BESS toggles
        st.markdown('<div class="section-header">BESS Overlay</div>', unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        show_2hr = bc1.checkbox("⚡ 2-Hour Storage", value=True)
        show_4hr = bc2.checkbox("⚡ 4-Hour Storage", value=True)

        # ── Build chart ─────────────────────
        fig = go.Figure()
        palette = ["#00d4ff","#ff8c42","#a78bfa","#34d399","#f472b6","#fbbf24"]
        summary_rows = []

        for idx, sel_date in enumerate(selected_dates):
            bdf = df[(df["Bus"]==bus) & (df["Date"]==sel_date)].sort_values("Hour").reset_index(drop=True)
            if bdf.empty: continue
            color = palette[idx % len(palette)]

            low_i  = bdf["LMP"].idxmin()
            high_i = bdf["LMP"].idxmax()
            low_hr  = bdf.loc[low_i,  "Hour"]
            high_hr = bdf.loc[high_i, "Hour"]
            min_lmp = bdf["LMP"].min()
            max_lmp = bdf["LMP"].max()
            spread  = max_lmp - min_lmp

            def win(center, hw): return (max(bdf["Hour"].min(),center-hw), min(bdf["Hour"].max(),center+hw))
            c2, d2 = win(low_hr,1), win(high_hr,1)
            c4, d4 = win(low_hr,2), win(high_hr,2)

            def net_rev(cw, dw):
                ca = bdf[(bdf["Hour"]>=cw[0])&(bdf["Hour"]<=cw[1])]["LMP"].mean()
                da = bdf[(bdf["Hour"]>=dw[0])&(bdf["Hour"]<=dw[1])]["LMP"].mean()
                return round(da-ca, 2)

            rev2 = net_rev(c2,d2)
            rev4 = net_rev(c4,d4)
            summary_rows.append({"Date":sel_date,"Min LMP":round(min_lmp,2),
                                  "Low Hour":low_hr,"Max LMP":round(max_lmp,2),
                                  "High Hour":high_hr,"Spread":round(spread,2),
                                  "2H Rev ($/MWh)":rev2,"4H Rev ($/MWh)":rev4})

            suffix = f" ({sel_date})" if len(selected_dates)>1 else ""
            fig.add_trace(go.Scatter(x=bdf["Hour"], y=bdf["LMP"],
                name=f"LMP{suffix}", line=dict(color=color,width=2.5),
                hovertemplate="Hour %{x}<br>LMP: $%{y:.2f}<extra></extra>"))

            # Only shade on single-date or first overlay
            if idx == 0:
                def shade(x0,x1,c,lbl,op=0.10):
                    fig.add_vrect(x0=x0,x1=x1,fillcolor=c,opacity=op,line_width=0,
                                  annotation_text=lbl,annotation_position="top left",
                                  annotation_font=dict(size=9,color=c))
                if show_2hr:
                    shade(*c2,"#00e676","CHG 2H"); shade(*d2,"#ff9800","DIS 2H")
                if show_4hr:
                    shade(*c4,"#00bcd4","CHG 4H",0.07); shade(*d4,"#ff5722","DIS 4H",0.07)

            # Storage bands (single date only to avoid clutter)
            if len(selected_dates) == 1:
                band2, band4 = spread*0.12, spread*0.20
                if show_2hr:
                    bdf["_s2"] = bdf["Hour"].apply(lambda h: -1 if c2[0]<=h<=c2[1] else (1 if d2[0]<=h<=d2[1] else 0))
                    fig.add_trace(go.Scatter(x=bdf["Hour"], y=bdf["LMP"]+bdf["_s2"]*band2,
                        name="2H Band", line=dict(color="#00e676",width=2,dash="dot",shape="hv"),
                        hovertemplate="Hour %{x}<br>2H: $%{y:.2f}<extra></extra>"))
                if show_4hr:
                    bdf["_s4"] = bdf["Hour"].apply(lambda h: -1 if c4[0]<=h<=c4[1] else (1 if d4[0]<=h<=d4[1] else 0))
                    fig.add_trace(go.Scatter(x=bdf["Hour"], y=bdf["LMP"]+bdf["_s4"]*band4,
                        name="4H Band", line=dict(color="#ff8c42",width=2,dash="dash",shape="hv"),
                        hovertemplate="Hour %{x}<br>4H: $%{y:.2f}<extra></extra>"))

            # Min/Max markers
            fig.add_trace(go.Scatter(x=[low_hr], y=[min_lmp], mode="markers",
                marker=dict(size=12,color="#00e676",symbol="triangle-up"),
                name=f"Low ${min_lmp:.2f}{suffix}", showlegend=True,
                hovertemplate=f"Hour {low_hr} — Min: ${min_lmp:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[high_hr], y=[max_lmp], mode="markers",
                marker=dict(size=12,color="#ff8c42",symbol="triangle-down"),
                name=f"High ${max_lmp:.2f}{suffix}", showlegend=True,
                hovertemplate=f"Hour {high_hr} — Max: ${max_lmp:.2f}<extra></extra>"))

        title_suffix = f"{selected_dates[0]}" if len(selected_dates)==1 else f"{len(selected_dates)} dates overlaid"
        fig.update_layout(template="plotly_dark",
            title=dict(text=f"LMP Intraday — <b>{bus}</b>  |  {title_suffix}", font=dict(size=14)),
            xaxis=dict(title="Hour Ending", tickmode="linear", dtick=1,
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="LMP ($/MWh)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h",y=1.10,x=0,bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
            hovermode="x unified", height=480)
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary Table ────────────────────
        if summary_rows:
            sdf_sum = pd.DataFrame(summary_rows)
            st.markdown('<div class="section-header">Daily Summary</div>', unsafe_allow_html=True)
            st.dataframe(sdf_sum, use_container_width=True)

            # Strategy recommendation (last date)
            spread_val = sdf_sum["Spread"].iloc[-1]
            if spread_val > 80:   st.success("✅  Pure Merchant Arbitrage Opportunity")
            elif spread_val > 40: st.warning("⚠️  Solar + Storage Overbuild Recommended")
            else:                 st.error("❌  Low Spread → Capacity / Ancillary Market Focus")

        # ── Gemini Copilot ───────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 Gemini Copilot</div>', unsafe_allow_html=True)
        if summary_rows:
            r = summary_rows[-1]
            lmp_context = f"""
Bus: {bus}
Date(s): {', '.join(selected_dates)}
Lowest LMP: ${r['Min LMP']}/MWh at Hour {r['Low Hour']}
Highest LMP: ${r['Max LMP']}/MWh at Hour {r['High Hour']}
LMP Spread: ${r['Spread']}/MWh
2-Hour BESS arbitrage (net spread): ${r['2H Rev ($/MWh)']}/MWh
4-Hour BESS arbitrage (net spread): ${r['4H Rev ($/MWh)']}/MWh
Charge window 2H: Hour {r['Low Hour']-1} to {r['Low Hour']+1}
Discharge window 2H: Hour {r['High Hour']-1} to {r['High Hour']+1}
Charge window 4H: Hour {r['Low Hour']-2} to {r['Low Hour']+2}
Discharge window 4H: Hour {r['High Hour']-2} to {r['High Hour']+2}
"""
            with st.expander("💬 Ask the Copilot about this Bus", expanded=True):
                user_q2 = st.text_input("Ask a question (or leave blank for auto-summary)", key="lmp_q",
                                        placeholder="e.g. Is 4H storage better than 2H here? What's driving the price spike?")
                if st.button("✨ Generate Insight", key="lmp_gen"):
                    with st.spinner("Gemini is analysing …"):
                        answer2 = gemini_copilot(lmp_context, user_q2)
                    st.markdown(f'<div class="copilot-box"><div class="copilot-title">⚡ Copilot Analysis — {bus}</div><div class="copilot-text">{answer2}</div></div>', unsafe_allow_html=True)

    # ════════════════════════════════
    #  TAB 2 – TOP BUSES BY SPREAD
    # ════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Top Buses by LMP Spread</div>', unsafe_allow_html=True)
        t2c1, t2c2 = st.columns([2,3])
        with t2c1:
            top_n = st.slider("Show top N buses", 5, 50, 20)
            date_for_top = st.selectbox("For date", sorted(df["Date"].unique()), key="top_date")
        with t2c2:
            min_spread_filter = st.slider("Min spread threshold ($/MWh)", 0, 200, 0)

        day_df = df[df["Date"] == date_for_top]
        bus_stats = (day_df.groupby("Bus")["LMP"]
                     .agg(Min_LMP="min", Max_LMP="max",
                          Spread=lambda x: round(x.max()-x.min(),2),
                          Avg_LMP="mean", Std_Dev="std")
                     .round(2).reset_index()
                     .sort_values("Spread", ascending=False))
        bus_stats = bus_stats[bus_stats["Spread"] >= min_spread_filter].head(top_n)

        # Colour-code strategy column
        def strategy_label(s):
            if s > 80:  return "Merchant Arbitrage ✅"
            if s > 40:  return "Solar + Storage ⚠️"
            return "Ancillary / Capacity ❌"
        bus_stats["Strategy"] = bus_stats["Spread"].apply(strategy_label)

        # Bar chart
        fig_top = px.bar(bus_stats, x="Bus", y="Spread", color="Spread",
                         color_continuous_scale=["#f07070","#f0c040","#5de0a5"],
                         template="plotly_dark", hover_data=["Min_LMP","Max_LMP","Avg_LMP"],
                         title=f"Top {top_n} Buses by LMP Spread — {date_for_top}")
        fig_top.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False,
                              height=380, margin=dict(b=120))
        st.plotly_chart(fig_top, use_container_width=True)

        def _col_spread(val):
            if val > 80: return "background-color:#1f4b2e;color:#5de0a5"
            if val > 40: return "background-color:#3d3510;color:#f0c040"
            return "background-color:#2e1a1a;color:#f07070"

        st.dataframe(bus_stats.style.applymap(_col_spread, subset=["Spread"]),
                     use_container_width=True, height=320)

        # Quick jump to Bus Analysis
        jump2 = st.selectbox("📈  Jump to Bus Analysis →",
                              ["— select —"] + bus_stats["Bus"].tolist(), key="top_jump")
        if jump2 != "— select —":
            if st.button(f"Open {jump2} in Bus Analysis", type="primary", key="top_jump_btn"):
                st.session_state.linked_bus = jump2
                st.rerun()

    # ════════════════════════════════
    #  TAB 3 – REVENUE EXPORT
    # ════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">BESS Revenue Summary – All Buses</div>', unsafe_allow_html=True)
        ex_date = st.selectbox("Select date for export", sorted(df["Date"].unique()), key="ex_date")
        gen_btn = st.button("⚙️  Compute Revenue for All Buses", type="primary")

        if gen_btn:
            export_rows = []
            exp_df = df[df["Date"] == ex_date]
            bus_list_ex = exp_df["Bus"].unique()
            prog = st.progress(0)

            for i, b in enumerate(bus_list_ex):
                bdf_ex = exp_df[exp_df["Bus"]==b].sort_values("Hour").reset_index(drop=True)
                if len(bdf_ex) < 4: continue
                li  = bdf_ex["LMP"].idxmin()
                hi  = bdf_ex["LMP"].idxmax()
                lh  = bdf_ex.loc[li,"Hour"]
                hh  = bdf_ex.loc[hi,"Hour"]
                mn  = bdf_ex["LMP"].min()
                mx  = bdf_ex["LMP"].max()
                sp  = round(mx-mn, 2)

                def net(cw,dw):
                    ca = bdf_ex[(bdf_ex["Hour"]>=cw[0])&(bdf_ex["Hour"]<=cw[1])]["LMP"].mean()
                    da = bdf_ex[(bdf_ex["Hour"]>=dw[0])&(bdf_ex["Hour"]<=dw[1])]["LMP"].mean()
                    return round(da-ca,2)

                r2 = net((max(bdf_ex["Hour"].min(),lh-1),min(bdf_ex["Hour"].max(),lh+1)),
                         (max(bdf_ex["Hour"].min(),hh-1),min(bdf_ex["Hour"].max(),hh+1)))
                r4 = net((max(bdf_ex["Hour"].min(),lh-2),min(bdf_ex["Hour"].max(),lh+2)),
                         (max(bdf_ex["Hour"].min(),hh-2),min(bdf_ex["Hour"].max(),hh+2)))

                if sp > 80:   strat = "Merchant Arbitrage"
                elif sp > 40: strat = "Solar + Storage Overbuild"
                else:         strat = "Ancillary / Capacity"

                export_rows.append({"Bus":b,"Date":ex_date,"Min LMP":round(mn,2),"Low Hour":lh,
                                     "Max LMP":round(mx,2),"High Hour":hh,"Spread":sp,
                                     "2H Net Rev ($/MWh)":r2,"4H Net Rev ($/MWh)":r4,"Strategy":strat})
                prog.progress((i+1)/len(bus_list_ex))

            prog.empty()
            ex_df_out = pd.DataFrame(export_rows).sort_values("Spread",ascending=False).reset_index(drop=True)
            st.success(f"✅  Computed revenue for {len(ex_df_out)} buses")

            # Summary KPIs
            e1,e2,e3,e4 = st.columns(4)
            with e1: metric_card("Buses Analysed", str(len(ex_df_out)))
            with e2: metric_card("Avg Spread", f"${ex_df_out['Spread'].mean():.2f}")
            with e3: metric_card("Best 2H Rev", f"${ex_df_out['2H Net Rev ($/MWh)'].max():.2f}")
            with e4: metric_card("Best 4H Rev", f"${ex_df_out['4H Net Rev ($/MWh)'].max():.2f}")
            st.markdown("")

            def _col_sp(val):
                if val > 80: return "background-color:#1f4b2e;color:#5de0a5"
                if val > 40: return "background-color:#3d3510;color:#f0c040"
                return "background-color:#2e1a1a;color:#f07070"

            st.dataframe(ex_df_out.style.applymap(_col_sp, subset=["Spread"]),
                         use_container_width=True, height=400)

            # CSV download
            csv_buf = io.StringIO()
            ex_df_out.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️  Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"ERCOT_BESS_Revenue_{ex_date}.csv",
                mime="text/csv",
                type="primary"
            )
