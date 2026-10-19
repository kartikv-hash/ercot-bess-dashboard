import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import math
import io
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Sunstripe | ERCOT LMP & BESS Intelligence", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "selected_bus"   not in st.session_state: st.session_state.selected_bus   = None
if "lmp_df"         not in st.session_state: st.session_state.lmp_df         = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []

SECOND_APP_URL = "https://fatal-flaw-o7aks4agtoffgyydbvrguj.streamlit.app/"

# ─────────────────────────────────────────────
#  HARDCODED SETTLEMENT POINTS CSV
# ─────────────────────────────────────────────
_CSV_NAME = "Settlement_Points_02202026_094122.csv"

def _find_csv():
    """Search common Streamlit Cloud paths for the settlement CSV."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _CSV_NAME),
        os.path.join(os.getcwd(), _CSV_NAME),
        _CSV_NAME,
        f"/mount/src/ercot-bess-dashboard/{_CSV_NAME}",
        f"/app/ercot-bess-dashboard/{_CSV_NAME}",
        f"/home/appuser/ercot-bess-dashboard/{_CSV_NAME}",
    ]
    # Also walk one level from cwd and script dir
    for base in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
        try:
            for item in os.listdir(base):
                full = os.path.join(base, item)
                if os.path.isfile(full) and item == _CSV_NAME:
                    candidates.insert(0, full)
                elif os.path.isdir(full):
                    sub = os.path.join(full, _CSV_NAME)
                    if os.path.isfile(sub):
                        candidates.insert(0, sub)
        except Exception:
            pass
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

SETTLEMENT_CSV_PATH = _find_csv()

@st.cache_data
def load_settlement_points():
    """Load the ERCOT settlement points master CSV (bundled with app)."""
    if SETTLEMENT_CSV_PATH is None:
        # Show debug info to help locate the file
        import glob
        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files = glob.glob(os.path.join(cwd, "**/*.csv"), recursive=True)
        st.sidebar.warning(
            f"**CSV Debug Info**\n\n"
            f"- CWD: `{cwd}`\n"
            f"- Script dir: `{script_dir}`\n"
            f"- CSV files found:\n" +
            "\n".join([f"  - `{f}`" for f in csv_files[:10]]) if csv_files else "  - None"
        )
        return None
    try:
        sp = pd.read_csv(SETTLEMENT_CSV_PATH)
        sp.columns = [c.strip() for c in sp.columns]
        # Normalise key columns
        sp["SUBSTATION"] = sp["SUBSTATION"].astype(str).str.strip().str.upper()
        sp["ELECTRICAL_BUS"] = sp["ELECTRICAL_BUS"].astype(str).str.strip()
        sp["SETTLEMENT_LOAD_ZONE"] = sp["SETTLEMENT_LOAD_ZONE"].astype(str).str.strip()
        sp["VOLTAGE_LEVEL"] = pd.to_numeric(sp["VOLTAGE_LEVEL"], errors="coerce")
        sp["HUB"] = sp["HUB"].astype(str).str.strip()
        sp["HUB_BUS_NAME"] = sp["HUB_BUS_NAME"].astype(str).str.strip()
        return sp
    except Exception as e:
        return None

sp_df = load_settlement_points()

# ─────────────────────────────────────────────
#  FUZZY MATCHING HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def get_unique_substations():
    """Return sorted list of unique substation names from ERCOT CSV."""
    if sp_df is None:
        return []
    return sorted(sp_df["SUBSTATION"].dropna().unique().tolist())

def clean_osm_name(name: str) -> str:
    """Clean an OSM substation name for matching against ERCOT names."""
    clean = name.upper().strip()
    # Remove common suffixes that OSM includes but ERCOT doesn't
    for suffix in [" SUBSTATION", " SWITCHING STATION", " SWITCHYARD",
                   " SWITCH", " SUB", " SS", " STATION"]:
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)].strip()
    # Remove non-alphanumeric for comparison
    return clean

def fuzzy_match_substation(osm_name: str, top_n: int = 8):
    """Return top N fuzzy matches from ERCOT substations for an OSM name."""
    from thefuzz import fuzz, process
    subs = get_unique_substations()
    if not subs:
        return []
    cleaned = clean_osm_name(osm_name)
    # Try multiple scorers and merge
    results_ratio = process.extract(cleaned, subs, scorer=fuzz.ratio, limit=top_n)
    results_partial = process.extract(cleaned, subs, scorer=fuzz.partial_ratio, limit=top_n)
    results_sort = process.extract(cleaned, subs, scorer=fuzz.token_sort_ratio, limit=top_n)

    # Combine and deduplicate, keeping best score
    combined = {}
    for result_set in [results_ratio, results_partial, results_sort]:
        for item in result_set:
            name, score = item[0], item[1]
            if name not in combined or score > combined[name]:
                combined[name] = score
    # Sort by score descending
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def get_buses_for_substation(substation_name: str) -> pd.DataFrame:
    """Return all ERCOT buses at a given substation."""
    if sp_df is None:
        return pd.DataFrame()
    mask = sp_df["SUBSTATION"] == substation_name.upper().strip()
    return sp_df[mask][["ELECTRICAL_BUS", "NODE_NAME", "VOLTAGE_LEVEL",
                        "SETTLEMENT_LOAD_ZONE", "HUB", "HUB_BUS_NAME",
                        "PSSE_BUS_NAME"]].reset_index(drop=True)


# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  :root { --ss-red: #c92532; --ss-ink: #171717; --ss-muted: #666666; --ss-line: #dededb; --ss-paper: #f7f7f4; }
  [data-testid="stAppViewContainer"] { background: var(--ss-paper); color: var(--ss-ink); }
  [data-testid="stHeader"] { background: rgba(247,247,244,0.92); }
  [data-testid="stSidebar"] { background: #171717; border-right: 3px solid var(--ss-red); }
  [data-testid="stSidebar"] * { color: #f4f4f1 !important; }
  [data-testid="stSidebar"] [data-baseweb="radio"] label { color: #f4f4f1 !important; }
  h1, h2, h3 { color: var(--ss-ink) !important; letter-spacing: -0.025em; }
  h1 { font-weight: 750 !important; }
  p, li, label, [data-testid="stMarkdownContainer"] { color: var(--ss-ink); }
  a { color: var(--ss-red) !important; }
  .card { background:#ffffff; border:1px solid var(--ss-line); border-left:4px solid var(--ss-red); border-radius:8px; padding:15px 18px; margin-bottom:4px; box-shadow:0 2px 8px rgba(0,0,0,0.025); }
  .card-title { color:#666666; font-size:10px; text-transform:uppercase; letter-spacing:1.1px; font-weight:700; }
  .card-value { color:#171717; font-size:22px; font-weight:750; margin-top:5px; overflow-wrap:anywhere; }
  .card-sub { color:var(--ss-red); font-size:11px; margin-top:4px; }
  .section-header { font-size:11px; font-weight:750; color:var(--ss-red); text-transform:uppercase; letter-spacing:1.2px; margin:18px 0 10px 0; }
  .match-badge { display:inline-block; background:#f1f1ee; border:1px solid #dededb; border-radius:5px; padding:3px 10px; margin:2px 4px; font-size:12px; color:#333333; }
  .match-score { color:var(--ss-red); font-weight:750; }
  div[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--ss-line); border-top:3px solid var(--ss-red); padding:12px 14px; border-radius:8px; }
  div[data-testid="stMetricLabel"] p { color:#666666 !important; font-size:11px !important; text-transform:uppercase; letter-spacing:.7px; }
  div[data-testid="stMetricValue"] { color:#171717 !important; }
  .stButton > button { border-radius:6px; border:1px solid #171717; background:#171717; color:#ffffff; font-weight:650; }
  .stButton > button:hover { border-color:var(--ss-red); background:var(--ss-red); color:#ffffff; }
  div[data-testid="stFileUploader"] { background:rgba(255,255,255,.04); border-radius:8px; }
  [data-testid="stDataFrame"] { border:1px solid var(--ss-line); border-radius:8px; }
  hr { border-color:#dededb !important; }
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
#  LEGACY QUICK SPREAD PROXY (NOT DISPATCH OPTIMISATION)
# ─────────────────────────────────────────────
def bess_calc(bdf: pd.DataFrame, half_w: int):
    """Return a rolling-average spread proxy for legacy ranking/export views only.

    Use the BESS Duration Assessment tab for the optimisation-based 2H/4H comparison.
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
        "SETTLEMENT_POINT":"Bus","Settlement Point":"Bus","SettlementPoint":"Bus",
        "SETTLEMENTPOINT":"Bus","settlementPoint":"Bus","Node":"Bus",
        "LMP":"LMP","SETTLEMENT_POINT_PRICE":"LMP","SettlementPointPrice":"LMP","Price":"LMP",
        "DSTFlag":"DST","DSTFLAG":"DST","DST Flag":"DST","DST_FLAG":"DST",
    }
    # Detect SCED / timestamp-style inputs before the canonical columns are added.
    def _normalise_col_name(name):
        return "".join(ch for ch in str(name).lower() if ch.isalnum())
    timestamp_col = next((c for c in raw.columns if _normalise_col_name(c) in {
        "scedtimestamp", "timestamp", "timestamputc", "datetime", "intervaltimestamp",
        "settlementtimestamp", "intervalendingtimestamp"
    }), None)
    raw = raw.rename(columns={c: rename_rules[c] for c in raw.columns if c in rename_rules})
    for col in ["Date","Hour","Bus","LMP"]:
        if col not in raw.columns:
            raw[col] = None
    if timestamp_col is not None and timestamp_col in raw.columns:
        parsed_ts = pd.to_datetime(raw[timestamp_col], errors="coerce")
        # Naive ERCOT timestamps are treated as ERCOT local wall time. Each timestamp is
        # mapped to its hour-beginning bucket; the optimiser averages sub-hourly records.
        if parsed_ts.notna().any():
            if raw["Date"].isna().all() or raw["Date"].astype(str).str.strip().isin(["", "None", "nan"]).all():
                raw["Date"] = parsed_ts.dt.strftime("%Y-%m-%d")
            if raw["Hour"].isna().all() or raw["Hour"].astype(str).str.strip().isin(["", "None", "nan"]).all():
                raw["Hour"] = parsed_ts.dt.hour + 1
    if raw["Hour"].dtype == object:
        raw["Hour"] = (raw["Hour"].astype(str).str.strip()
                       .str.extract(r"(\d+)")[0].astype(float))
    raw["LMP"]  = pd.to_numeric(raw["LMP"],  errors="coerce")
    raw["Date"] = raw["Date"].astype(str).str.strip()
    raw["Bus"]  = raw["Bus"].astype(str).str.strip()
    raw = raw.dropna(subset=["Hour","LMP"]).reset_index(drop=True)
    return raw


# ─────────────────────────────────────────────
#  DURATION-AWARE BESS PRICE-TAKER BACKTEST
# ─────────────────────────────────────────────
def _prepare_hourly_dispatch_day(day_df: pd.DataFrame):
    """Aggregate duplicate records to one LMP per hour and retain missing hours as unavailable."""
    if day_df is None or day_df.empty or "Hour" not in day_df or "LMP" not in day_df:
        return None
    tmp = day_df[["Hour", "LMP"]].copy()
    tmp["Hour"] = pd.to_numeric(tmp["Hour"], errors="coerce")
    tmp["LMP"] = pd.to_numeric(tmp["LMP"], errors="coerce")
    tmp = tmp.dropna(subset=["Hour"])
    if tmp.empty:
        return None
    tmp["Hour"] = tmp["Hour"].round().astype(int)
    hourly = tmp.groupby("Hour", as_index=True)["LMP"].mean().sort_index()
    min_h, max_h = int(hourly.index.min()), int(hourly.index.max())
    if min_h >= 1 and max_h <= 24:
        grid = list(range(1, 25))  # ERCOT Hour Ending 1–24
    elif min_h >= 0 and max_h <= 23:
        grid = list(range(0, 24))  # hour-beginning 0–23
    else:
        grid = sorted(hourly.index.unique().tolist())
    result = pd.DataFrame({"Hour": grid})
    result["LMP"] = result["Hour"].map(hourly)
    # A full day needs enough price points to make a 2h/4h duration comparison meaningful.
    if result["LMP"].notna().sum() < max(12, int(0.70 * len(result))):
        return None
    return result.reset_index(drop=True)


def _optimise_one_day(day_df: pd.DataFrame, duration_hours: int,
                      round_trip_efficiency: float = 0.88,
                      max_efc_per_day: float = 1.0,
                      degradation_cost_per_mwh: float = 0.0):
    """Maximise hourly arbitrage margin with charge/discharge, SOC, cycling and mode constraints.

    The model is normalised to 1 MW AC. Results therefore scale linearly to a larger plant
    under the price-taker assumption. SOC is cyclic over each day (start SOC = end SOC).
    """
    try:
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import lil_matrix
    except Exception as exc:
        raise RuntimeError("SciPy is required for the dispatch optimiser. Install requirements.txt (scipy>=1.11).") from exc

    hourly = _prepare_hourly_dispatch_day(day_df)
    if hourly is None:
        return None
    prices = hourly["LMP"].to_numpy(dtype=float)
    valid = np.isfinite(prices)
    prices_objective = np.where(valid, prices, 0.0)
    n = len(prices)
    if n < 12:
        return None

    p_mw = 1.0
    e_mwh = p_mw * float(duration_hours)
    eta_c = math.sqrt(float(round_trip_efficiency))
    eta_d = math.sqrt(float(round_trip_efficiency))
    dt = 1.0

    # x = [charge(n), discharge(n), SOC(n+1), charge-mode binary(n)]
    c0 = 0
    d0 = n
    s0 = 2 * n
    z0 = 3 * n + 1
    nvars = 4 * n + 1
    obj = np.zeros(nvars, dtype=float)
    obj[c0:c0+n] = prices_objective * dt
    obj[d0:d0+n] = -(prices_objective - float(degradation_cost_per_mwh)) * dt

    lower = np.zeros(nvars, dtype=float)
    upper = np.full(nvars, np.inf, dtype=float)
    upper[c0:c0+n] = np.where(valid, p_mw, 0.0)
    upper[d0:d0+n] = np.where(valid, p_mw, 0.0)
    upper[s0:s0+n+1] = e_mwh
    upper[z0:z0+n] = 1.0
    integrality = np.zeros(nvars, dtype=int)
    integrality[z0:z0+n] = 1

    # n state transitions + 1 cyclic SOC + 2n charge/discharge mode rows + 1 cycle cap
    nrows = 3 * n + 2
    A = lil_matrix((nrows, nvars), dtype=float)
    lo = np.full(nrows, -np.inf, dtype=float)
    hi = np.full(nrows, np.inf, dtype=float)
    row = 0
    for t in range(n):
        A[row, s0+t] = -1.0
        A[row, s0+t+1] = 1.0
        A[row, c0+t] = -eta_c * dt
        A[row, d0+t] = (1.0 / eta_d) * dt
        lo[row] = hi[row] = 0.0
        row += 1

    A[row, s0] = 1.0
    A[row, s0+n] = -1.0
    lo[row] = hi[row] = 0.0
    row += 1

    for t in range(n):
        # If z=1, charging is allowed and discharge is forced to zero.
        A[row, c0+t] = 1.0
        A[row, z0+t] = -p_mw
        hi[row] = 0.0
        row += 1
    for t in range(n):
        A[row, d0+t] = 1.0
        A[row, z0+t] = p_mw
        hi[row] = p_mw
        row += 1

    # Limit discharge throughput to a user-set number of equivalent full cycles per day.
    for t in range(n):
        A[row, d0+t] = dt
    hi[row] = e_mwh * float(max_efc_per_day)

    result = milp(
        c=obj,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lo, hi),
        options={"time_limit": 8.0, "mip_rel_gap": 0.005},
    )
    if result.x is None:
        return None
    x = result.x
    charge = np.clip(x[c0:c0+n], 0.0, p_mw)
    discharge = np.clip(x[d0:d0+n], 0.0, p_mw)
    soc = np.clip(x[s0+1:s0+n+1], 0.0, e_mwh)
    real_prices = np.where(valid, prices, np.nan)
    charged_mwh = float(np.nansum(charge * dt))
    discharged_mwh = float(np.nansum(discharge * dt))
    charge_cost = float(np.nansum(np.where(valid, charge * prices, 0.0) * dt))
    discharge_revenue = float(np.nansum(np.where(valid, discharge * prices, 0.0) * dt))
    gross_margin = discharge_revenue - charge_cost
    degradation_cost = discharged_mwh * float(degradation_cost_per_mwh)
    net_margin = gross_margin - degradation_cost

    ch_denom = float(np.sum(charge * dt))
    dis_denom = float(np.sum(discharge * dt))
    avg_charge = (float(np.nansum(charge * real_prices * dt) / ch_denom)
                  if ch_denom > 1e-8 else np.nan)
    avg_discharge = (float(np.nansum(discharge * real_prices * dt) / dis_denom)
                     if dis_denom > 1e-8 else np.nan)

    dispatch = hourly.copy()
    dispatch["Charge_MW"] = charge
    dispatch["Discharge_MW"] = discharge
    dispatch["SOC_MWh"] = soc
    dispatch["Duration_Hours"] = int(duration_hours)
    dispatch["EFC"] = discharged_mwh / e_mwh if e_mwh else 0.0
    return {
        "duration": int(duration_hours),
        "gross_margin_per_mw_day": gross_margin,
        "net_margin_per_mw_day": net_margin,
        "discharge_revenue_per_mw_day": discharge_revenue,
        "charge_cost_per_mw_day": charge_cost,
        "degradation_cost_per_mw_day": degradation_cost,
        "charged_mwh_per_mw": charged_mwh,
        "discharged_mwh_per_mw": discharged_mwh,
        "efc": discharged_mwh / e_mwh if e_mwh else 0.0,
        "avg_charge_price": avg_charge,
        "avg_discharge_price": avg_discharge,
        "dispatch": dispatch,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
    }


@st.cache_data(show_spinner=False)
def calculate_duration_assessment(selected_df: pd.DataFrame,
                                  round_trip_efficiency: float,
                                  max_efc_per_day: float,
                                  degradation_cost_per_mwh: float):
    """Backtest 2h and 4h systems day by day for the selected node and dates."""
    daily_rows = []
    dispatch_frames = []
    skipped_days = 0
    if selected_df is None or selected_df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    for analysis_date, day in selected_df.groupby("_analysis_date", dropna=True):
        d2 = _optimise_one_day(day, 2, round_trip_efficiency, max_efc_per_day, degradation_cost_per_mwh)
        d4 = _optimise_one_day(day, 4, round_trip_efficiency, max_efc_per_day, degradation_cost_per_mwh)
        if d2 is None or d4 is None:
            skipped_days += 1
            continue
        date_label = pd.Timestamp(analysis_date).strftime("%Y-%m-%d")
        daily_rows.append({
            "Date": date_label,
            "2H Gross Margin ($/MW-day)": d2["gross_margin_per_mw_day"],
            "4H Gross Margin ($/MW-day)": d4["gross_margin_per_mw_day"],
            "2H Net Margin ($/MW-day)": d2["net_margin_per_mw_day"],
            "4H Net Margin ($/MW-day)": d4["net_margin_per_mw_day"],
            "2H Charge Price ($/MWh)": d2["avg_charge_price"],
            "2H Discharge Price ($/MWh)": d2["avg_discharge_price"],
            "4H Charge Price ($/MWh)": d4["avg_charge_price"],
            "4H Discharge Price ($/MWh)": d4["avg_discharge_price"],
            "2H Discharged MWh/MW-day": d2["discharged_mwh_per_mw"],
            "4H Discharged MWh/MW-day": d4["discharged_mwh_per_mw"],
            "2H Equivalent Full Cycles": d2["efc"],
            "4H Equivalent Full Cycles": d4["efc"],
        })
        for duration, result in [(2, d2), (4, d4)]:
            out = result["dispatch"].copy()
            out["Date"] = date_label
            out["Duration"] = f"{duration}H"
            dispatch_frames.append(out)

    daily = pd.DataFrame(daily_rows)
    dispatch_all = pd.concat(dispatch_frames, ignore_index=True) if dispatch_frames else pd.DataFrame()
    if not daily.empty:
        daily = daily.sort_values("Date").reset_index(drop=True)
    return daily, dispatch_all, skipped_days


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## SUNSTRIPE <span style='color:#e03a46'>/</span> ENERGY INTELLIGENCE", unsafe_allow_html=True)
    st.caption("ERCOT LMP • Battery storage analytics")
    st.caption("CSV support: hourly prices or SCED timestamps (hourly averages for duration screening)")
    st.markdown("---")
    page = st.radio("Module", [
        "🏠  Home",
        "🗺️  Node Analyser",
        "📈  LMP Price Analysis",
        "🤖  AI Copilot"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### LMP DATA")
    uploaded = st.file_uploader("Upload ERCOT LMP CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = load_lmp(uploaded)
        st.session_state.lmp_df = df
        st.success(f"✅ {len(df):,} rows loaded")
    elif st.session_state.lmp_df is not None:
        df = st.session_state.lmp_df
    else:
        df = None

    # Settlement points status
    st.markdown("---")
    st.markdown("### SETTLEMENT POINTS")
    if sp_df is not None:
        st.success(f"✅ {sp_df['SUBSTATION'].nunique():,} substations · {len(sp_df):,} buses")
    else:
        st.error("❌ Settlement CSV not found")

    st.markdown("---")
    st.markdown("### OTHER TOOLS")
    st.markdown(f'<a href="{SECOND_APP_URL}" target="_blank"><button style="width:100%;padding:8px;background:#1e2a3a;color:#4fc3f7;border:1px solid #2a3d55;border-radius:8px;cursor:pointer;font-size:13px;">📍 Open Fatal Flaw Analyser — SiteIQ</button></a>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px 0;">
        <div style="font-size:52px;">⚡</div>
        <h1 style="font-size:32px;font-weight:800;margin:8px 0 4px 0;">Sunstripe ERCOT LMP & BESS Intelligence</h1>
        <p style="color:#7880a8;font-size:15px;margin:0;">Nodal price signals • Dispatch optimisation • Duration screening</p>
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
                Sunstripe ERCOT BESS Intelligence
            </div>
            <div style="color:#7880a8;font-size:13px;line-height:1.7;margin-bottom:20px;">
                <b style="color:#5de0a5;">Current App</b><br><br>
                🗺️ &nbsp;Hub & Node Analyser with OSM map<br>
                📈 &nbsp;LMP Price Analysis & BESS strategy<br>
                🔌 &nbsp;Substation → Bus mapping from ERCOT data<br>
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
        SUNSTRIPE ENERGY INTELLIGENCE  •  ERCOT nodal pricing  •  Battery duration screening
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 – NODE ANALYSER
# ══════════════════════════════════════════════
elif page == "🗺️  Node Analyser":
    st.title("🗺️  Hub & Node Analyser")
    st.caption("Discovers transmission substations from OpenStreetMap, then matches them to ERCOT settlement points & buses.")

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

        # ── MAP ──
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

        # ── Substation List ──
        st.markdown('<div class="section-header">Substation List</div>', unsafe_allow_html=True)
        def _style_type(val):
            if val=="Hub": return "background-color:#3b1f5e;color:#bf7fff;font-weight:600"
            return "background-color:#1a3550;color:#4fc3f7;font-weight:600"
        st.dataframe(
            sdf.drop(columns=["_id"]).style.applymap(_style_type, subset=["Type"]),
            use_container_width=True, height=300
        )

        # ══════════════════════════════════════
        #  🔌 SUBSTATION → ERCOT BUS MAPPING
        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown('<div class="section-header">🔌 Substation → ERCOT Bus Mapping</div>', unsafe_allow_html=True)
        st.caption("Select an OSM substation, then pick the closest ERCOT match to see all buses at that substation.")

        if sp_df is None:
            st.error("Settlement Points CSV not loaded. Place `Settlement_Points_02202026_094122.csv` alongside the app.")
            st.stop()

        sel_osm = st.selectbox("Select an OSM Substation", sdf["Name"].tolist(), key="osm_sub_sel")

        if sel_osm:
            # Perform fuzzy matching
            with st.spinner("Finding closest ERCOT matches …"):
                matches = fuzzy_match_substation(sel_osm, top_n=8)

            if matches:
                # Build display labels with scores
                match_options = [f"{name}  (score: {score})" for name, score in matches]
                match_names   = [name for name, score in matches]

                st.markdown(f"**OSM Name:** `{sel_osm}` → **Cleaned:** `{clean_osm_name(sel_osm)}`")

                selected_match_label = st.selectbox(
                    "Select the best ERCOT Substation match",
                    match_options,
                    key="ercot_match_sel"
                )
                # Extract the actual name from the label
                selected_ercot_sub = selected_match_label.split("  (score:")[0].strip()

                # Show buses at this substation
                buses_at_sub = get_buses_for_substation(selected_ercot_sub)

                if not buses_at_sub.empty:
                    st.markdown(f"### 🔋 Buses at **{selected_ercot_sub}**")

                    k1, k2, k3 = st.columns(3)
                    with k1:
                        metric_card("Total Buses", str(len(buses_at_sub)))
                    with k2:
                        zones = buses_at_sub["SETTLEMENT_LOAD_ZONE"].unique()
                        metric_card("Load Zone(s)", ", ".join(zones))
                    with k3:
                        voltages = sorted(buses_at_sub["VOLTAGE_LEVEL"].dropna().unique())
                        metric_card("Voltage Levels", ", ".join([f"{v:.0f}" if v == int(v) else f"{v}" for v in voltages]) + " kV")

                    st.dataframe(buses_at_sub, use_container_width=True, height=250)

                    # ── Direct LMP plotting if LMP data is loaded ──
                    if df is not None:
                        st.markdown("---")
                        st.markdown('<div class="section-header">📈 LMP for Buses at This Substation</div>', unsafe_allow_html=True)

                        # Find which buses from this substation exist in the LMP data
                        lmp_buses = set(df["Bus"].unique())
                        sub_buses = set(buses_at_sub["ELECTRICAL_BUS"].unique())
                        matched_buses = sorted(lmp_buses & sub_buses)

                        if matched_buses:
                            st.success(f"✅ **{len(matched_buses)}** bus(es) from {selected_ercot_sub} found in your LMP data")

                            sel_lmp_buses = st.multiselect(
                                "Select buses to plot",
                                matched_buses,
                                default=matched_buses[:min(5, len(matched_buses))],
                                key="sub_lmp_buses"
                            )

                            dates = sorted(df["Date"].unique().tolist())
                            sel_date_node = st.selectbox("Select Date", dates, key="node_lmp_date") if len(dates) > 1 else dates[0]

                            if sel_lmp_buses:
                                PALETTE = ["#c92532", "#171717", "#777777", "#e58c22", "#9d6b73", "#5f6972", "#b8b8b2", "#d4a1a6"]
                                fig_node_lmp = go.Figure()
                                node_summary = []

                                for bi, bus_name in enumerate(sel_lmp_buses):
                                    bdf = (df[(df["Bus"]==bus_name) & (df["Date"]==sel_date_node)]
                                           .sort_values("Hour").reset_index(drop=True))
                                    if bdf.empty:
                                        continue

                                    col = PALETTE[bi % len(PALETTE)]
                                    fig_node_lmp.add_trace(go.Scatter(
                                        x=bdf["Hour"], y=bdf["LMP"],
                                        name=bus_name,
                                        line=dict(color=col, width=2.5),
                                        hovertemplate=f"Bus: {bus_name}<br>Hour %{{x}}<br>LMP: $%{{y:.2f}}<extra></extra>"
                                    ))

                                    # Calc BESS revenue
                                    if len(bdf) >= 3:
                                        r2, _, lh, hh, _, _ = bess_calc(bdf, 1)
                                        r4, _, _, _, _, _   = bess_calc(bdf, 2)
                                        sp = round(bdf["LMP"].max() - bdf["LMP"].min(), 2)
                                        node_summary.append({
                                            "Bus": bus_name,
                                            "Min LMP": round(bdf["LMP"].min(), 2),
                                            "Max LMP": round(bdf["LMP"].max(), 2),
                                            "Spread": sp,
                                            "3H Window Proxy Spread ($/MWh)": r2,
                                            "5H Window Proxy Spread ($/MWh)": r4,
                                        })

                                fig_node_lmp.update_layout(
                                    template="plotly_white",
                                    title=f"LMP at {selected_ercot_sub} Buses — {sel_date_node}",
                                    xaxis=dict(title="Hour Ending", tickmode="linear", dtick=1),
                                    yaxis=dict(title="LMP ($/MWh)"),
                                    legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                                    hovermode="x unified", height=480
                                )
                                st.plotly_chart(fig_node_lmp, use_container_width=True)

                                if node_summary:
                                    st.markdown("**Quick rolling-average proxy (not optimised dispatch revenue)**")
                                    sum_df = pd.DataFrame(node_summary).sort_values("Spread", ascending=False)
                                    st.dataframe(sum_df, use_container_width=True)

                                    # Quick jump to full analysis
                                    best = sum_df.iloc[0]["Bus"]
                                    if st.button(f"📈 Open full analysis for **{best}**", key="jump_lmp_from_node"):
                                        st.session_state.selected_bus = best
                                        st.info(f"Switched to **{best}** — open the BESS Duration Assessment tab.")
                        else:
                            st.warning(
                                f"No buses from **{selected_ercot_sub}** were found in your LMP data. "
                                f"The LMP CSV has buses like: {', '.join(list(lmp_buses)[:8])}…\n\n"
                                f"The substation has buses like: {', '.join(list(sub_buses)[:8])}…"
                            )
                else:
                    st.info(f"No buses found at substation **{selected_ercot_sub}** in the ERCOT CSV.")
            else:
                st.warning("No fuzzy matches found. The OSM name may not correspond to an ERCOT substation.")

        # ── Real-Time LMP from ERCOT public API ──
        st.markdown("---")
        st.markdown('<div class="section-header">⚡ Real-Time LMP for Selected Substation</div>', unsafe_allow_html=True)

        sel_sub = st.selectbox("Select Substation", sdf["Name"].tolist(), key="rtlmp_sub")

        st.caption("ERCOT settlement point names may differ from OSM names. Edit below if needed.")
        sp_guess = sel_sub.upper().replace(" ","_").replace("-","_")[:20]
        settlement_point = st.text_input("ERCOT Settlement Point Name", value=sp_guess,
                                         help="Must match an ERCOT bus/hub name exactly e.g. HB_NORTH, LZ_HOUSTON")

        fetch_rt = st.button("📡  Fetch Real-Time LMP", type="primary")

        if fetch_rt and settlement_point.strip():
            from datetime import datetime, timedelta, timezone
            now_utc   = datetime.now(timezone.utc)
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

                    fields  = [f["name"] for f in rt_json.get("fields", [])]
                    records = rt_json.get("data", [])

                    if not records:
                        st.warning(f"No real-time data returned for **{sp_clean}**. "
                                   "Check the settlement point name matches an ERCOT bus exactly (e.g. HB_NORTH, LZ_WEST).")
                    else:
                        rt_df = pd.DataFrame(records, columns=fields)
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

                        st.session_state["rt_summary"] = {
                            "sp": sp_clean, "latest": latest_lmp,
                            "avg": avg_lmp, "max": max_lmp, "min": min_lmp
                        }
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
                            template="plotly_white",
                            title=f"Real-Time LMP — {sp_clean}  (Last 2 Hours)",
                            xaxis_title="Time (UTC)",
                            yaxis_title="LMP ($/MWh)",
                            height=380, margin=dict(t=50)
                        )
                        st.plotly_chart(fig_rt, use_container_width=True)

                        with st.expander("📄 Raw real-time data"):
                            st.dataframe(rt_df[[ts_col, lmp_col]].rename(
                                columns={ts_col:"Timestamp (UTC)", lmp_col:"LMP ($/MWh)"}),
                                use_container_width=True)

                except requests.exceptions.HTTPError as e:
                    st.error(f"ERCOT API returned an error: {rt_resp.status_code}.")
                    with st.expander("Debug info"):
                        st.code(rt_resp.text[:500])
                except Exception as e:
                    st.error(f"Could not fetch real-time data: {e}")

        # ── Link to LMP Analysis ──
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

        # ── AI Workflow Summary ──
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 AI Node & LMP Summary</div>', unsafe_allow_html=True)

        ai_key = st.session_state.get("api_key", "")

        if not ai_key:
            st.info("🔑 Enter your Anthropic API key in the sidebar to enable the AI summary.")
        else:
            nearest_5 = sdf.head(5)[["Name","Type","Voltage","Distance (mi)","Operator"]].to_string(index=False)
            hub_list   = sdf[sdf["Type"]=="Hub"][["Name","Voltage","Distance (mi)"]].head(5).to_string(index=False)
            node_list  = sdf[sdf["Type"]=="Node"][["Name","Voltage","Distance (mi)"]].head(5).to_string(index=False)
            volt_dist  = sdf["Voltage"].value_counts().to_string()

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

The user has just run a substation search on the Sunstripe ERCOT BESS Intelligence.
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔋 BESS Duration Assessment",
        "🔌 Substation Lookup",
        "📊 Top N Buses by Spread",
        "📥 Export Revenue Table"
    ])

    # ═══ TAB 1 – DURATION ASSESSMENT ══════════
    with tab1:
        st.markdown("### BESS duration decision")
        st.caption("Compare 2-hour and 4-hour storage using the selected node's historical hourly LMPs and an explicit dispatch model.")

        bus_list = sorted(df["Bus"].dropna().astype(str).unique().tolist())
        default_idx = bus_list.index(st.session_state.selected_bus) if st.session_state.selected_bus in bus_list else 0
        bus = st.selectbox("Settlement point / bus", bus_list, index=default_idx, key="duration_bus")
        st.session_state.selected_bus = bus

        if sp_df is not None:
            bus_info = sp_df[sp_df["ELECTRICAL_BUS"].astype(str) == str(bus)]
            if not bus_info.empty:
                row0 = bus_info.iloc[0]
                st.markdown(
                    f"**Substation:** `{row0['SUBSTATION']}` &nbsp; | &nbsp; "
                    f"**Load zone:** `{row0['SETTLEMENT_LOAD_ZONE']}` &nbsp; | &nbsp; "
                    f"**Voltage:** `{row0['VOLTAGE_LEVEL']} kV` &nbsp; | &nbsp; "
                    f"**Hub:** `{row0['HUB'] if row0['HUB'] not in ['nan',''] else '—'}`"
                )

        bus_data = df[df["Bus"].astype(str) == str(bus)].copy()
        bus_data["_analysis_date"] = pd.to_datetime(bus_data["Date"], errors="coerce").dt.date
        bus_data = bus_data.dropna(subset=["_analysis_date", "Hour", "LMP"])
        valid_dates = sorted(bus_data["_analysis_date"].dropna().unique().tolist())
        if not valid_dates:
            st.warning("No usable date/hour/LMP records were found for this bus. Check the uploaded CSV's Date, Hour and LMP fields.")
            st.stop()

        left, right = st.columns([2, 1])
        with left:
            period = st.date_input(
                "Historical analysis period",
                value=(valid_dates[0], valid_dates[-1]),
                min_value=valid_dates[0], max_value=valid_dates[-1],
                key="duration_period"
            )
        with right:
            plant_mw = st.number_input("Reference plant power (MW)", min_value=0.1, max_value=5000.0,
                                       value=100.0, step=10.0, key="duration_plant_mw",
                                       help="The optimisation is run per 1 MW, then scaled linearly under a price-taker assumption.")
        if isinstance(period, (tuple, list)) and len(period) == 2:
            start_date, end_date = period
        else:
            start_date = end_date = period
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        chosen = bus_data[(bus_data["_analysis_date"] >= start_date) & (bus_data["_analysis_date"] <= end_date)].copy()
        if chosen.empty:
            st.warning("No records exist for the selected date range.")
            st.stop()

        st.markdown('<div class="section-header">Model assumptions</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            rte_pct = st.slider("Round-trip efficiency (%)", min_value=60, max_value=98, value=88, step=1,
                                help="Split symmetrically between charging and discharging: each one-way efficiency is sqrt(RTE).")
        with a2:
            efc_limit = st.slider("Maximum equivalent full cycles/day", min_value=0.25, max_value=3.0,
                                  value=1.0, step=0.25,
                                  help="Editable dispatch-throughput limit, not an ERCOT rule or warranty limit.")
        with a3:
            degradation_cost = st.number_input("Degradation / variable cost ($/MWh discharged)", min_value=0.0,
                                               max_value=500.0, value=0.0, step=1.0,
                                               help="Optional variable cost applied to discharged MWh. Leave at zero if not yet estimated.")
        with a4:
            energy_capex = st.number_input("Installed energy cost ($/kWh)", min_value=0.0, max_value=3000.0,
                                           value=0.0, step=25.0,
                                           help="Enter a project-specific cost to test whether the extra 2 MWh/MW of a 4-hour battery pays for itself. Zero disables the capital-cost test.")
        capital_rate = st.number_input("Annualized capital recovery / carrying rate (%)", min_value=0.0,
                                       max_value=100.0, value=10.0, step=0.5,
                                       help="Finance assumption only. Replace with your own capital recovery, financing and life assumptions.")

        st.caption("The 2H case is 1 MW / 2 MWh; the 4H case is 1 MW / 4 MWh. Both use the same MW rating and price series. The added 4H energy capacity is 2 MWh for each MW of power.")

        with st.spinner("Optimising daily charge/discharge schedules for 2H and 4H systems …"):
            try:
                daily, dispatch_all, skipped_days = calculate_duration_assessment(
                    chosen[["Date", "Hour", "LMP", "_analysis_date"]],
                    rte_pct / 100.0, float(efc_limit), float(degradation_cost)
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Duration optimisation failed: {exc}")
                st.stop()

        if daily.empty:
            st.warning("No days had enough hourly price records to optimise. Please upload a dataset with at least 12 distinct hourly LMP points per day.")
            st.stop()
        if skipped_days:
            st.warning(f"Skipped {skipped_days} date(s) with insufficient hourly coverage. Duplicate records within the same hour were averaged before optimisation.")

        # Annualised historical run-rate is a sample statistic, not a forward forecast.
        day_count = len(daily)
        avg_2_per_mw_day = float(daily["2H Net Margin ($/MW-day)"].mean())
        avg_4_per_mw_day = float(daily["4H Net Margin ($/MW-day)"].mean())
        period_2 = float(daily["2H Net Margin ($/MW-day)"].sum()) * plant_mw
        period_4 = float(daily["4H Net Margin ($/MW-day)"].sum()) * plant_mw
        annual_2 = avg_2_per_mw_day * 365.0 * plant_mw
        annual_4 = avg_4_per_mw_day * 365.0 * plant_mw
        annual_inc_per_mw = (avg_4_per_mw_day - avg_2_per_mw_day) * 365.0
        annual_inc_plant = annual_inc_per_mw * plant_mw
        annualised_rate = capital_rate / 100.0
        incremental_energy_capex_plant = 2.0 * 1000.0 * energy_capex * plant_mw
        annualized_incremental_capex = incremental_energy_capex_plant * annualised_rate
        incremental_contribution_after_capex = annual_inc_plant - annualized_incremental_capex
        breakeven_capex_per_kwh = (annual_inc_per_mw / (2000.0 * annualised_rate)
                                    if annualised_rate > 0 else np.nan)
        cap_productivity_2 = (avg_2_per_mw_day * 365.0 / 2.0)
        cap_productivity_4 = (avg_4_per_mw_day * 365.0 / 4.0)

        st.markdown('<div class="section-header">Duration comparison</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("2H net arbitrage margin", f"${annual_2:,.0f}/yr", f"{plant_mw:g} MW plant · annualised run-rate")
        with k2:
            metric_card("4H net arbitrage margin", f"${annual_4:,.0f}/yr", f"{plant_mw:g} MW plant · annualised run-rate")
        with k3:
            metric_card("4H incremental margin", f"${annual_inc_plant:,.0f}/yr", "vs 2H, before extra energy-capacity capex")
        with k4:
            metric_card("Value of extra duration", f"${annual_inc_per_mw/2:,.0f}", "per added MWh of capacity per year")

        comparison = pd.DataFrame([
            {"Metric": "Historical period net margin", "2-hour": period_2, "4-hour": period_4, "Units": "$ total for selected period and reference plant"},
            {"Metric": "Annualised net margin", "2-hour": annual_2, "4-hour": annual_4, "Units": "$ / year for reference plant"},
            {"Metric": "Annualised margin per installed MWh", "2-hour": cap_productivity_2, "4-hour": cap_productivity_4, "Units": "$ / MWh nameplate / year / MW"},
            {"Metric": "Average discharge price", "2-hour": daily["2H Discharge Price ($/MWh)"].mean(), "4-hour": daily["4H Discharge Price ($/MWh)"].mean(), "Units": "$ / MWh"},
            {"Metric": "Average charge price", "2-hour": daily["2H Charge Price ($/MWh)"].mean(), "4-hour": daily["4H Charge Price ($/MWh)"].mean(), "Units": "$ / MWh"},
            {"Metric": "Equivalent full cycles/day", "2-hour": daily["2H Equivalent Full Cycles"].mean(), "4-hour": daily["4H Equivalent Full Cycles"].mean(), "Units": "cycles / day"},
        ])
        st.dataframe(comparison, hide_index=True, use_container_width=True,
                     column_config={"2-hour": st.column_config.NumberColumn(format="$%.2f"),
                                    "4-hour": st.column_config.NumberColumn(format="$%.2f")})

        st.markdown('<div class="section-header">Historical LMP with optimised dispatch</div>', unsafe_allow_html=True)
        date_options = daily["Date"].astype(str).tolist()
        rep_date = st.selectbox("Choose a date to inspect the dispatch schedule", date_options,
                                index=len(date_options)-1, key="duration_plot_date")
        view = dispatch_all[dispatch_all["Date"].astype(str) == str(rep_date)].copy()
        if not view.empty:
            lmp_plot = view[view["Duration"] == "2H"].sort_values("Hour")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lmp_plot["Hour"], y=lmp_plot["LMP"], mode="lines+markers",
                                     name="Observed hourly LMP", line=dict(color="#171717", width=2.5),
                                     marker=dict(size=5, color="#171717"),
                                     hovertemplate="Hour %{x}<br>LMP: $%{y:.2f}/MWh<extra></extra>"))
            for duration, color_charge, color_discharge, label in [
                ("2H", "#c92532", "#e58c22", "2-hour"),
                ("4H", "#8b5c68", "#5c6570", "4-hour"),
            ]:
                dd = view[view["Duration"] == duration].sort_values("Hour")
                charge_rows = dd[dd["Charge_MW"] > 0.02]
                discharge_rows = dd[dd["Discharge_MW"] > 0.02]
                if not charge_rows.empty:
                    fig.add_trace(go.Scatter(x=charge_rows["Hour"], y=charge_rows["LMP"], mode="markers",
                                             name=f"{label} charging", marker=dict(symbol="triangle-down", size=12, color=color_charge),
                                             customdata=charge_rows[["Charge_MW", "SOC_MWh"]],
                                             hovertemplate=f"{label} charge<br>Hour %{{x}}<br>LMP: $%{{y:.2f}}/MWh<br>Charge: %{{customdata[0]:.2f}} MW<extra></extra>"))
                if not discharge_rows.empty:
                    fig.add_trace(go.Scatter(x=discharge_rows["Hour"], y=discharge_rows["LMP"], mode="markers",
                                             name=f"{label} discharging", marker=dict(symbol="triangle-up", size=12, color=color_discharge),
                                             customdata=discharge_rows[["Discharge_MW", "SOC_MWh"]],
                                             hovertemplate=f"{label} discharge<br>Hour %{{x}}<br>LMP: $%{{y:.2f}}/MWh<br>Discharge: %{{customdata[0]:.2f}} MW<extra></extra>"))
            fig.update_layout(template="plotly_white", title=f"{bus} — observed LMP and price-taker dispatch — {rep_date}",
                              xaxis=dict(title="Hour Ending", tickmode="linear", dtick=1),
                              yaxis=dict(title="LMP ($/MWh)"), hovermode="x unified", height=480,
                              legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Triangles mark modelled charge/discharge intervals. The schedule is an ex-post perfect-foresight backtest, not a real-time forecast or submitted ERCOT bid.")

        st.markdown('<div class="section-header">How the LMP trend changes the duration signal</div>', unsafe_allow_html=True)
        monthly = daily.copy()
        monthly["Month"] = pd.to_datetime(monthly["Date"]).dt.to_period("M").astype(str)
        monthly = monthly.groupby("Month", as_index=False)[["2H Net Margin ($/MW-day)", "4H Net Margin ($/MW-day)" ]].mean()
        if len(monthly):
            fig_month = go.Figure()
            fig_month.add_trace(go.Scatter(x=monthly["Month"], y=monthly["2H Net Margin ($/MW-day)"],
                                           name="2H", mode="lines+markers", line=dict(color="#c92532", width=2.5)))
            fig_month.add_trace(go.Scatter(x=monthly["Month"], y=monthly["4H Net Margin ($/MW-day)"],
                                           name="4H", mode="lines+markers", line=dict(color="#171717", width=2.5)))
            fig_month.update_layout(template="plotly_white", title="Average daily net arbitrage margin by month (per MW)",
                                    xaxis_title="Month", yaxis_title="Net margin ($/MW-day)", height=350,
                                    legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(fig_month, use_container_width=True)

        st.markdown('<div class="section-header">Duration recommendation</div>', unsafe_allow_html=True)
        if energy_capex > 0 and annualised_rate > 0:
            if incremental_contribution_after_capex > 0:
                st.success(
                    f"**4-hour passes the incremental energy-capacity screen** under the entered assumptions. "
                    f"The estimated incremental annual energy-arbitrage margin is ${annual_inc_plant:,.0f}; "
                    f"the annualised carrying cost of the extra {2*plant_mw:,.0f} MWh is ${annualized_incremental_capex:,.0f}; "
                    f"net incremental contribution is ${incremental_contribution_after_capex:,.0f}/year."
                )
            else:
                st.warning(
                    f"**2-hour is favoured on energy arbitrage plus the entered incremental energy-capacity cost.** "
                    f"4-hour adds ${annual_inc_plant:,.0f}/year in estimated net arbitrage margin, versus "
                    f"${annualized_incremental_capex:,.0f}/year of annualised extra energy-capacity cost. "
                    f"Other revenues (ancillary services, capacity/RA, tolling) could change the decision."
                )
        else:
            raw_winner = "4-hour" if annual_4 >= annual_2 else "2-hour"
            productivity_winner = "2-hour" if cap_productivity_2 >= cap_productivity_4 else "4-hour"
            if np.isfinite(breakeven_capex_per_kwh):
                breakeven_line = f"The estimated break-even installed energy CAPEX for the additional 2 MWh/MW is about ${max(0.0, breakeven_capex_per_kwh):,.0f}/kWh at a {capital_rate:.1f}% annualised carrying rate."
            else:
                breakeven_line = "Set a positive annualised capital recovery rate to calculate a break-even energy CAPEX."
            st.info(
                f"**LMP-only signal:** {raw_winner} captures the higher total historical net arbitrage margin per MW, "
                f"while {productivity_winner} has the higher margin per MWh of installed energy capacity. "
                f"{breakeven_line} Enter a project-specific installed energy cost above to get a capital-adjusted screen; "
                f"without it, historical LMPs alone cannot establish investment feasibility."
            )

        st.markdown('<div class="section-header">Daily backtest results</div>', unsafe_allow_html=True)
        st.dataframe(daily, hide_index=True, use_container_width=True, height=300)
        csv_data = daily.to_csv(index=False).encode("utf-8")
        st.download_button("Download daily 2H vs 4H backtest CSV", data=csv_data,
                           file_name=f"Sunstripe_BESS_Duration_{bus}_{start_date}_{end_date}.csv",
                           mime="text/csv")

        with st.expander("Research basis, optimisation method and limitations"):
            st.markdown("""
**Research used to frame the duration decision**

- **ERCOT, *Security Constrained Unit Commitment (SCUC)-Based Tool to Manage the Grid with Large Penetration of Energy Storage Resources* (August 2025).** ERCOT explains that the duration signal from energy-price arbitrage should be read from the duration of high-price discharge opportunities versus low-price charging opportunities. Its discussion says energy arbitrage by itself did not provide a sufficient signal for investment in duration beyond roughly 2–4 hours; ancillary-service requirements and other revenue constructs also matter. [ERCOT report (PDF)](https://www.ercot.com/files/docs/2025/08/29/Security-Constrained-Unit-Commitment.pdf)

- **Turan & Cole, *Revenue Analysis for Energy Storage Systems in the United States* (NREL/NLR, 2024).** The study evaluates historical and projected prices, durations from 1 to 12 hours, location, and round-trip efficiency. Its published summary finds that revenue generally rises with duration, but marginal value declines as duration increases, and economics depend on location and efficiency. [Report page](https://research-hub.nlr.gov/en/publications/revenue-analysis-for-energy-storage-systems-in-the-united-states/) · [Full report](https://www.nrel.gov/docs/fy25osti/89906.pdf)

- **Modo Energy, *ERCOT: Annual battery energy storage revenue report 2024* (published March 2025).** This market review illustrates why nodal energy arbitrage should not be treated as total BESS revenue: realised battery earnings include multiple market products, and reported returns can change as deployed storage and market conditions change. [Research article](https://modoenergy.com/research/ercot-battery-energy-storage-revenues-energy-markets-ancillary-services-capacity-location-owners-operators-annual-report-2024)

**What this app calculates**

The optimiser treats historical LMPs as fixed (a price-taker / ex-post backtest), runs a separate daily schedule for each duration, and maximises energy-market margin subject to a 1 MW AC power limit, a 2 MWh or 4 MWh energy limit, round-trip efficiency split symmetrically between charge and discharge, a user-set discharge-throughput / equivalent-full-cycle limit, no simultaneous charging and discharging, and equal start/end state of charge each day. Missing hourly prices cannot be dispatched; duplicate records in an hour are averaged.

**Objective, per 1 MW of power**

`Net margin = Σ(discharge MW × LMP × Δt) − Σ(charge MW × LMP × Δt) − degradation cost`

The annualised figure is the average daily margin in the selected history multiplied by 365. It is a simple historical run-rate, not a price forecast or a bankable revenue forecast. The capital test only compares the incremental 2 MWh/MW energy capacity using the user's stated installed $/kWh and annualised carrying rate.

**Not included:** ancillary-service revenues, capacity/resource-adequacy payments, tolling contracts, negative-price or bid strategy beyond observed LMPs, market-price impact from the project, congestion deliverability studies, network losses/auxiliary load, charging limits imposed by interconnection, taxes, fixed O&M, augmentation/replacement, warranties, degradation not entered above, financing structure, construction costs, and merchant-price forecasts. Use this as an initial screening tool, not final project investment advice.
            """)

    # ═══ TAB 2 – Substation Lookup ════════════
    with tab2:
        st.markdown('<div class="section-header">🔌 Find Buses by ERCOT Substation</div>', unsafe_allow_html=True)
        st.caption("Any 2H/4H values shown in this lookup are quick rolling-average spread proxies, not optimised revenue.")
        st.caption("Search for an ERCOT substation name to see all its buses and their LMP data.")

        if sp_df is None:
            st.error("Settlement Points CSV not loaded.")
            st.stop()

        all_subs = get_unique_substations()

        search_mode = st.radio("Search mode", ["Browse substations", "Fuzzy search by name"], horizontal=True, key="sub_search_mode")

        if search_mode == "Browse substations":
            # Filter by load zone
            zone_filter = st.selectbox("Filter by Load Zone", ["All"] + sorted(sp_df["SETTLEMENT_LOAD_ZONE"].unique().tolist()), key="zone_filt")
            if zone_filter != "All":
                filtered_subs = sorted(sp_df[sp_df["SETTLEMENT_LOAD_ZONE"] == zone_filter]["SUBSTATION"].unique().tolist())
            else:
                filtered_subs = all_subs

            chosen_sub = st.selectbox(f"Select Substation ({len(filtered_subs)} available)", filtered_subs, key="browse_sub")
        else:
            fuzzy_input = st.text_input("Type a substation name to search", key="fuzzy_sub_input")
            if fuzzy_input.strip():
                matches = fuzzy_match_substation(fuzzy_input, top_n=10)
                match_labels = [f"{n}  (score: {s})" for n, s in matches]
                chosen_label = st.selectbox("Best matches", match_labels, key="fuzzy_sub_result")
                chosen_sub = chosen_label.split("  (score:")[0].strip()
            else:
                chosen_sub = None

        if chosen_sub:
            buses_here = get_buses_for_substation(chosen_sub)

            if not buses_here.empty:
                k1, k2, k3 = st.columns(3)
                with k1:
                    metric_card("Substation", chosen_sub)
                with k2:
                    metric_card("Total Buses", str(len(buses_here)))
                with k3:
                    zones = buses_here["SETTLEMENT_LOAD_ZONE"].unique()
                    metric_card("Load Zone", ", ".join(zones))

                st.dataframe(buses_here, use_container_width=True, height=250)

                # Find which of these buses are in the LMP data
                lmp_buses = set(df["Bus"].unique())
                sub_buses = set(buses_here["ELECTRICAL_BUS"].unique())
                matched_lmp = sorted(lmp_buses & sub_buses)

                if matched_lmp:
                    st.success(f"✅ **{len(matched_lmp)}** bus(es) found in your LMP data")

                    sel_plot_buses = st.multiselect(
                        "Select buses to plot",
                        matched_lmp,
                        default=matched_lmp[:min(5, len(matched_lmp))],
                        key="tab2_plot_buses"
                    )

                    dates = sorted(df["Date"].unique().tolist())
                    sel_date_t2 = st.selectbox("Date", dates, key="tab2_date") if len(dates) > 1 else dates[0]

                    if sel_plot_buses:
                        PALETTE = ["#c92532", "#171717", "#777777", "#e58c22", "#9d6b73", "#5f6972", "#b8b8b2", "#d4a1a6"]
                        fig_t2 = go.Figure()
                        rev_rows_t2 = []

                        for bi, bname in enumerate(sel_plot_buses):
                            bdf = (df[(df["Bus"]==bname) & (df["Date"]==sel_date_t2)]
                                   .sort_values("Hour").reset_index(drop=True))
                            if bdf.empty:
                                continue
                            c = PALETTE[bi % len(PALETTE)]
                            fig_t2.add_trace(go.Scatter(
                                x=bdf["Hour"], y=bdf["LMP"],
                                name=bname,
                                line=dict(color=c, width=2.5),
                                hovertemplate=f"Bus: {bname}<br>Hour %{{x}}<br>LMP: $%{{y:.2f}}<extra></extra>"
                            ))
                            if len(bdf) >= 3:
                                r2, _, _, _, _, _ = bess_calc(bdf, 1)
                                r4, _, _, _, _, _ = bess_calc(bdf, 2)
                                sp_val = round(bdf["LMP"].max() - bdf["LMP"].min(), 2)
                                rev_rows_t2.append({
                                    "Bus": bname, "Spread": sp_val,
                                    "3H Window Proxy Spread ($/MWh)": r2, "5H Window Proxy Spread ($/MWh)": r4,
                                    "Avg LMP": round(bdf["LMP"].mean(), 2),
                                })

                        fig_t2.update_layout(
                            template="plotly_white",
                            title=f"LMP at {chosen_sub} — {sel_date_t2}",
                            xaxis=dict(title="Hour Ending", tickmode="linear", dtick=1),
                            yaxis=dict(title="LMP ($/MWh)"),
                            legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)"),
                            hovermode="x unified", height=480,
                        )
                        st.plotly_chart(fig_t2, use_container_width=True)

                        if rev_rows_t2:
                            st.markdown("**Quick rolling-average proxy (not optimised dispatch revenue)**")
                            rev_df_t2 = pd.DataFrame(rev_rows_t2).sort_values("Spread", ascending=False)
                            st.dataframe(rev_df_t2, use_container_width=True)

                            best_b = rev_df_t2.iloc[0]["Bus"]
                            if st.button(f"📈 Full BESS analysis for **{best_b}**", key="jump_from_tab2"):
                                st.session_state.selected_bus = best_b
                                st.info(f"Switched to **{best_b}** — open the 'BESS Duration Assessment' tab.")
                else:
                    st.warning(
                        f"None of the {len(sub_buses)} buses at **{chosen_sub}** appear in your uploaded LMP data. "
                        f"Buses at this substation: {', '.join(list(sub_buses)[:10])}…"
                    )
            else:
                st.info(f"No data found for substation **{chosen_sub}**.")

    # ═══ TAB 3 – Top N Buses ══════════════════
    with tab3:
        st.markdown('<div class="section-header">Top N Buses by LMP Spread</div>', unsafe_allow_html=True)
        st.caption("Spread rankings are an initial locator only. Use the BESS Duration Assessment tab for optimised 2H/4H dispatch economics.")
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
            rev_rows.append({"Bus":row["Bus"],"3H Window Proxy Spread ($/MWh)":r2,"5H Window Proxy Spread ($/MWh)":r4,
                             "Low Hr":lh2,"High Hr":hh2})
        top_df = bus_stats.merge(pd.DataFrame(rev_rows), on="Bus")

        # Enrich with substation info
        if sp_df is not None:
            sub_lookup = sp_df.drop_duplicates("ELECTRICAL_BUS")[["ELECTRICAL_BUS","SUBSTATION","SETTLEMENT_LOAD_ZONE"]]
            top_df = top_df.merge(sub_lookup, left_on="Bus", right_on="ELECTRICAL_BUS", how="left").drop(columns=["ELECTRICAL_BUS"], errors="ignore")
            top_df["SUBSTATION"] = top_df["SUBSTATION"].fillna("—")
            top_df["SETTLEMENT_LOAD_ZONE"] = top_df["SETTLEMENT_LOAD_ZONE"].fillna("—")

        fig_top = px.bar(
            top_df, x="Bus", y="Spread", color="Spread",
            color_continuous_scale=["#f0d7d9","#c92532","#171717"],
            labels={"Spread":"LMP Spread ($/MWh)"}, template="plotly_white",
            title=f"Top {top_n} Buses by LMP Spread — {top_date}"
        )
        fig_top.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False, height=420)
        st.plotly_chart(fig_top, use_container_width=True)

        fig_sc = px.scatter(
            top_df, x="Spread", y="5H Window Proxy Spread ($/MWh)", text="Bus", size="Spread",
            color="5H Window Proxy Spread ($/MWh)", color_continuous_scale=["#dededb","#c92532","#171717"],
            template="plotly_white", title="Spread vs 5H Quick rolling-average proxy (not optimised dispatch revenue)",
            labels={"Spread":"LMP Spread ($/MWh)","5H Window Proxy Spread ($/MWh)":"5H Window Proxy Spread ($/MWh)"}
        )
        fig_sc.update_traces(textposition="top center", textfont=dict(size=8))
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

        def _style_spread(val):
            if val>80:  return "background-color:#f8e4e5;color:#9e1722"
            if val>40:  return "background-color:#f1f1ee;color:#555555"
            return "background-color:#ffffff;color:#171717"
        st.dataframe(
            top_df.style.applymap(_style_spread, subset=["Spread"]),
            use_container_width=True, height=320
        )
        if not top_df.empty:
            best_bus = top_df.iloc[0]["Bus"]
            if st.button(f"📈  Analyse top bus: {best_bus}"):
                st.session_state.selected_bus = best_bus
                st.info(f"Switched to **{best_bus}** — open the 'BESS Duration Assessment' tab.")

    # ═══ TAB 4 – Export ═══════════════════════
    with tab4:
        st.markdown('<div class="section-header">Export quick spread proxies for all buses</div>', unsafe_allow_html=True)
        dates_exp = sorted(df["Date"].unique().tolist())
        exp_date  = st.selectbox("Select Date for Export", dates_exp, key="exp_date") if len(dates_exp)>1 else dates_exp[0]

        if st.button("⚙️  Compute quick proxy for All Buses", type="primary"):
            with st.spinner("Calculating quick rolling-average proxies for all buses …"):
                exp_rows = []
                for bus_name in df["Bus"].unique():
                    bdf_e = (df[(df["Bus"]==bus_name) & (df["Date"]==exp_date)]
                             .sort_values("Hour").reset_index(drop=True))
                    if len(bdf_e) < 3:
                        continue
                    r2, _, lh, hh, cw2, dw2 = bess_calc(bdf_e, 1)
                    r4, _, _,  _,  cw4, dw4 = bess_calc(bdf_e, 2)
                    sp = round(bdf_e["LMP"].max() - bdf_e["LMP"].min(), 2)
                    rec = "Use the BESS Duration Assessment optimiser; spread alone is not a feasibility verdict"

                    row_data = {
                        "Date": exp_date, "Bus": bus_name,
                        "Min LMP": round(bdf_e["LMP"].min(),2),
                        "Max LMP": round(bdf_e["LMP"].max(),2),
                        "Avg LMP": round(bdf_e["LMP"].mean(),2),
                        "Spread ($/MWh)": sp,
                        "3H Proxy Charge Window":    f"Hr {cw2[0]:.0f}–{cw2[1]:.0f}",
                        "3H Proxy Discharge Window": f"Hr {dw2[0]:.0f}–{dw2[1]:.0f}",
                        "5H Proxy Charge Window":    f"Hr {cw4[0]:.0f}–{cw4[1]:.0f}",
                        "5H Proxy Discharge Window": f"Hr {dw4[0]:.0f}–{dw4[1]:.0f}",
                        "3H Window Proxy Spread ($/MWh)": r2,
                        "5H Window Proxy Spread ($/MWh)": r4,
                        "Screening Note": rec,
                    }

                    # Add substation info
                    if sp_df is not None:
                        bus_sp = sp_df[sp_df["ELECTRICAL_BUS"] == bus_name]
                        if not bus_sp.empty:
                            row_data["Substation"] = bus_sp.iloc[0]["SUBSTATION"]
                            row_data["Load Zone"]  = bus_sp.iloc[0]["SETTLEMENT_LOAD_ZONE"]
                        else:
                            row_data["Substation"] = "—"
                            row_data["Load Zone"]  = "—"

                    exp_rows.append(row_data)

            exp_df = (pd.DataFrame(exp_rows)
                      .sort_values("Spread ($/MWh)", ascending=False)
                      .reset_index(drop=True))
            st.success(f"✅  Computed {len(exp_df)} buses")
            st.dataframe(exp_df, use_container_width=True, height=400)
            st.download_button(
                label="📥  Download CSV",
                data=exp_df.to_csv(index=False).encode("utf-8"),
                file_name=f"Sunstripe_Quick_Spread_Proxy_{exp_date}.csv",
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

    dates_ai   = sorted(df["Date"].unique().tolist())
    bus_list_ai = sorted(df["Bus"].unique().tolist())
    top_buses_ai = (
        df.groupby("Bus")["LMP"]
        .agg(spread=lambda x: x.max()-x.min(), avg="mean", max="max", min="min")
        .sort_values("spread", ascending=False)
        .head(10).round(2).reset_index()
    )

    # Enrich AI context with substation info for top buses
    sub_context = ""
    if sp_df is not None:
        sub_lookup = sp_df.drop_duplicates("ELECTRICAL_BUS")[["ELECTRICAL_BUS","SUBSTATION","SETTLEMENT_LOAD_ZONE"]]
        top_enriched = top_buses_ai.merge(sub_lookup, left_on="Bus", right_on="ELECTRICAL_BUS", how="left")
        sub_context = f"""
Substation mapping for top buses:
{top_enriched[['Bus','SUBSTATION','SETTLEMENT_LOAD_ZONE','spread']].to_string(index=False)}

Settlement Points database: {sp_df['SUBSTATION'].nunique()} unique substations, {len(sp_df)} total bus entries across 4 load zones.
"""

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
{sub_context}

Duration methodology note:
- The Single Bus / BESS Duration Assessment tab uses an hourly price-taker MILP optimisation for 2-hour and 4-hour systems, with SOC, efficiency, charge/discharge power, no simultaneous charging/discharging, daily cyclic SOC, and a configurable EFC/day cap.
- The legacy quick spread ranking/export modules still display a rolling-average proxy; do not treat it as optimised revenue or bankable economics.
- LMP-only margin excludes ancillary services, capacity/tolling revenues, price impact, fixed O&M, and capital cost unless explicitly entered in the duration screen.

Answer concisely and specifically using numbers from the dataset. Focus on actionable BESS development insights.
When the user asks about a bus, also mention which substation it belongs to and its load zone if available.
"""

    st.markdown('<div class="section-header">Suggested Questions</div>', unsafe_allow_html=True)
    suggestions = [
        "Which buses have the best arbitrage opportunity?",
        "What is the recommended BESS strategy for the top bus?",
        "Summarise the overall LMP price trends in this dataset",
        "Which substations have the most buses with high spreads?",
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
