# app.py — Sales Forecasting & Demand Intelligence System
# Consolidated single-file Streamlit dashboard
# Author: AjaiKumar | July 2026
#
# Run with:  streamlit run app.py
# This file is fully self-contained — no external module imports needed.
# All data is loaded directly from train.csv in the same directory.

import faulthandler
faulthandler.enable()

import warnings
warnings.filterwarnings("ignore")

import os
# ── Thread-safety fix: limit OpenBLAS / OMP / MKL to 1 thread ────────────────
# MUST be set BEFORE importing numpy, scipy, or scikit-learn.
# Multi-threaded OpenBLAS is a known cause of SIGSEGV (segfault) on
# Linux containers with limited resources (e.g. Streamlit Cloud free tier).
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# Prophet, CmdStanPy, and scikit-learn are all imported lazily inside their
# respective @st.cache functions.  This prevents native-library segfaults
# and OOM kills during the Streamlit startup / health-check window.

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
CHARTS_DIR = ROOT / "charts"
DATA_PATH  = ROOT / "train.csv"

# ── Design tokens ──────────────────────────────────────────────────────────────
COLORS = ["#38bdf8", "#a855f7", "#f472b6", "#10b981", "#fb923c", "#facc15"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#1e293b",
    font=dict(color="#e2e8f0", family="sans-serif", size=12),
    xaxis=dict(gridcolor="#334155", linecolor="#334155", showgrid=True),
    yaxis=dict(gridcolor="#334155", linecolor="#334155", showgrid=True),
    legend=dict(bgcolor="#1e293b", bordercolor="#475569", borderwidth=1),
    margin=dict(l=60, r=20, t=50, b=60),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#475569", font_color="#e2e8f0"),
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
*, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif !important; }
[data-testid="stAppViewContainer"] { padding-top: 1rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
.sidebar-brand {
    font-size: 1.05rem; font-weight: 700; color: #38bdf8;
    letter-spacing: 0.04em; padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #334155; margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-card .metric-label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 0.3rem;
}
.metric-card .metric-value { font-size: 1.9rem; font-weight: 800; color: #f1f5f9; line-height: 1; }
.metric-card .metric-delta { font-size: 0.82rem; margin-top: 0.35rem; }
.metric-card .metric-delta.positive { color: #10b981; }
.metric-card .metric-delta.negative { color: #f87171; }
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #38bdf8;
    letter-spacing: 0.04em; text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px solid #334155;
}
.insight-box {
    background: rgba(56,189,248,0.08); border-left: 3px solid #38bdf8;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.75rem 0;
    font-size: 0.92rem; color: #cbd5e1;
}
.warning-box {
    background: rgba(248,113,113,0.08); border-left: 3px solid #f87171;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.75rem 0;
    font-size: 0.92rem; color: #fca5a5;
}
.success-box {
    background: rgba(16,185,129,0.08); border-left: 3px solid #10b981;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.75rem 0;
    font-size: 0.92rem; color: #6ee7b7;
}
.hr-styled { border: none; border-top: 1px solid #334155; margin: 1.5rem 0; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def hr():
    st.markdown('<hr class="hr-styled">', unsafe_allow_html=True)

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def metric_card(label, value, delta="", positive=True):
    delta_class = "positive" if positive else "negative"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True,
    )

def page_title(icon, title, subtitle=""):
    sub_html = f'<p style="color:#94a3b8;font-size:0.95rem;margin-top:0.25rem;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div style="padding:0.5rem 0 1rem 0;border-bottom:1px solid #334155;margin-bottom:1.5rem;">
            <h1 style="margin:0;font-size:2rem;color:#f1f5f9;">{icon} {title}</h1>
            {sub_html}
        </div>
        """, unsafe_allow_html=True,
    )

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def warning_box(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def success_box(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)


def show_chart(filename, caption=""):
    path = CHARTS_DIR / filename
    if path.exists():
        try:
            st.image(str(path), caption=caption, use_column_width=True)
        except Exception:
            st.info(f"📊 Chart loading… please refresh the page if this persists. (`{filename}`)")
    else:
        st.info(f"📊 Chart `{filename}` is not available in this deployment.")



def download_csv(df, filename, label="⬇️ Download CSV"):
    st.download_button(label=label, data=df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

def download_plotly_html(fig, filename, label="⬇️ Download Chart (HTML)"):
    st.download_button(label=label,
                       data=fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                       file_name=filename, mime="text/html")


# ══════════════════════════════════════════════════════════════════════════════
# CMDSTAN STARTUP — install once, cached as a resource
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Installing CmdStan (first-time only)…")
def _ensure_cmdstan():
    """
    Installs CmdStan into a writable cache directory if it is not already
    available.  On Streamlit Cloud this runs exactly once per deployment;
    the result is held in the resource cache for the lifetime of the process.
    Returns True on success, False on failure (caller shows error message).

    NOTE: cmdstanpy.cmdstan_path() RAISES an exception (not returns None/False)
    when CmdStan is not found. The nested try/except handles this correctly:
      - inner try  → if cmdstan_path() succeeds, CmdStan is already present
      - inner except → CmdStan is missing, so install it
    """
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    try:
        import cmdstanpy
        try:
            path = cmdstanpy.cmdstan_path()   # raises if not installed
            os.environ["CMDSTAN"] = path
            return True
        except Exception:
            # CmdStan is not installed — install it now
            cmdstanpy.install_cmdstan(overwrite=False, verbose=False)
            path = cmdstanpy.cmdstan_path()
            os.environ["CMDSTAN"] = path
            return True
    except Exception as exc:
        # Non-fatal — Prophet pages will show a clear error via try/except
        logging.warning(f"CmdStan installation failed: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (@st.cache_data for pure data, @st.cache_resource for models)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading Superstore data…")
def get_cleaned_train():
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date", "Ship Date"], dayfirst=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["shipping_delay"] = (df["ship_date"] - df["order_date"]).dt.days
    df["year"]    = df["order_date"].dt.year
    df["month"]   = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.quarter
    return df

@st.cache_data(show_spinner=False)
def get_monthly_sales():
    df = get_cleaned_train().copy()
    df["month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby("month_start")["sales"].sum().reset_index().rename(columns={"month_start": "date"})

@st.cache_data(show_spinner=False)
def get_category_monthly():
    df = get_cleaned_train().copy()
    df["month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby(["month_start", "category"])["sales"].sum().reset_index().rename(columns={"month_start": "date"})

@st.cache_data(show_spinner=False)
def get_region_monthly():
    df = get_cleaned_train().copy()
    df["month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby(["month_start", "region"])["sales"].sum().reset_index().rename(columns={"month_start": "date"})


# ── Prophet helper utilities ────────────────────────────────────────────────────

def _daily_from_trans(df_sub, df_all):
    """Aggregate sub-DataFrame to a daily ds/y series spanning the full date range."""
    all_dates = pd.date_range(
        start=df_all["order_date"].min(),
        end=df_all["order_date"].max(),
        freq="D"
    )
    daily = df_sub.groupby("order_date")["sales"].sum()
    return (
        daily.reindex(all_dates, fill_value=0.0)
             .reset_index()
             .rename(columns={"index": "ds", "order_date": "ds", "sales": "y"})
    )

def _monthly_split(forecast, cutoff):
    """Split a Prophet forecast DataFrame into historical fitted and future portions."""
    fc = forecast.copy()
    fc["month"] = fc["ds"].dt.to_period("M").dt.to_timestamp()
    hist_fc = (
        fc[fc["ds"] <= cutoff]
        .groupby("month")[["yhat"]].sum()
        .reset_index()
        .rename(columns={"month": "date"})
    )
    fut_fc = (
        fc[fc["ds"] > cutoff]
        .groupby("month")[["yhat", "yhat_lower", "yhat_upper"]].sum()
        .reset_index()
        .rename(columns={"month": "date"})
    )
    for col in ["yhat_lower", "yhat_upper"]:
        fut_fc[col] = fut_fc[col].clip(lower=0)
    return hist_fc, fut_fc


# ── Prophet two-stage cache pattern ────────────────────────────────────────────
#
# Stage 1  →  @st.cache_resource  trains the Prophet model object ONCE.
#             The model is held in memory (not pickled) for the process lifetime.
#             Using a string `model_key` makes the cache key lightweight & hashable.
#
# Stage 2  →  @st.cache_data  calls model.predict() and returns a plain DataFrame.
#             DataFrames are safely pickle-serialised by Streamlit Cloud.
#             The leading underscore on `_model` tells Streamlit to skip hashing
#             that argument (Prophet objects are not hashable).

@st.cache_resource(show_spinner="Training Prophet model (first-time only)…")
def _train_prophet_model(model_key: str):
    """
    Train a Prophet model identified by *model_key*.

    model_key must be one of:
      "overall"          — full-dataset model
      "cat:<category>"   — per-category model
      "reg:<region>"     — per-region model

    CmdStan and Prophet are imported HERE (lazily) so they never touch
    the process at startup — avoiding native-library segfaults and OOM
    kills before the health-check endpoint is ready.
    The Prophet object is stored in the resource cache and reused on every
    subsequent page navigation without retraining.
    """
    # ── Ensure CmdStan is available before importing/calling Prophet ────────
    # _ensure_cmdstan() is itself @st.cache_resource, so the install/check
    # runs exactly once per process lifetime regardless of how many models
    # are trained.
    cmdstan_ok = _ensure_cmdstan()
    if not cmdstan_ok:
        raise RuntimeError(
            "CmdStan could not be installed on this deployment. "
            "Prophet model training is unavailable. "
            "Check the app logs for the exact installation error."
        )

    # ── Lazy imports — executed ONLY after CmdStan is guaranteed to be present ──
    import cmdstanpy as _cmdstanpy          # noqa: F401 (needed for Prophet)
    from prophet import Prophet             # noqa: F811


    df = get_cleaned_train()

    if model_key == "overall":
        df_sub = df
    elif model_key.startswith("cat:"):
        category = model_key[4:]
        df_sub = df[df["category"] == category]
    elif model_key.startswith("reg:"):
        region = model_key[4:]
        df_sub = df[df["region"] == region]
    else:
        raise ValueError(f"Unknown model_key: {model_key!r}")

    daily = _daily_from_trans(df_sub, df)

    m = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(daily)
    return m


@st.cache_data(show_spinner=False)
def _generate_forecast(_model, model_key: str, periods: int = 90):
    """
    Generate a Prophet forecast DataFrame.

    *_model* is excluded from hashing (leading underscore).
    *model_key* and *periods* are the actual cache discriminators.
    Returns the raw forecast DataFrame.
    """
    future = _model.make_future_dataframe(periods=periods, freq="D")
    return _model.predict(future)


# ── High-level Prophet forecast accessors ──────────────────────────────────────

@st.cache_data(show_spinner="Running overall Prophet forecast (first-time only)…")
def get_prophet_overall():
    """Return (hist_monthly_df, future_monthly_df) for the full dataset."""
    model    = _train_prophet_model("overall")
    forecast = _generate_forecast(model, "overall", periods=90)

    df = get_cleaned_train()
    cutoff = df["order_date"].max()

    daily = _daily_from_trans(df, df)
    daily["month"] = daily["ds"].dt.to_period("M").dt.to_timestamp()
    hist = (
        daily.groupby("month")["y"].sum()
             .reset_index()
             .rename(columns={"month": "date", "y": "sales"})
    )
    _, fut = _monthly_split(forecast, cutoff)
    return hist, fut


@st.cache_data(show_spinner="Running category forecast…")
def get_prophet_category(category: str):
    """Return (hist_monthly_df, future_monthly_df) for a single category."""
    model_key = f"cat:{category}"
    model     = _train_prophet_model(model_key)
    forecast  = _generate_forecast(model, model_key, periods=90)

    df = get_cleaned_train()
    cutoff = df["order_date"].max()

    daily = _daily_from_trans(df[df["category"] == category], df)
    daily["month"] = daily["ds"].dt.to_period("M").dt.to_timestamp()
    hist = (
        daily.groupby("month")["y"].sum()
             .reset_index()
             .rename(columns={"month": "date", "y": "sales"})
    )
    _, fut = _monthly_split(forecast, cutoff)
    return hist, fut


@st.cache_data(show_spinner="Running region forecast…")
def get_prophet_region(region: str):
    """Return (hist_monthly_df, future_monthly_df) for a single region."""
    model_key = f"reg:{region}"
    model     = _train_prophet_model(model_key)
    forecast  = _generate_forecast(model, model_key, periods=90)

    df = get_cleaned_train()
    cutoff = df["order_date"].max()

    daily = _daily_from_trans(df[df["region"] == region], df)
    daily["month"] = daily["ds"].dt.to_period("M").dt.to_timestamp()
    hist = (
        daily.groupby("month")["y"].sum()
             .reset_index()
             .rename(columns={"month": "date", "y": "sales"})
    )
    _, fut = _monthly_split(forecast, cutoff)
    return hist, fut


@st.cache_data(show_spinner="Computing anomaly detection…")
def get_anomaly_data():
    # Lazy sklearn import — deferred to avoid C-runtime init at startup
    from sklearn.ensemble import IsolationForest  # noqa: F401

    df = get_cleaned_train()
    wk = df.groupby(pd.Grouper(key="order_date", freq="W-MON"))["sales"].sum().reset_index().rename(columns={"order_date": "date"})
    wk["roll_mean"]   = wk["sales"].rolling(8, min_periods=4, center=True).mean()
    wk["roll_std"]    = wk["sales"].rolling(8, min_periods=4, center=True).std()
    wk["z_score"]     = (wk["sales"] - wk["roll_mean"]) / wk["roll_std"]
    wk["zscore_flag"] = wk["z_score"].abs() > 2.0
    wk["roll_median"] = wk["sales"].rolling(4, min_periods=1, center=True).median()
    wk["deviation"]   = wk["sales"] - wk["roll_median"]
    wk["month"]       = wk["date"].dt.month
    X = wk[["sales", "deviation", "month"]].values
    wk["iforest_flag"] = IsolationForest(
        contamination=0.04, random_state=42, n_estimators=100, n_jobs=1
    ).fit_predict(X) == -1
    wk["expected"] = wk["sales"].rolling(5, center=True, min_periods=1).median()
    wk["dev_pct"]  = ((wk["sales"] - wk["expected"]) / wk["expected"].replace(0, np.nan) * 100).round(1)
    wk["year"]     = wk["date"].dt.year
    return wk

@st.cache_data(show_spinner="Computing product segmentation…")
def get_segmentation_data():
    # Lazy sklearn imports — deferred to avoid C-runtime init at startup
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = get_cleaned_train().copy()
    agg = df.groupby("sub-category").agg(
        total_sales=("sales", "sum"), purchase_frequency=("order_id", "nunique"), aov=("sales", "mean")
    ).reset_index()
    df["month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    ms = df.groupby(["sub-category", "month_start"])["sales"].sum().reset_index()
    all_months = pd.date_range(df["order_date"].min(), df["order_date"].max(), freq="MS")
    vols = [{"sub-category": s, "sales_volatility": ms[ms["sub-category"] == s].set_index("month_start").reindex(all_months, fill_value=0)["sales"].std()}
            for s in agg["sub-category"]]
    agg = pd.merge(agg, pd.DataFrame(vols), on="sub-category")
    ys = df.groupby(["sub-category", "year"])["sales"].sum().unstack(fill_value=0).reset_index()
    ys["growth_rate"] = 0.0
    if 2017 in ys.columns and 2018 in ys.columns:
        m = ys[2017] > 0
        ys.loc[m, "growth_rate"] = (ys.loc[m, 2018] - ys.loc[m, 2017]) / ys.loc[m, 2017]
    agg = pd.merge(agg, ys[["sub-category", "growth_rate"]], on="sub-category")
    feats = ["total_sales", "purchase_frequency", "aov", "sales_volatility", "growth_rate"]
    X = agg[feats].values
    Xs = StandardScaler().fit_transform(X)
    agg["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(Xs)
    cm = agg.groupby("cluster").mean(numeric_only=True)
    rem = list(cm.index)
    lmap = {}
    for label, col in [("High Value Stable", "purchase_frequency"), ("Growing Demand", "growth_rate"), ("Seasonal Products", "sales_volatility")]:
        idx = cm.loc[rem, col].idxmax()
        lmap[idx] = label
        rem.remove(idx)
    if rem:
        lmap[rem[0]] = "Declining Products"
    agg["segment"]    = agg["cluster"].map(lmap)
    agg["growth_pct"] = (agg["growth_rate"] * 100).round(2)
    for c in ["total_sales", "aov", "sales_volatility"]:
        agg[c] = agg[c].round(2)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(Xs)
    agg["pca_x"] = pca_coords[:, 0].round(4)
    agg["pca_y"] = pca_coords[:, 1].round(4)
    agg["pca_var"] = f"{pca.explained_variance_ratio_[0]*100:.1f}% / {pca.explained_variance_ratio_[1]*100:.1f}%"
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈", layout="wide", initial_sidebar_state="expanded",
)
inject_css()

# NOTE: CmdStan is installed lazily — see _train_prophet_model() — so that
# the Streamlit health-check endpoint starts responding immediately and is
# not blocked by a multi-minute CmdStan download/compilation.

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            📈 Sales Intelligence<br>
            <span style="font-size:0.75rem;color:#64748b;font-weight:400;">
                Demand Forecasting System
            </span>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="font-size:0.78rem;color:#64748b;margin-bottom:1rem;">
            Data Range: Jan 2015 – Dec 2018<br>
            Best Model: <span style="color:#38bdf8;">Facebook Prophet</span><br>
            MAPE: <span style="color:#10b981;">16.04%</span>
        </div>
        """, unsafe_allow_html=True,
    )
    st.divider()
    PAGE = st.radio(
        "Navigate to",
        ["🏠 Home",
         "📊 Sales Dashboard",
         "🔮 Forecast Explorer",
         "🗂️ Category Analysis",
         "🗺️ Region Analysis",
         "🚨 Anomaly Center",
         "🎯 Demand Segmentation",
         "📋 Executive Summary"],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Home":
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);
                    border:1px solid #1e40af;border-radius:16px;padding:2.5rem 2rem;
                    margin-bottom:2rem;text-align:center;">
            <div style="font-size:3rem;">📈</div>
            <h1 style="margin:0.5rem 0 0.25rem 0;font-size:2.4rem;color:#f1f5f9;font-weight:800;letter-spacing:-0.02em;">
                Sales Forecasting &amp; Demand Intelligence
            </h1>
            <p style="color:#94a3b8;font-size:1.05rem;max-width:700px;margin:0 auto;">
                End-to-end machine learning pipeline built on 4 years of Superstore retail data
                (2015–2018) with Prophet, SARIMA, and XGBoost models.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )

    section_header("Project-Wide Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    with c1: metric_card("Best Model MAPE", "16.04%", "Prophet outperforms SARIMA by 35%", positive=True)
    with c2: metric_card("Total Historical Sales", "$2.29M", "Jan 2015 – Dec 2018", positive=True)
    with c3: metric_card("Fastest Growing Region", "South", "+159.92% YoY (Q1 2019)", positive=True)
    with c4: metric_card("Fastest Growing Category", "Furniture", "+54.82% YoY (Q1 2019)", positive=True)
    with c5: metric_card("Anomalies Detected", "9 weeks", "Isolation Forest (4% contamination)", positive=False)
    hr()

    section_header("Historical Sales Overview")
    col_left, col_right = st.columns([2, 1], gap="medium")
    with col_left:
        show_chart("monthly_sales_trend.png", "Monthly Retail Sales 2015–2018")
    with col_right:
        show_chart("sales_by_category.png", "Revenue by Product Category")
    insight("Sales grew consistently from 2015 to 2018, driven by Technology and Furniture. Strong Q4 seasonality is visible each year — November/December peak is ~2× the annual baseline.")
    hr()

    section_header("Model Comparison Snapshot")
    col_m1, col_m2 = st.columns([1, 1], gap="medium")
    with col_m1:
        show_chart("model_comparison_metrics.png", "MAPE & R² — SARIMA vs Prophet vs XGBoost")
    with col_m2:
        st.markdown("#### Why Facebook Prophet Won")
        st.markdown("""
| Model | RMSE | MAE | MAPE | R² |
|:------|-----:|----:|-----:|---:|
| **Prophet** | $14,718 | $10,382 | **16.04%** | **0.675** |
| SARIMA | $16,013 | $13,480 | 24.91% | 0.615 |
| XGBoost | $21,947 | $17,676 | 27.04% | 0.277 |
        """)
        success_box("Prophet captures both the weekly B2B cycle and the Q4 holiday surge, delivering 35% lower forecast error than SARIMA.")
    hr()

    section_header("Explore the Dashboard")
    nav_cols = st.columns(4, gap="medium")
    pages = [
        ("📊", "Sales Dashboard", "Monthly & category trends, segment splits, shipping delays"),
        ("🔮", "Forecast Explorer", "SARIMA · Prophet · XGBoost interactive forecast viewer"),
        ("🗂️", "Category Analysis", "Furniture · Technology · Office Supplies Q1 2019 forecasts"),
        ("🗺️", "Region Analysis",  "East · West · Central · South growth rankings & allocation"),
        ("🚨", "Anomaly Center",   "Isolation Forest & Z-Score weekly anomaly detection"),
        ("🎯", "Demand Segmentation", "K-Means product clusters with inventory recommendations"),
        ("📋", "Executive Summary", "Business recommendations backed by model outputs"),
    ]
    for i, (icon, title, desc) in enumerate(pages):
        with nav_cols[i % 4]:
            st.markdown(
                f"""
                <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                            padding:1rem 1.1rem;margin-bottom:0.75rem;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-weight:700;color:#f1f5f9;margin:0.3rem 0 0.2rem 0;">{title}</div>
                    <div style="font-size:0.82rem;color:#64748b;">{desc}</div>
                </div>
                """, unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SALES DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊 Sales Dashboard":
    page_title("📊", "Sales Dashboard", "Historical sales performance — Jan 2015 to Dec 2018")
    df_raw = get_cleaned_train()

    with st.sidebar:
        st.markdown("### 🔎 Filters")
        years = sorted(df_raw["year"].unique())
        year_range = st.slider("Year range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
        cats     = st.multiselect("Category", sorted(df_raw["category"].unique()), default=sorted(df_raw["category"].unique()))
        regions  = st.multiselect("Region",   sorted(df_raw["region"].unique()),   default=sorted(df_raw["region"].unique()))
        segments = st.multiselect("Segment",  sorted(df_raw["segment"].unique()),  default=sorted(df_raw["segment"].unique()))

    mask = (df_raw["year"].between(year_range[0], year_range[1]) &
            df_raw["category"].isin(cats) & df_raw["region"].isin(regions) & df_raw["segment"].isin(segments))
    df = df_raw[mask].copy()

    section_header("Portfolio KPIs")
    total_sales  = df["sales"].sum()
    total_orders = df["order_id"].nunique()
    avg_ov       = df.groupby("order_id")["sales"].sum().mean()
    avg_delay    = df["shipping_delay"].mean()
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Total Revenue", f"${total_sales:,.0f}", f"{total_orders:,} orders")
    with c2: metric_card("Unique Orders", f"{total_orders:,}", f"{df['customer_id'].nunique():,} customers")
    with c3: metric_card("Avg Order Value", f"${avg_ov:,.2f}", "Per unique order")
    with c4: metric_card("Avg Shipping Delay", f"{avg_delay:.1f} days", "All ship modes", positive=False)
    hr()

    section_header("Monthly Sales Trend")
    df["month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month_start")["sales"].sum().reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly["month_start"], y=monthly["sales"], mode="lines+markers",
        name="Monthly Sales", line=dict(color=COLORS[0], width=2.5), marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>",
    ))
    fig_trend.update_layout(**PLOTLY_LAYOUT, title="Monthly Sales", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    col_c, col_d = st.columns([5, 1])
    with col_c:
        st.plotly_chart(fig_trend, use_container_width=True)
    with col_d:
        st.markdown("<br><br>", unsafe_allow_html=True)
        download_plotly_html(fig_trend, "monthly_sales_trend.html")
        download_csv(monthly.rename(columns={"month_start": "date"}), "monthly_sales.csv")
    hr()

    section_header("Revenue Breakdown")
    col_cat, col_reg = st.columns(2, gap="medium")
    with col_cat:
        cat_sales = df.groupby("category")["sales"].sum().reset_index().sort_values("sales", ascending=False)
        fig_cat = px.bar(cat_sales, x="category", y="sales", color="category",
                         color_discrete_sequence=COLORS, title="Sales by Category")
        fig_cat.update_layout(**PLOTLY_LAYOUT, showlegend=False, yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_cat, use_container_width=True)
    with col_reg:
        reg_sales = df.groupby("region")["sales"].sum().reset_index().sort_values("sales", ascending=False)
        fig_reg = px.bar(reg_sales, x="region", y="sales", color="region",
                         color_discrete_sequence=COLORS[1:], title="Sales by Region")
        fig_reg.update_layout(**PLOTLY_LAYOUT, showlegend=False, yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig_reg, use_container_width=True)
    hr()

    section_header("Segment Share & Shipping Performance")
    col_seg, col_ship = st.columns(2, gap="medium")
    with col_seg:
        seg_sales = df.groupby("segment")["sales"].sum().reset_index()
        fig_pie = px.pie(seg_sales, names="segment", values="sales",
                         color_discrete_sequence=COLORS, title="Revenue by Customer Segment", hole=0.45)
        fig_pie.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]})
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_ship:
        ship_delay = df.groupby("ship_mode")["shipping_delay"].mean().reset_index().sort_values("shipping_delay")
        fig_ship = px.bar(ship_delay, x="shipping_delay", y="ship_mode", orientation="h",
                          color="shipping_delay", color_continuous_scale="Blues", title="Avg Shipping Delay by Mode (days)")
        fig_ship.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_ship, use_container_width=True)
    hr()

    section_header("Category × Region Sales Heatmap")
    pivot = df.pivot_table(index="region", columns="category", values="sales", aggfunc="sum").fillna(0)
    fig_heat = px.imshow(pivot, color_continuous_scale="Blues", title="Sales Heatmap — Region vs Category",
                         text_auto=".2s", aspect="auto")
    fig_heat.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]})
    st.plotly_chart(fig_heat, use_container_width=True)
    insight("Technology dominates every region. South shows proportionally stronger Furniture demand relative to its total size.")
    hr()

    section_header("Filtered Transaction Data")
    search = st.text_input("🔍 Search product name", placeholder="e.g. 'Chair', 'Phone'…")
    display_df = df[["order_date", "category", "sub-category", "region", "segment",
                      "ship_mode", "sales", "shipping_delay", "product_name"]].copy()
    for col in ["category", "sub-category", "region", "segment", "ship_mode", "product_name"]:
        display_df[col] = display_df[col].astype(str)
    if search:
        display_df = display_df[display_df["product_name"].str.contains(search, case=False, na=False)]
    st.caption(f"Showing {len(display_df):,} rows")
    st.dataframe(display_df.sort_values("order_date", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=320,
                 column_config={
                     "sales": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                     "shipping_delay": st.column_config.NumberColumn("Ship Delay (days)", format="%d"),
                     "order_date": st.column_config.DateColumn("Order Date", format="MMM D, YYYY"),
                 })
    download_csv(display_df, "filtered_transactions.csv", "⬇️ Download Filtered Data (CSV)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔮 Forecast Explorer":
    page_title("🔮", "Forecast Explorer", "Interactive Prophet forecast · 2018 back-test & Q1 2019 outlook")

    MODEL_METRICS = {
        "Prophet":  {"RMSE": 14718.11, "MAE": 10382.49, "MAPE": 16.04, "R2": 0.6748, "train_ms": 217.0, "pred_ms": 178.3},
        "SARIMA":   {"RMSE": 16012.77, "MAE": 13479.57, "MAPE": 24.91, "R2": 0.6151, "train_ms":  98.8, "pred_ms":   1.4},
        "XGBoost":  {"RMSE": 21946.53, "MAE": 17675.65, "MAPE": 27.04, "R2": 0.2769, "train_ms": 704.9, "pred_ms":  56.9},
    }

    section_header("Model Comparison — 2018 Out-of-Sample Test")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Best MAPE", "16.04%", "Prophet — 35% lower than SARIMA", positive=True)
    with c2: metric_card("Best RMSE", "$14,718", "Prophet", positive=True)
    with c3: metric_card("Best R²", "0.675", "Prophet explains 67.5% variance", positive=True)
    with c4: metric_card("Fastest Train", "98.8 ms", "SARIMA — 7× faster than XGBoost", positive=True)
    hr()

    section_header("Prophet Forecast — Interactive Chart with 95% Confidence Bands")
    try:
        with st.spinner("Computing Prophet forecast (cached after first run)…"):
            hist, fut = get_prophet_overall()
    except Exception as _prophet_exc:
        st.error("⚠️ Prophet forecast model failed during this run.")
        st.exception(_prophet_exc)
        st.stop()
    monthly = get_monthly_sales()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["sales"], mode="lines",
        name="Actual (Historical)", line=dict(color=COLORS[0], width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["yhat"], mode="lines+markers",
        name="Q1 2019 Forecast", line=dict(color="#10b981", width=2.5, dash="dash"),
        marker=dict(size=6), hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=pd.concat([fut["date"], fut["date"][::-1]]).tolist(),
        y=pd.concat([fut["yhat_upper"], fut["yhat_lower"][::-1]]).tolist(),
        fill="toself", fillcolor="rgba(16,185,129,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% Confidence Interval", hoverinfo="skip"))
    fig.add_vline(x="2018-12-31", line_dash="dot", line_color="#475569")
    fig.add_annotation(x="2018-12-31", y=0.95, yref="paper", text="Forecast Start →", showarrow=False, font=dict(color="#94a3b8"), xanchor="left")
    fig.update_layout(**PLOTLY_LAYOUT, title="Prophet — Full Historical + Q1 2019 Forecast",
                      yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=450)
    col_c, col_d = st.columns([5, 1])
    with col_c:
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        download_plotly_html(fig, "prophet_forecast_interactive.html")
        download_csv(fut, "prophet_q1_2019_forecast.csv", "⬇️ Download Forecast (CSV)")
    success_box("Prophet is the recommended production model — lowest MAPE (16.04%) with full interpretable seasonal decomposition.")
    hr()

    section_header("Full Model Comparison Table")
    metrics_df = pd.DataFrame(MODEL_METRICS).T.reset_index().rename(columns={"index": "Model"})
    metrics_df.columns = ["Model", "RMSE ($)", "MAE ($)", "MAPE (%)", "R²", "Train Time (ms)", "Predict Time (ms)"]
    st.dataframe(
        metrics_df,
        column_config={
            "RMSE ($)": st.column_config.NumberColumn("RMSE ($)", format="$%,.2f"),
            "MAE ($)": st.column_config.NumberColumn("MAE ($)", format="$%,.2f"),
            "MAPE (%)": st.column_config.NumberColumn("MAPE (%)", format="%.2f%%"),
            "R²": st.column_config.NumberColumn("R²", format="%.4f"),
            "Train Time (ms)": st.column_config.NumberColumn("Train Time (ms)", format="%.1f"),
            "Predict Time (ms)": st.column_config.NumberColumn("Predict Time (ms)", format="%.1f"),
        },
        use_container_width=True,
        hide_index=True,
    )
    download_csv(metrics_df, "model_comparison_metrics.csv", "⬇️ Download Metrics (CSV)")
    hr()

    section_header("MAPE & R² Bar Comparison")
    col_m, col_r = st.columns(2, gap="medium")
    with col_m:
        fig_mape = go.Figure(go.Bar(x=["SARIMA", "Prophet", "XGBoost"], y=[24.91, 16.04, 27.04],
            marker_color=[COLORS[2], COLORS[0], COLORS[3]], text=["24.91%", "16.04%", "27.04%"], textposition="outside"))
        fig_mape.update_layout(**PLOTLY_LAYOUT, title="MAPE (%) — Lower is Better", yaxis_title="MAPE (%)", height=350)
        st.plotly_chart(fig_mape, use_container_width=True)
    with col_r:
        fig_r2 = go.Figure(go.Bar(x=["SARIMA", "Prophet", "XGBoost"], y=[0.6151, 0.6748, 0.2769],
            marker_color=[COLORS[2], COLORS[0], COLORS[3]], text=["0.6151", "0.6748", "0.2769"], textposition="outside"))
        fig_r2.update_layout(**PLOTLY_LAYOUT, title="R² Score — Higher is Better", yaxis_title="R²", height=350)
        st.plotly_chart(fig_r2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CATEGORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🗂️ Category Analysis":
    page_title("🗂️", "Category Analysis", "Prophet Q1 2019 forecasts — Furniture · Technology · Office Supplies")

    CATEGORY_META = {
        "Furniture":       {"yoy": 54.82, "q1_2018": 23597.98,  "q4_2018": 90348.25,  "q1_2019": 36533.56,  "color": COLORS[0]},
        "Technology":      {"yoy":  5.03, "q1_2018": 55690.40,  "q4_2018": 104249.68, "q1_2019": 58493.45,  "color": COLORS[1]},
        "Office Supplies": {"yoy":  9.55, "q1_2018": 42972.50,  "q4_2018": 83818.75,  "q1_2019": 47078.06,  "color": COLORS[2]},
    }

    section_header("Q1 2019 Forecast Summary")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Fastest Growing", "Furniture", "+54.82% YoY", positive=True)
    with c2: metric_card("Furniture Q1 2019", "$36,534", "vs $23,598 Q1 2018", positive=True)
    with c3: metric_card("Technology Q1 2019", "$58,493", "+5.03% YoY · Largest volume", positive=True)
    with c4: metric_card("Office Supplies Q1 2019", "$47,078", "+9.55% YoY", positive=True)
    hr()

    section_header("Individual Category Forecast")
    cat = st.selectbox("Select a category", list(CATEGORY_META.keys()), key="cat_sel")
    meta = CATEGORY_META[cat]
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1: metric_card("YoY Growth", f"+{meta['yoy']:.2f}%", "", positive=True)
    with c2: metric_card("Q1 2019 Forecast", f"${meta['q1_2019']:,.2f}", "Jan + Feb + Mar 2019")
    with c3: metric_card("Q4 2018 Actual", f"${meta['q4_2018']:,.2f}", "Holiday peak baseline")

    try:
        with st.spinner(f"Computing {cat} forecast…"):
            hist, fut = get_prophet_category(cat)
    except Exception as _prophet_exc:
        st.error(f"⚠️ Prophet category forecast failed for '{cat}'.")
        st.exception(_prophet_exc)
        st.stop()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], mode="lines", name="Historical Actual",
        line=dict(color=meta["color"], width=2), hovertemplate="<b>%{x|%b %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["yhat"], mode="lines+markers", name="Q1 2019 Forecast",
        line=dict(color="#10b981", width=2.5, dash="dash"), marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=pd.concat([fut["date"], fut["date"][::-1]]).tolist(),
        y=pd.concat([fut["yhat_upper"], fut["yhat_lower"][::-1]]).tolist(),
        fill="toself", fillcolor="rgba(16,185,129,0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", hoverinfo="skip"))
    fig.add_vline(x="2018-12-31", line_dash="dot", line_color="#475569")
    fig.add_annotation(x="2018-12-31", y=0.95, yref="paper", text="Forecast →", showarrow=False, font=dict(color="#94a3b8"), xanchor="left")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{cat} — Monthly Sales & Q1 2019 Forecast",
                      yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=420)
    col_c, col_d = st.columns([5, 1])
    with col_c:
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        download_plotly_html(fig, f"forecast_{cat.replace(' ', '_').lower()}.html")
        fc_dl = pd.concat([hist.rename(columns={"sales": "actual_sales"}), fut[["date", "yhat", "yhat_lower", "yhat_upper"]]])
        download_csv(fc_dl, f"forecast_{cat.replace(' ', '_').lower()}.csv", "⬇️ Download Forecast (CSV)")
    hr()

    section_header("All Category Forecasts Compared")
    all_fut_rows = [{"Category": c_name, "Q1 2018 Actual": c_meta["q1_2018"],
                     "Q4 2018 Actual": c_meta["q4_2018"], "Q1 2019 Forecast": c_meta["q1_2019"],
                     "YoY Growth (%)": c_meta["yoy"]} for c_name, c_meta in CATEGORY_META.items()]
    compare_df = pd.DataFrame(all_fut_rows)
    fig_cmp = px.bar(
        compare_df.melt(id_vars="Category", value_vars=["Q1 2018 Actual", "Q4 2018 Actual", "Q1 2019 Forecast"],
                        var_name="Period", value_name="Sales ($)"),
        x="Category", y="Sales ($)", color="Period", barmode="group",
        color_discrete_sequence=[COLORS[2], COLORS[1], COLORS[0]],
        title="Q1 2018 · Q4 2018 · Q1 2019 — Category Comparison",
    )
    fig_cmp.update_layout(**PLOTLY_LAYOUT, yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=380)
    st.plotly_chart(fig_cmp, use_container_width=True)
    insight("Furniture's Q1 2019 forecast nearly doubles its Q1 2018 baseline. The QoQ drop from Q4 is seasonal — compare YoY for true growth signal.")
    hr()

    section_header("YoY & QoQ Growth Matrix")
    st.dataframe(
        compare_df,
        column_config={
            "Q1 2018 Actual": st.column_config.NumberColumn("Q1 2018 Actual", format="$%,.2f"),
            "Q4 2018 Actual": st.column_config.NumberColumn("Q4 2018 Actual", format="$%,.2f"),
            "Q1 2019 Forecast": st.column_config.NumberColumn("Q1 2019 Forecast", format="$%,.2f"),
            "YoY Growth (%)": st.column_config.NumberColumn("YoY Growth (%)", format="+%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )
    download_csv(compare_df, "category_growth_comparison.csv", "⬇️ Download Growth Table (CSV)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REGION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🗺️ Region Analysis":
    page_title("🗺️", "Region Analysis", "Separate Prophet Q1 2019 forecasts by region · allocation & risk")

    REGION_META = {
        "South":   {"yoy": 159.92, "q1_2018": 13642.18, "q4_2018": 55778.02, "q1_2019": 35459.32, "alloc": 23.73, "rank": 1, "color": COLORS[3]},
        "East":    {"yoy":  95.25, "q1_2018": 18042.82, "q4_2018": 97214.36, "q1_2019": 35227.75, "alloc": 23.57, "rank": 2, "color": COLORS[1]},
        "West":    {"yoy":  -0.20, "q1_2018": 50540.72, "q4_2018": 79573.39, "q1_2019": 50437.55, "alloc": 33.75, "rank": 3, "color": COLORS[0]},
        "Central": {"yoy": -29.27, "q1_2018": 40035.16, "q4_2018": 45850.90, "q1_2019": 28318.25, "alloc": 18.95, "rank": 4, "color": COLORS[2]},
    }

    section_header("Q1 2019 Regional Snapshot")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Fastest Growth (YoY)", "South", "+159.92% — Surge market", positive=True)
    with c2: metric_card("Volume Leader", "West", "$50,438 · 33.75% of Q1 demand", positive=True)
    with c3: metric_card("Contracting", "Central", "-29.27% YoY · Reduce stock", positive=False)
    with c4: metric_card("Total Q1 Forecast", "$149,443", "All four regions combined", positive=True)
    hr()

    section_header("Individual Region Forecast")
    reg = st.selectbox("Select a region", list(REGION_META.keys()), key="reg_sel")
    meta = REGION_META[reg]
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("YoY Growth", f"{meta['yoy']:+.2f}%", "Q1 2019 vs Q1 2018", positive=meta["yoy"] > 0)
    with c2: metric_card("Q1 2019 Forecast", f"${meta['q1_2019']:,.2f}", "Jan + Feb + Mar 2019")
    with c3: metric_card("Allocation Weight", f"{meta['alloc']:.2f}%", "Share of total Q1 demand")
    with c4: metric_card("Growth Rank", f"#{meta['rank']}", "Among 4 regions", positive=meta["rank"] <= 2)

    try:
        with st.spinner(f"Computing {reg} forecast…"):
            hist, fut = get_prophet_region(reg)
    except Exception as _prophet_exc:
        st.error(f"⚠️ Prophet region forecast failed for '{reg}'.")
        st.exception(_prophet_exc)
        st.stop()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], mode="lines", name="Historical Actual",
        line=dict(color=meta["color"], width=2), hovertemplate="<b>%{x|%b %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fut["date"], y=fut["yhat"], mode="lines+markers", name="Q1 2019 Forecast",
        line=dict(color="#10b981", width=2.5, dash="dash"), marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=pd.concat([fut["date"], fut["date"][::-1]]).tolist(),
        y=pd.concat([fut["yhat_upper"], fut["yhat_lower"][::-1]]).tolist(),
        fill="toself", fillcolor="rgba(16,185,129,0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", hoverinfo="skip"))
    fig.add_vline(x="2018-12-31", line_dash="dot", line_color="#475569")
    fig.add_annotation(x="2018-12-31", y=0.95, yref="paper", text="Forecast →", showarrow=False, font=dict(color="#94a3b8"), xanchor="left")
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{reg} Region — Monthly Sales & Q1 2019 Forecast",
                      yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=420)
    col_c, col_d = st.columns([5, 1])
    with col_c:
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        download_plotly_html(fig, f"forecast_region_{reg.lower()}.html")
        fc_dl = pd.concat([hist.rename(columns={"sales": "actual_sales"}), fut[["date", "yhat", "yhat_lower", "yhat_upper"]]])
        download_csv(fc_dl, f"forecast_region_{reg.lower()}.csv", "⬇️ Download Forecast (CSV)")
    if meta["yoy"] < 0:
        warning_box(f"**{reg}** is contracting ({meta['yoy']:+.2f}% YoY). Reduce procurement budgets and rebalance allocation to South/East.")
    else:
        insight(f"**{reg}** shows {meta['yoy']:+.1f}% YoY growth. Consider increasing safety stock and regional distribution capacity to match the surge.")
    hr()

    section_header("Recommended Q1 2019 Inventory Allocation")
    alloc_df = pd.DataFrame([{"Region": r, "Allocation (%)": m["alloc"], "Q1 Forecast ($)": m["q1_2019"], "YoY Growth (%)": m["yoy"]}
                              for r, m in REGION_META.items()]).sort_values("Allocation (%)", ascending=False)
    fig_alloc = px.bar(alloc_df, x="Region", y="Allocation (%)", color="Region",
                       color_discrete_sequence=[COLORS[3], COLORS[0], COLORS[1], COLORS[2]],
                       text="Allocation (%)", title="Recommended Inventory Allocation by Region (%)")
    fig_alloc.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_alloc.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=360)
    st.plotly_chart(fig_alloc, use_container_width=True)
    hr()

    section_header("Complete Regional Growth & Allocation Matrix")
    table_df = pd.DataFrame([{"Region": r, "Q1 2018 Actual": m["q1_2018"], "Q4 2018 Actual": m["q4_2018"],
                               "Q1 2019 Forecast": m["q1_2019"], "YoY Growth (%)": m["yoy"],
                               "Allocation Weight (%)": m["alloc"], "YoY Rank": m["rank"]}
                              for r, m in REGION_META.items()]).sort_values("YoY Rank")
    st.dataframe(
        table_df,
        column_config={
            "Q1 2018 Actual": st.column_config.NumberColumn("Q1 2018 Actual", format="$%,.2f"),
            "Q4 2018 Actual": st.column_config.NumberColumn("Q4 2018 Actual", format="$%,.2f"),
            "Q1 2019 Forecast": st.column_config.NumberColumn("Q1 2019 Forecast", format="$%,.2f"),
            "YoY Growth (%)": st.column_config.NumberColumn("YoY Growth (%)", format="+%.2f%%"),
            "Allocation Weight (%)": st.column_config.NumberColumn("Allocation Weight (%)", format="%.2f%%"),
            "YoY Rank": st.column_config.NumberColumn("YoY Rank", format="%d"),
        },
        use_container_width=True,
        hide_index=True,
    )
    download_csv(table_df, "region_growth_allocation.csv", "⬇️ Download Allocation Matrix (CSV)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANOMALY CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🚨 Anomaly Center":
    page_title("🚨", "Anomaly Center", "Isolation Forest & Rolling Z-Score weekly anomaly detection")
    with st.spinner("Computing anomaly detection (cached after first run)…"):
        wk = get_anomaly_data()

    with st.sidebar:
        st.markdown("### 🔎 Filters")
        years_avail = sorted(wk["year"].unique())
        sel_years   = st.multiselect("Filter by year", years_avail, default=years_avail)
        method      = st.radio("Detection Method", ["Isolation Forest", "Z-Score", "Both (Union)"], index=0)

    wk_filt = wk[wk["year"].isin(sel_years)].copy()
    n_if    = wk_filt["iforest_flag"].sum()
    n_zs    = wk_filt["zscore_flag"].sum()
    n_com   = (wk_filt["iforest_flag"] & wk_filt["zscore_flag"]).sum()

    section_header("Detection Summary")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Weeks Analysed", str(len(wk_filt)), f"{sel_years[0]}–{sel_years[-1]}")
    with c2: metric_card("I-Forest Anomalies", str(n_if), "Context-aware model", positive=False)
    with c3: metric_card("Z-Score Anomalies", str(n_zs), "±2σ threshold model", positive=False)
    with c4: metric_card("Common Flags (Both)", str(n_com), "High-confidence anomalies", positive=False)
    hr()

    section_header("Weekly Sales Anomaly Timeline")
    flag_col = {"Isolation Forest": "iforest_flag", "Z-Score": "zscore_flag", "Both (Union)": None}[method]
    anom_df  = wk_filt[wk_filt["iforest_flag"] | wk_filt["zscore_flag"]] if method == "Both (Union)" else wk_filt[wk_filt[flag_col]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk_filt["date"], y=wk_filt["sales"], mode="lines", name="Weekly Sales",
        line=dict(color=COLORS[0], width=1.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>"))
    if method == "Z-Score":
        fig.add_trace(go.Scatter(x=wk_filt["date"], y=wk_filt["roll_mean"], mode="lines",
            name="8-Week Rolling Mean", line=dict(color=COLORS[1], width=1.4, dash="dot")))
        fig.add_trace(go.Scatter(
            x=pd.concat([wk_filt["date"], wk_filt["date"][::-1]]).tolist(),
            y=pd.concat([wk_filt["roll_mean"] + 2*wk_filt["roll_std"],
                         (wk_filt["roll_mean"] - 2*wk_filt["roll_std"])[::-1]]).tolist(),
            fill="toself", fillcolor="rgba(168,85,247,0.08)", line=dict(color="rgba(0,0,0,0)"),
            name="±2σ Band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anom_df["date"], y=anom_df["sales"], mode="markers",
        name=f"Anomalies ({method})", marker=dict(size=10, color="#ef4444", symbol="circle", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Sales: $%{y:,.0f}<br>Expected: $%{customdata[0]:,.0f}<br>Deviation: %{customdata[1]:+.1f}%<extra></extra>",
        customdata=anom_df[["expected", "dev_pct"]].values))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Weekly Sales Anomalies — {method}",
                      yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=430)
    col_c, col_d = st.columns([5, 1])
    with col_c:
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        download_plotly_html(fig, f"anomalies_{method.lower().replace(' ', '_')}.html")
    hr()

    if len(anom_df) > 0:
        section_header("Anomaly Deviation Severity")
        fig_dev = px.bar(anom_df.sort_values("dev_pct"),
                         x="dev_pct", y=anom_df.loc[anom_df.index, "date"].dt.strftime("%d %b %Y"),
                         orientation="h", color="dev_pct", color_continuous_scale="RdYlGn",
                         title="Anomaly Deviation from Expected Sales (%)",
                         labels={"dev_pct": "Deviation (%)", "y": "Week"})
        fig_dev.update_layout(**PLOTLY_LAYOUT, height=max(250, len(anom_df) * 40), showlegend=False)
        st.plotly_chart(fig_dev, use_container_width=True)
        hr()

    section_header("Anomaly Detail Table")
    detail = anom_df[["date", "sales", "expected", "dev_pct", "z_score", "iforest_flag", "zscore_flag"]].copy()
    detail["date_str"]  = detail["date"].dt.strftime("%Y-%m-%d")
    detail["direction"] = detail["dev_pct"].apply(lambda x: "⬆️ Surge" if x > 0 else "⬇️ Drop")
    display_detail = detail[["date_str", "sales", "expected", "dev_pct", "direction", "iforest_flag", "zscore_flag"]].rename(columns={
        "date_str": "Week Of", "sales": "Actual Sales ($)", "expected": "Expected ($)",
        "dev_pct": "Deviation (%)", "direction": "Direction", "iforest_flag": "I-Forest Flag", "zscore_flag": "Z-Score Flag"})
    st.dataframe(display_detail, use_container_width=True,
                 column_config={"Actual Sales ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "Expected ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "Deviation (%)": st.column_config.NumberColumn(format="%.1f%%")})
    download_csv(display_detail, "anomaly_table.csv", "⬇️ Download Anomaly Table (CSV)")
    hr()
    insight("Production Recommendation: Deploy Isolation Forest. It catches critical pre-holiday logistics drops (October) that the Z-Score model misses entirely.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DEMAND SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🎯 Demand Segmentation":
    page_title("🎯", "Demand Segmentation", "K-Means product cluster analysis with inventory recommendation cards")
    SEGMENT_COLORS = {"High Value Stable": COLORS[0], "Growing Demand": COLORS[3],
                      "Seasonal Products": COLORS[2], "Declining Products": COLORS[4] if len(COLORS) > 4 else "#fb923c"}
    INVENTORY_RULES = {
        "High Value Stable":  ("🟦", "Automated Reorder Point (ROP)", "10% safety stock buffer. Weekly automated replenishment. Negotiate long-term fixed-price contracts.", "success"),
        "Growing Demand":     ("🟩", "Growth Ramping", "Increase Q1 stock levels 20-30% YoY. Use Prophet forecast upper CI bounds for purchase orders.", "success"),
        "Seasonal Products":  ("🟨", "Dynamic Seasonal Stocking", "2× inventory in October. Slash by 60% in January. Prefer on-demand ordering for large capital items.", "warning"),
        "Declining Products": ("🟥", "Inventory Drawdown", "Reduce orders by 20% YoY. Run clearance promotions. Transition Tables/Bookcases to drop-shipping.", "warning"),
    }

    with st.spinner("Computing segmentation (cached after first run)…"):
        agg = get_segmentation_data()
    segments = sorted(agg["segment"].unique())

    section_header("Segmentation Summary")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Sub-Categories", "17", "5 engineered features · k=4 clusters")
    with c2: metric_card("High Value Stable", str(len(agg[agg["segment"] == "High Value Stable"])), "Core revenue engine (~48% of sales)")
    with c3: metric_card("Fastest Growing", "Appliances", "+54.56% YoY · Growing Demand cluster")
    with c4: metric_card("Declining Products", str(len(agg[agg["segment"] == "Declining Products"])), "Reduce procurement 20% YoY")
    hr()

    section_header("Product Cluster Map — PCA Projection (2D)")
    pca_var    = agg["pca_var"].iloc[0]
    fig_scatter = go.Figure()
    for seg in segments:
        seg_df = agg[agg["segment"] == seg]
        fig_scatter.add_trace(go.Scatter(
            x=seg_df["pca_x"], y=seg_df["pca_y"], mode="markers+text", name=seg,
            text=seg_df["sub-category"], textposition="top center",
            textfont=dict(size=9, color="#cbd5e1"),
            marker=dict(size=seg_df["total_sales"] / seg_df["total_sales"].max() * 30 + 12,
                        color=SEGMENT_COLORS.get(seg, COLORS[0]), line=dict(color="white", width=1.5), opacity=0.9),
            hovertemplate="<b>%{text}</b><br>Segment: " + seg + "<br>Total Sales: $%{customdata[0]:,.0f}<br>AOV: $%{customdata[1]:,.0f}<br>Growth: %{customdata[2]:+.1f}%<extra></extra>",
            customdata=seg_df[["total_sales", "aov", "growth_pct"]].values))
    fig_scatter.update_layout(**PLOTLY_LAYOUT, title=f"Product Clusters in PCA Space ({pca_var} variance explained)",
                               xaxis_title="Principal Component 1", yaxis_title="Principal Component 2", height=500)
    col_c, col_d = st.columns([5, 1])
    with col_c:
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_d:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        download_plotly_html(fig_scatter, "segmentation_pca_interactive.html")
    hr()

    section_header("Total Sales by Business Segment")
    seg_totals = agg.groupby("segment")["total_sales"].sum().reset_index().sort_values("total_sales", ascending=False)
    fig_bar = px.bar(seg_totals, x="segment", y="total_sales", color="segment",
                     color_discrete_map=SEGMENT_COLORS, text="total_sales", title="Cumulative Sales by Cluster")
    fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False, yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=360)
    st.plotly_chart(fig_bar, use_container_width=True)
    hr()

    section_header("Inventory Recommendation Cards")
    cols = st.columns(2, gap="medium")
    for i, (seg, (icon, rule_name, rule_detail, rule_type)) in enumerate(INVENTORY_RULES.items()):
        seg_subs = agg[agg["segment"] == seg]["sub-category"].tolist()
        with cols[i % 2]:
            color  = "#0d3321" if rule_type == "success" else "#3b1515"
            border = "#10b981" if rule_type == "success" else "#ef4444"
            st.markdown(
                f"""
                <div style="background:{color};border:1px solid {border};border-radius:10px;padding:1.1rem 1.3rem;margin-bottom:0.8rem;">
                    <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;margin-bottom:0.4rem;">{icon} {seg}</div>
                    <div style="font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:0.3rem;text-transform:uppercase;">{rule_name}</div>
                    <div style="font-size:0.88rem;color:#cbd5e1;margin-bottom:0.5rem;">{rule_detail}</div>
                    <div style="font-size:0.77rem;color:#64748b;">Sub-categories: {', '.join(seg_subs)}</div>
                </div>
                """, unsafe_allow_html=True)
    hr()

    section_header("Full Product Segmentation Table")
    search     = st.text_input("🔍 Search sub-category", placeholder="e.g. 'Chairs', 'Paper'…")
    seg_filter = st.multiselect("Filter by segment", segments, default=segments)
    display    = agg[["sub-category", "segment", "total_sales", "purchase_frequency", "aov", "sales_volatility", "growth_pct"]].copy()
    display.columns = ["Sub-Category", "Segment", "Total Sales ($)", "Order Frequency", "AOV ($)", "Volatility ($)", "YoY Growth (%)"]
    if search:
        display = display[display["Sub-Category"].str.contains(search, case=False)]
    if seg_filter:
        display = display[display["Segment"].isin(seg_filter)]
    st.dataframe(display.sort_values("Total Sales ($)", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=360,
                 column_config={"Total Sales ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "AOV ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "Volatility ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "YoY Growth (%)": st.column_config.NumberColumn(format="%.2f%%")})
    download_csv(display, "product_segmentation.csv", "⬇️ Download Segmentation Table (CSV)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📋 Executive Summary":
    page_title("📋", "Executive Summary", "Data-driven strategic recommendations — synthesised from all model outputs")

    section_header("Projected Annual Business Impact")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Revenue Upside", "$45–55K", "Capture regional & category demand growth", positive=True)
    with c2: metric_card("Warehouse Savings", "~12%", "Drop-ship Tables/Bookcases → free shelf space", positive=True)
    with c3: metric_card("Transshipment Savings", "~20%", "Correct regional allocation weights", positive=True)
    with c4: metric_card("Q4 Revenue Protected", "$10K+", "Pre-holiday safety stock buffer", positive=True)
    hr()

    RECS = [
        {"id": "1", "icon": "📦", "area": "Demand",      "title": "Fund Furniture Acceleration",
         "observation": "Furniture is the fastest-growing category at +54.82% YoY. Technology remains the largest absolute revenue contributor at $58.5k Q1 forecast.",
         "evidence": "Prophet Q1 2019 forecasts: Furniture $36,533 (+54.82%) · Technology $58,493 (+5.03%) · Office Supplies $47,078 (+9.55%).",
         "action": "Increase Q1 2019 Furniture procurement budget by 55% YoY. Hold Technology orders flat, aligned with the $58.5k point forecast.",
         "impact": "$12,900 in incremental Q1 Furniture revenue captured while keeping Technology inventory lean.",
         "priority": "🔴 High", "effort": "Medium", "positive": True},
        {"id": "2", "icon": "🗺️", "area": "Warehousing", "title": "Realign Regional Inventory Weights",
         "observation": "South (+159.92% YoY) and East (+95.25% YoY) are expanding explosively. Central (-29.27%) is contracting.",
         "evidence": "Regional Prophet models: South $35,459 · East $35,228 · West $50,438 · Central $28,318.",
         "action": "Set allocation: West 33.75% · South 23.73% · East 23.57% · Central 18.95%. Reduce Central safety stock by 30%.",
         "impact": "Prevents $15,000 in stockout losses while eliminating Central overstock waste.",
         "priority": "🔴 High", "effort": "Medium", "positive": True},
        {"id": "3", "icon": "🏭", "area": "Warehousing", "title": "Drop-Ship Bulky Declining Items",
         "observation": "Tables and Bookcases occupy massive warehouse footprints but show stagnant YoY growth in the Declining Products K-Means cluster.",
         "evidence": "K-Means: Tables ($202k sales, +0.22% YoY) and Bookcases ($113k, +14.27%) classified as Declining Products.",
         "action": "Transition Tables and Bookcases to a drop-shipping model. Run clearance promotions on Envelopes and Fasteners.",
         "impact": "Frees 15-20% of local warehouse capacity, saving $8,000 annually in storage overhead.",
         "priority": "🔴 High", "effort": "Low", "positive": False},
        {"id": "4", "icon": "🚨", "area": "Risk",        "title": "Pre-Holiday Safety Stock Buffer",
         "observation": "Shipping network experiences severe delays in October immediately before Q4 peak, causing dramatic weekly sales drops.",
         "evidence": "Isolation Forest anomalies: Oct 1 ($8,824 vs expected $19,165, -54%) and Oct 29 ($6,423 vs expected $20,572, -69%).",
         "action": "Establish 10% safety stock buffer for high-velocity items in late September. Upgrade key B2B accounts to First-Class shipping during peak weeks.",
         "impact": "Protects $10,000+ in Q4 revenue from pre-holiday shipping cancellations.",
         "priority": "🔴 High", "effort": "Low", "positive": False},
        {"id": "5", "icon": "🔁", "area": "Procurement", "title": "Automate Core Product Replenishment",
         "observation": "Chairs, Phones, Storage, and Binders are the core revenue engine — 48% of total sales, consistently high purchase frequency.",
         "evidence": "K-Means High Value Stable cluster: Phones $327k, Chairs $322k, Storage $219k, Binders $200k.",
         "action": "Implement Reorder Point (ROP) automated replenishment. Negotiate long-term fixed-price supply contracts for Phones and Binders.",
         "impact": "Eliminates human error, reduces unit procurement costs by 5%, ensures zero stockouts on core items.",
         "priority": "🟡 Medium", "effort": "High", "positive": True},
        {"id": "6", "icon": "📣", "area": "Marketing",   "title": "Align Campaigns With Seasonal & B2B Cycles",
         "observation": "Q4 delivers 2× baseline sales. Positive anomaly spikes appear consistently at end-of-quarter — driven by corporate budget flushing.",
         "evidence": "Prophet yearly component confirms Nov/Dec peak. IF positive anomalies: 2015-03-23 (+727%), 2015-09-14 (+109%), 2018-03-26 (+147%).",
         "action": "Launch Q4 digital ad campaigns in early October. Run B2B catalog promotions in mid-March and mid-September.",
         "impact": "Syncing spend with natural buying momentum captures $12,000 in incremental annual revenue.",
         "priority": "🟢 Normal", "effort": "Low", "positive": True},
        {"id": "7", "icon": "💰", "area": "Pricing",     "title": "Launch B2B Loyalty & Bundle Programme",
         "observation": "Corporate and Home Office segments represent 48.0% of total revenue and purchase high-AOV items.",
         "evidence": "EDA: Consumer 51%, Corporate 31%, Home Office 18% by revenue. Chairs AOV $531, Storage $264, Phones $374.",
         "action": "Launch volume-tiered B2B Loyalty Programme. Create Home Office Setup Bundles. Implement subscription reordering for Paper/Labels.",
         "impact": "Increases B2B AOV by 12% and boosts retention, driving $18,000 in additional annual B2B sales.",
         "priority": "🟢 Normal", "effort": "High", "positive": True},
    ]

    section_header("Strategic Recommendations")
    area_filter     = st.multiselect("Filter by area", sorted({r["area"] for r in RECS}), default=sorted({r["area"] for r in RECS}))
    priority_filter = st.multiselect("Filter by priority", ["🔴 High", "🟡 Medium", "🟢 Normal"], default=["🔴 High", "🟡 Medium", "🟢 Normal"])

    for rec in RECS:
        if rec["area"] not in area_filter or rec["priority"] not in priority_filter:
            continue
        with st.expander(f"{rec['icon']} **{rec['id']}.** {rec['title']} · _{rec['area']}_ · {rec['priority']}", expanded=False):
            col_obs, col_imp = st.columns(2, gap="medium")
            with col_obs:
                st.markdown("**🔍 Observation**"); st.markdown(rec["observation"])
                st.markdown("**📊 Evidence**");    st.markdown(rec["evidence"])
            with col_imp:
                st.markdown("**✅ Recommended Action**"); st.markdown(rec["action"])
                st.markdown("**💹 Expected Business Impact**")
                if rec["positive"]:
                    success_box(rec["impact"])
                else:
                    warning_box(rec["impact"])
    hr()

    section_header("Priority Action Matrix")
    matrix_df = pd.DataFrame([{"Priority": r["priority"], "Recommendation": r["title"], "Area": r["area"],
                                "Effort": r["effort"], "Annual Impact": r["impact"][:60] + "…"} for r in RECS])
    st.dataframe(matrix_df, use_container_width=True, hide_index=True,
                 column_config={"Priority": st.column_config.TextColumn(width="small"),
                                "Effort":   st.column_config.TextColumn(width="small")})
    download_csv(matrix_df, "executive_recommendations.csv", "⬇️ Download Recommendations (CSV)")
    hr()

    section_header("Financial Impact Waterfall")
    fig_wf = go.Figure(go.Waterfall(
        name="Annual Benefit", orientation="v",
        measure=["relative"]*7 + ["total"],
        x=["Demand\n(Furniture)", "Regional\nAllocation", "Warehousing\n(Drop-ship)",
           "Supply Chain\nBuffer", "Procurement\nAutomation", "Marketing\nTiming", "B2B\nLoyalty", "Total Impact"],
        y=[12900, 15000, 8000, 10000, 5000, 12000, 18000, 0],
        connector=dict(line=dict(color="#334155")),
        increasing=dict(marker=dict(color=COLORS[0])),
        decreasing=dict(marker=dict(color="#f87171")),
        totals=dict(marker=dict(color=COLORS[3])),
        text=["$12,900", "$15,000", "$8,000", "$10,000", "$5,000", "$12,000", "$18,000", ""],
        textposition="outside",
        hovertemplate="%{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ))
    fig_wf.update_layout(**PLOTLY_LAYOUT, title="Projected Annual Financial Benefit by Initiative ($)",
                         yaxis_tickprefix="$", yaxis_tickformat=",.0f", height=420)
    st.plotly_chart(fig_wf, use_container_width=True)
    success_box("Implementing all seven recommendations is projected to deliver $80,900 in annual net financial benefit — combining revenue capture, cost savings, and risk protection.")
