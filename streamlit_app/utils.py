# utils.py
# --- Utility helpers for MongoDB-backed Streamlit pages ---

import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Numeric / plotting / analysis
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram

# For weather analyses
from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# HTTP
import requests

# Load .env if present (useful for local testing)
load_dotenv()


# -----------------------------------------------------------------------------
# Mongo helpers
# -----------------------------------------------------------------------------
def get_mongo_client():
    """
    Return a MongoClient using Streamlit secrets if available,
    otherwise fall back to environment variables.
    """
    try:
        import streamlit as st
        uri = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI"))
    except Exception:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri)


def get_db_and_collection():
    """
    Return a tuple (db, coll) based on secrets/env variables.
    """
    try:
        import streamlit as st
        db_name = st.secrets.get("MONGO_DB", os.getenv("MONGO_DB", "ind320"))
        coll_name = st.secrets.get("MONGO_COLL", os.getenv("MONGO_COLL", "production_mba_hour"))
    except Exception:
        db_name = os.getenv("MONGO_DB", "ind320")
        coll_name = os.getenv("MONGO_COLL", "production_mba_hour")

    client = get_mongo_client()
    return client[db_name], client[db_name][coll_name]


def fetch_price_areas() -> list[str]:
    """Return all unique price areas."""
    db, coll = get_db_and_collection()
    areas = coll.distinct("price_area")
    return sorted([a for a in areas if a])


def fetch_groups() -> list[str]:
    """Return all unique production groups."""
    db, coll = get_db_and_collection()
    groups = coll.distinct("production_group")
    return sorted([g for g in groups if g])


def fetch_pie_data(price_area: str) -> pd.DataFrame:
    """Aggregate total kWh by production group for one price area."""
    db, coll = get_db_and_collection()
    pipeline = [
        {"$match": {"price_area": price_area, "production_group": {"$ne": "*"}}},
        {"$group": {"_id": "$production_group", "quantity_kwh": {"$sum": "$quantity_kwh"}}},
        {"$project": {"production_group": "$_id", "quantity_kwh": 1, "_id": 0}},
        {"$sort": {"quantity_kwh": -1}},
    ]
    docs = list(coll.aggregate(pipeline))
    return pd.DataFrame(docs)


def fetch_line_data(price_area: str, groups: list[str], year: int, month: int) -> pd.DataFrame:
    """Fetch hourly docs for the selected month and return pivoted DataFrame."""
    db, coll = get_db_and_collection()
    from datetime import datetime, timezone
    import calendar

    start_utc = datetime(year, month, 1, tzinfo=timezone.utc)
    end_utc = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59, tzinfo=timezone.utc)

    match = {
        "price_area": price_area,
        "production_group": {"$in": groups},
        "start_time": {"$gte": start_utc, "$lte": end_utc},
    }

    cursor = coll.find(match, {"_id": 0, "production_group": 1, "start_time": 1, "quantity_kwh": 1})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    pivot = df.pivot_table(
        index="start_time",
        columns="production_group",
        values="quantity_kwh",
        aggfunc="sum"
    ).sort_index()

    return pivot


# -----------------------------------------------------------------------------
# Open-Meteo ERA5 (archive API)
# -----------------------------------------------------------------------------
CITY_MAP = {
    "NO1": {"city": "Oslo",         "lat": 59.9139,  "lon": 10.7522},
    "NO2": {"city": "Kristiansand", "lat": 58.1467,  "lon": 7.9956},
    "NO3": {"city": "Trondheim",    "lat": 63.4305,  "lon": 10.3951},
    "NO4": {"city": "Tromsø",       "lat": 69.6492,  "lon": 18.9553},
    "NO5": {"city": "Bergen",       "lat": 60.39299, "lon": 5.32415},
}

def get_latlon(area: str):
    meta = CITY_MAP.get(area)
    if not meta:
        raise ValueError(f"Unknown price area: {area}")
    return meta["lat"], meta["lon"], meta["city"]

OPENMETEO_ERA5 = "https://archive-api.open-meteo.com/v1/era5"

# Cache côté Streamlit (1h). Si utils.py est aussi utilisé hors Streamlit, fallback.
try:
    import streamlit as st

    @st.cache_data(ttl=3600)
    def fetch_openmeteo(area: str, year: int,
                        hourly=("temperature_2m","precipitation",
                                "wind_speed_10m","wind_gusts_10m","wind_direction_10m")) -> pd.DataFrame:
        lat, lon, _ = get_latlon(area)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date":   f"{year}-12-31",
            "hourly": ",".join(hourly),
            "timezone": "UTC",
        }
        r = requests.get(OPENMETEO_ERA5, params=params, timeout=30)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        df = pd.DataFrame({"time": pd.to_datetime(h.get("time", []), utc=True)})
        for v in hourly:
            df[v] = pd.to_numeric(h.get(v, []), errors="coerce")
        return df

except Exception:
    def fetch_openmeteo(area: str, year: int,
                        hourly=("temperature_2m","precipitation",
                                "wind_speed_10m","wind_gusts_10m","wind_direction_10m")) -> pd.DataFrame:
        lat, lon, _ = get_latlon(area)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date":   f"{year}-12-31",
            "hourly": ",".join(hourly),
            "timezone": "UTC",
        }
        r = requests.get(OPENMETEO_ERA5, params=params, timeout=30)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        df = pd.DataFrame({"time": pd.to_datetime(h.get("time", []), utc=True)})
        for v in hourly:
            df[v] = pd.to_numeric(h.get(v, []), errors="coerce")
        return df


# -----------------------------------------------------------------------------
# Advanced analyses for Elhub: STL & Spectrogram
# -----------------------------------------------------------------------------
def plot_stl_plotly(
    df: pd.DataFrame,
    price_area: str = "NO1",
    production_group: str = "hydro",
    period: int = 24*7,        # weekly seasonality on hourly data
    seasonal: int = 13,        # odd >= 7
    trend: int = 201,          # odd and > period
    robust: bool = True,
    time_col: str = "start_time",
    value_col: str = "quantity_kwh",
    area_col: str = "price_area",
    group_col: str = "production_group",
):
    """
    STL decomposition (LOESS) on Elhub production data.
    Returns (fig, res, info).
    """
    # 1) Filter selection
    d = df[(df[area_col] == price_area) & (df[group_col] == production_group)].copy()
    if d.empty:
        raise ValueError(f"No rows for area={price_area}, group={production_group}")

    # 2) Time indexing & hourly regularization
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    d = d[[value_col]].astype(float).groupby(level=0).sum()

    full_idx = pd.date_range(d.index.min(), d.index.max(), freq="h", tz=d.index.tz)
    d = d.reindex(full_idx)

    d[value_col] = d[value_col].interpolate(method="time", limit_direction="both").fillna(0.0)
    y = d[value_col].to_numpy()

    # enforce STL constraints: odd and > period
    def _make_odd(n, min_val):
        n = int(max(n, min_val))
        return n if n % 2 == 1 else n + 1

    period = int(period)
    seasonal = _make_odd(seasonal, 7)
    trend = _make_odd(trend, max(3, period + 1))
    if trend >= len(y):
        trend = _make_odd(len(y) - 1, max(3, period + 1))

    # 3) STL
    res = STL(y, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()

    # 4) Plot
    t = d.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=res.observed, mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(x=t, y=res.trend,    mode="lines", name="Trend"))
    fig.add_trace(go.Scatter(x=t, y=res.seasonal, mode="lines", name="Seasonal"))
    fig.add_trace(go.Scatter(x=t, y=res.resid,    mode="lines", name="Residual", opacity=0.5))
    fig.update_layout(
        title=f"STL — Area {price_area} · Group {production_group} "
              f"(period={period}, seasonal={seasonal}, trend={trend}, robust={robust})",
        xaxis_title="Time (UTC)",
        yaxis_title=value_col,
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )

    info = {
        "price_area": price_area,
        "production_group": production_group,
        "n_samples": int(len(y)),
        "params": {"period": period, "seasonal": seasonal, "trend": trend, "robust": robust},
        "time_range": (str(t.min()), str(t.max())),
        "value_col": value_col,
    }
    return fig, res, info


def plot_spectrogram_elhub(
    df: pd.DataFrame,
    price_area: str = "NO1",
    production_group: str = "hydro",
    time_col: str = "start_time",
    value_col: str = "quantity_kwh",
    area_col: str = "price_area",
    group_col: str = "production_group",
    window_len: int = 24*14,   # 2 weeks
    overlap: float = 0.5,      # 50%
    detrend: str = "constant",
    scaling: str = "density",
):
    """
    Spectrogram (time–frequency) on Elhub production series.
    Returns (fig, info).
    """
    # 1) Filter and regularize hourly series
    d = df[(df[area_col] == price_area) & (df[group_col] == production_group)].copy()
    if d.empty:
        raise ValueError(f"No rows for area={price_area}, group={production_group}")

    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    d = d[[value_col]].astype(float).groupby(level=0).sum()

    full_idx = pd.date_range(d.index.min(), d.index.max(), freq="h", tz=d.index.tz)
    d = d.reindex(full_idx)
    d[value_col] = d[value_col].interpolate(method="time", limit_direction="both").fillna(0.0)

    y = d[value_col].to_numpy()
    n = len(y)
    if n < max(window_len, 64):
        raise ValueError(f"Series too short ({n} pts) for window_len={window_len}")

    # 2) Spectrogram (fs = 1 sample/hour)
    fs = 1.0  # cycles per hour
    nperseg = int(window_len)
    noverlap = min(int(np.clip(overlap, 0.0, 0.95) * nperseg), nperseg - 1)

    f, t_bins, Sxx = spectrogram(
        y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        detrend=detrend, scaling=scaling, mode="psd"
    )
    # Convert to cycles/day for readability
    f_per_day = f * 24.0
    t0 = d.index[0]
    t_stamps = t0 + pd.to_timedelta(t_bins, unit="h")  # tz-aware timeline

    # Heatmap (dB)
    fig = go.Figure(go.Heatmap(
        x=t_stamps, y=f_per_day, z=10*np.log10(Sxx + 1e-12),
        coloraxis="coloraxis", zsmooth=False
    ))
    fig.update_layout(
        title=f"Spectrogram — Area {price_area} · Group {production_group} "
              f"(window={nperseg}h, overlap={int(overlap*100)}%)",
        xaxis_title="Time",
        yaxis_title="Frequency (cycles/day)",
        coloraxis=dict(colorbar=dict(title="Power (dB)")),
        template="plotly_white",
        height=650,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )

    # Summary
    peak_idx = np.argmax(Sxx, axis=0)
    peak_freq_per_day = f_per_day[peak_idx]
    info = {
        "n_samples": n,
        "time_range": (str(d.index.min()), str(d.index.max())),
        "window_len_hours": nperseg,
        "overlap": overlap,
        "dominant_freq_per_day": {
            "median": float(np.median(peak_freq_per_day)),
            "p10": float(np.percentile(peak_freq_per_day, 10)),
            "p90": float(np.percentile(peak_freq_per_day, 90)),
        },
        "notes": "≈1 c/d = daily cycle; ≈0.14 c/d = weekly cycle.",
    }
    return fig, info


# -----------------------------------------------------------------------------
# Weather QC: Temperature SPC (DCT) & Precipitation LOF
# -----------------------------------------------------------------------------
def plot_temp_spc_plotly(
    df: pd.DataFrame,
    time_col: str = "time",
    temp_col: str = "temperature_2m",
    cutoff: float = 0.05,
    n_sigma: float = 3.0,
    robust: bool = True,
):
    """
    DCT high-pass to remove seasonality -> SPC limits on SATV -> outliers on the temperature curve.
    Returns (fig, summary).
    """
    s = pd.to_numeric(df[temp_col], errors="coerce").to_numpy()
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    mask = (~np.isnan(s)) & (~t.isna())
    s, t = s[mask], t[mask]
    n = len(s)
    if n == 0:
        raise ValueError("Empty series for SPC.")

    # Fill small gaps
    s = pd.Series(s).interpolate(limit_direction="both").to_numpy()

    # DCT -> high-pass -> inverse DCT = SATV
    x = dct(s, norm="ortho")
    k = max(1, int(np.floor(cutoff * n)))
    x_hp = x.copy()
    x_hp[:k] = 0.0
    satv = idct(x_hp, norm="ortho")
    seasonal = s - satv

    if robust:
        center = float(np.median(satv))
        mad = float(np.median(np.abs(satv - center)))
        sigma = (1.4826 * mad) if mad > 0 else float(np.std(satv, ddof=1))
    else:
        center = float(np.mean(satv))
        sigma = float(np.std(satv, ddof=1))

    upper_satv = center + n_sigma * sigma
    lower_satv = center - n_sigma * sigma

    upper_curve = seasonal + upper_satv
    lower_curve = seasonal + lower_satv
    is_out = (satv > upper_satv) | (satv < lower_satv)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s, mode="lines", name="Temperature (°C)"))
    fig.add_trace(go.Scatter(x=t, y=upper_curve, mode="lines", name=f"Upper ({n_sigma}σ)", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=lower_curve, mode="lines", name=f"Lower ({n_sigma}σ)", line=dict(dash="dash")))
    if is_out.any():
        fig.add_trace(go.Scatter(x=t[is_out], y=s[is_out], mode="markers", name="Outliers", marker=dict(size=6)))
    fig.update_layout(
        title="Temperature with SPC limits (DCT-based seasonal adjustment)",
        xaxis_title="Time (UTC)", yaxis_title="Temperature (°C)",
        template="plotly_white", hovermode="x unified", height=480,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    summary = {
        "n_samples": int(n),
        "n_outliers": int(is_out.sum()),
        "outlier_pct": round(100.0 * is_out.mean(), 3),
        "params": {"cutoff": cutoff, "n_sigma": n_sigma, "robust": robust},
        "satv_stats": {"center": center, "sigma": sigma, "upper_satv": upper_satv, "lower_satv": lower_satv},
    }
    return fig, summary


def plot_precip_lof_plotly(
    df: pd.DataFrame,
    time_col: str = "time",
    precip_col: str = "precipitation",
    contamination: float = 0.01,
    n_neighbors: int = 40,
):
    """
    LOF anomaly detection on precipitation (handles many zeros via robust scaling).
    Returns (fig, summary, anomalies_df_sorted).
    """
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    p = pd.to_numeric(df[precip_col], errors="coerce").fillna(0.0)
    mask = ~t.isna()
    t, p = t[mask], p[mask]
    n = len(p)
    if n == 0:
        raise ValueError("Empty series for LOF.")

    # Feature: value + short moving average
    roll = p.rolling(6, min_periods=1, center=True).mean()
    X = np.c_[p.values, roll.values]

    X_scaled = RobustScaler().fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=int(n_neighbors), contamination=float(contamination))
    y_pred = lof.fit_predict(X_scaled)   # -1 outlier, 1 inlier
    scores = -lof.negative_outlier_factor_

    is_out = (y_pred == -1)
    anomalies = pd.DataFrame({"time": t.values, "precipitation": p.values, "score": scores})
    anomalies = anomalies[is_out].sort_values("score", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=p, mode="lines", name="Precipitation (mm)"))
    if is_out.any():
        fig.add_trace(go.Scatter(
            x=anomalies["time"], y=anomalies["precipitation"],
            mode="markers", name="LOF anomalies", marker=dict(size=6)
        ))
    fig.update_layout(
        title=f"Precipitation anomalies (LOF, contamination={contamination})",
        xaxis_title="Time (UTC)", yaxis_title="Precipitation (mm)",
        template="plotly_white", hovermode="x unified", height=480,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    summary = {
        "n_samples": int(n),
        "n_anomalies": int(is_out.sum()),
        "anomaly_pct": round(100.0 * is_out.mean(), 3),
        "params": {"contamination": float(contamination), "n_neighbors": int(n_neighbors)},
        "top_anomaly_score": float(anomalies["score"].max()) if not anomalies.empty else None,
    }
    return fig, summary, anomalies
