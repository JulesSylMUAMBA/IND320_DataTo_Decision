# utils.py
# --- Shared utilities for MongoDB-backed Streamlit dashboard (IND320) ---

import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

import plotly.graph_objects as go
import plotly.express as px

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.signal import spectrogram
from scipy.fft import dct, idct

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# Load environment variables (for local dev)
load_dotenv()

# Try to import Streamlit (for caching etc.). If not available, we fall back gracefully.
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False


# =============================================================================
# Mongo helpers
# =============================================================================
def get_mongo_client():
    """
    Return a MongoClient using Streamlit secrets if available,
    otherwise fall back to environment variables.
    """
    if HAS_STREAMLIT:
        uri = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI"))
    else:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri)


def get_db(db_name: str | None = None):
    """
    Return a reference to the MongoDB database.
    Default db name is 'ind320' (can be overridden by env or args).
    """
    if HAS_STREAMLIT:
        default_name = st.secrets.get("MONGO_DB", os.getenv("MONGO_DB", "ind320"))
    else:
        default_name = os.getenv("MONGO_DB", "ind320")

    db_name = db_name or default_name
    client = get_mongo_client()
    return client[db_name]


def get_collection(coll_name: str, db_name: str | None = None):
    """
    Return a specific collection in the target db.
    This is more flexible than a single default collection.
    """
    db = get_db(db_name)
    return db[coll_name]


def get_db_and_collection():
    """
    Kept for backward compatibility with older pages:
    Return (db, default_collection) based on secrets/env.
    Default collection name: 'production_mba_hour'.
    """
    if HAS_STREAMLIT:
        db_name = st.secrets.get("MONGO_DB", os.getenv("MONGO_DB", "ind320"))
        coll_name = st.secrets.get("MONGO_COLL", os.getenv("MONGO_COLL", "production_mba_hour"))
    else:
        db_name = os.getenv("MONGO_DB", "ind320")
        coll_name = os.getenv("MONGO_COLL", "production_mba_hour")

    db = get_db(db_name)
    return db, db[coll_name]


def fetch_price_areas(coll_name: str = "production_mba_hour") -> list[str]:
    """Return all unique price areas from a given collection."""
    coll = get_collection(coll_name)
    areas = coll.distinct("price_area")
    return sorted([a for a in areas if a])


def fetch_groups(coll_name: str = "production_mba_hour", group_field: str = "production_group") -> list[str]:
    """Return all unique production/consumption groups from a given collection."""
    coll = get_collection(coll_name)
    groups = coll.distinct(group_field)
    return sorted([g for g in groups if g and g != "*"])


def fetch_pie_data(price_area: str, coll_name: str = "production_mba_hour") -> pd.DataFrame:
    """
    Aggregate total kWh by production group for one price area.
    Used for exploratory pie charts.
    """
    coll = get_collection(coll_name)
    pipeline = [
        {"$match": {"price_area": price_area, "production_group": {"$ne": "*"}}},
        {"$group": {"_id": "$production_group", "quantity_kwh": {"$sum": "$quantity_kwh"}}},
        {"$project": {"production_group": "$_id", "quantity_kwh": 1, "_id": 0}},
        {"$sort": {"quantity_kwh": -1}},
    ]
    docs = list(coll.aggregate(pipeline))
    return pd.DataFrame(docs)


def fetch_line_data(
    price_area: str,
    groups: list[str],
    year: int,
    month: int,
    coll_name: str = "production_mba_hour",
    group_field: str = "production_group",
) -> pd.DataFrame:
    """
    Fetch hourly docs for the selected month and return pivoted DataFrame.
    Used for time series line charts (per group).
    """
    from datetime import timezone
    import calendar

    coll = get_collection(coll_name)

    start_utc = datetime(year, month, 1, tzinfo=timezone.utc)
    end_utc = datetime(
        year, month, calendar.monthrange(year, month)[1], 23, 59, 59, tzinfo=timezone.utc
    )

    match = {
        "price_area": price_area,
        group_field: {"$in": groups},
        "start_time": {"$gte": start_utc, "$lte": end_utc},
    }

    cursor = coll.find(match, {"_id": 0, group_field: 1, "start_time": 1, "quantity_kwh": 1})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    pivot = df.pivot_table(
        index="start_time",
        columns=group_field,
        values="quantity_kwh",
        aggfunc="sum",
    ).sort_index()

    return pivot


def fetch_energy_timeseries(
    coll_name: str,
    price_area: str,
    group: str,
    start_time_utc: datetime,
    end_time_utc: datetime,
    group_field: str = "production_group",
) -> pd.DataFrame:
    """
    Generic function to fetch hourly energy data (production or consumption)
    from MongoDB for one price area and one group in a time range.
    Returns a DataFrame indexed by 'start_time'.
    """
    coll = get_collection(coll_name)

    query = {
        "price_area": price_area,
        group_field: group,
        "start_time": {"$gte": start_time_utc, "$lt": end_time_utc},
    }

    docs = list(coll.find(query, {"_id": 0}))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df = df.set_index("start_time").sort_index()
    return df


def fetch_mean_by_price_area(
    coll_name: str,
    group_field: str,
    group: str,
    start_time_utc: datetime,
    end_time_utc: datetime,
) -> pd.DataFrame:
    """
    Compute mean quantity_kwh per price_area over a given time range and group.
    Useful for choropleth coloring on the map.
    """
    coll = get_collection(coll_name)
    pipeline = [
        {
            "$match": {
                "start_time": {"$gte": start_time_utc, "$lt": end_time_utc},
                group_field: group,
                "price_area": {"$ne": None},
            }
        },
        {
            "$group": {
                "_id": "$price_area",
                "mean_quantity_kwh": {"$avg": "$quantity_kwh"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "price_area": "$_id",
                "mean_quantity_kwh": 1,
            }
        },
        {"$sort": {"price_area": 1}},
    ]
    docs = list(coll.aggregate(pipeline))
    return pd.DataFrame(docs)


# =============================================================================
# Open-Meteo ERA5 (archive API)
# =============================================================================
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


if HAS_STREAMLIT:
    @st.cache_data(ttl=3600)
    def fetch_openmeteo(
        area: str,
        year: int,
        hourly=(
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ),
    ) -> pd.DataFrame:
        """
        Fetch ERA5 data for a full calendar year for a given price area.
        """
        lat, lon, _ = get_latlon(area)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
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
else:
    def fetch_openmeteo(
        area: str,
        year: int,
        hourly=(
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ),
    ) -> pd.DataFrame:
        lat, lon, _ = get_latlon(area)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
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


def fetch_era5_range_latlon(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly=(
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
    ),
) -> pd.DataFrame:
    """
    Fetch ERA5 data for an arbitrary date range and coordinates.
    Used for snow drift and wind analyses tied to map clicks.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    }
    r = requests.get(OPENMETEO_ERA5, params=params, timeout=30)
    r.raise_for_status()
    h = r.json().get("hourly", {})
    if "time" not in h or not h["time"]:
        return pd.DataFrame()
    df = pd.DataFrame({"time": pd.to_datetime(h["time"], utc=True)})
    for v in hourly:
        df[v] = pd.to_numeric(h.get(v, []), errors="coerce")
    return df.set_index("time").sort_index()


# =============================================================================
# STL & Spectrogram for Elhub
# =============================================================================
def plot_stl_plotly(
    df: pd.DataFrame,
    price_area: str = "NO1",
    production_group: str = "hydro",
    period: int = 24 * 7,      # weekly seasonality on hourly data
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
    fig.add_trace(go.Scatter(x=t, y=res.trend, mode="lines", name="Trend"))
    fig.add_trace(go.Scatter(x=t, y=res.seasonal, mode="lines", name="Seasonal"))
    fig.add_trace(go.Scatter(x=t, y=res.resid, mode="lines", name="Residual", opacity=0.5))
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
    window_len: int = 24 * 14,   # 2 weeks
    overlap: float = 0.5,        # 50%
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
        x=t_stamps, y=f_per_day, z=10 * np.log10(Sxx + 1e-12),
        coloraxis="coloraxis", zsmooth=False
    ))
    fig.update_layout(
        title=f"Spectrogram — Area {price_area} · Group {production_group} "
              f"(window={nperseg}h, overlap={int(overlap * 100)}%)",
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


# =============================================================================
# Weather QC: Temperature SPC (DCT) & Precipitation LOF
# =============================================================================
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
        xaxis_title="Time (UTC)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        hovermode="x unified",
        height=480,
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
        xaxis_title="Time (UTC)",
        yaxis_title="Precipitation (mm)",
        template="plotly_white",
        hovermode="x unified",
        height=480,
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


# =============================================================================
# Part 4: Snow drift & Wind Rose
# =============================================================================
def calculate_snow_drift_wind_rose(lat: float, lon: float, year: int):
    """
    Calculates Snow Drift (simplified accumulation/melt model) and
    returns (snow_drift_series, wind_data) for a "snow year":
        1 July of `year` -> 30 June of `year + 1`.
    """
    start_date = f"{year}-07-01"
    end_date = f"{year + 1}-06-30"

    df_weather = fetch_era5_range_latlon(
        lat,
        lon,
        start_date=start_date,
        end_date=end_date,
        hourly=("temperature_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"),
    )
    if df_weather.empty:
        return None, None

    df = df_weather.copy()
    df = df.sort_index()

    # Simple snow drift model in mm water equivalent
    MELT_RATE = 0.1  # mm/hour above 0°C

    accumulation = []
    current_snow = 0.0
    for _, row in df.iterrows():
        # Add precip (assume all is snow / water equivalent)
        precip = row.get("precipitation", 0.0) or 0.0
        temp = row.get("temperature_2m", 0.0) or 0.0

        current_snow += precip

        # Melt if above freezing
        if temp > 0:
            melt = min(current_snow, MELT_RATE)
            current_snow -= melt

        accumulation.append(current_snow)

    df["snow_drift_mm"] = accumulation

    wind_data = df[["wind_speed_10m", "wind_direction_10m"]].copy()
    snow_series = df[["snow_drift_mm"]]

    return snow_series, wind_data


def calculate_snow_drift_range(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Compute a simple yearly snow drift indicator for each snow year in [start_year, end_year].
    Returns a DataFrame with columns:
        - snow_year (e.g., "2020-2021")
        - max_snow_drift_mm
        - mean_snow_drift_mm
    """
    results = []
    for yr in range(start_year, end_year + 1):
        snow, wind = calculate_snow_drift_wind_rose(lat, lon, yr)
        if snow is None or snow.empty:
            continue

        max_drift = snow["snow_drift_mm"].max()
        mean_drift = snow["snow_drift_mm"].mean()

        results.append(
            {
                "snow_year": f"{yr}-{yr + 1}",
                "max_snow_drift_mm": max_drift,
                "mean_snow_drift_mm": mean_drift,
            }
        )

    if not results:
        return pd.DataFrame()

    df_yearly = pd.DataFrame(results)
    return df_yearly


def plot_wind_rose(wind_data: pd.DataFrame, title: str = "Wind Rose"):
    """
    Build a simple wind rose from wind speed and direction.
    Returns a Plotly Figure.
    """
    if wind_data is None or wind_data.empty:
        raise ValueError("No wind data to plot.")

    df_wind = wind_data.dropna(subset=["wind_speed_10m", "wind_direction_10m"]).copy()
    if df_wind.empty:
        raise ValueError("No valid (speed, direction) rows after cleaning.")

    # Normalize directions to [0, 360)
    df_wind["dir_norm"] = df_wind["wind_direction_10m"] % 360

    # Bin directions into 16 sectors of 22.5°
    bin_width = 22.5
    bins = np.arange(0, 360 + bin_width, bin_width)
    labels = [f"{int(b)}–{int(b + bin_width)}°" for b in bins[:-1]]
    df_wind["dir_sector"] = pd.cut(
        df_wind["dir_norm"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    sector_stats = (
        df_wind.groupby("dir_sector")["wind_speed_10m"]
        .mean()
        .reset_index()
        .rename(columns={"wind_speed_10m": "mean_speed"})
    )

    fig = px.bar_polar(
        sector_stats,
        r="mean_speed",
        theta="dir_sector",
        title=title,
        labels={"mean_speed": "Mean wind speed (m/s)", "dir_sector": "Direction sector"},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=60, b=10))
    return fig


# =============================================================================
# Part 4: Sliding Window Correlation
# =============================================================================
def calculate_sliding_correlation(
    df: pd.DataFrame,
    window_size: int,
    var_a: str,
    var_b: str,
    lag: int = 0,
) -> pd.Series:
    """
    Calculate rolling correlation between var_a and var_b
    (optionally lagging var_b).
    Returns a pandas Series indexed by the DataFrame index.
    """
    if var_a not in df.columns or var_b not in df.columns:
        raise ValueError(f"Missing columns {var_a} or {var_b} in DataFrame.")

    temp = df.copy()
    if lag != 0:
        temp[f"{var_b}_lagged"] = temp[var_b].shift(lag)
        var_b_name = f"{var_b}_lagged"
    else:
        var_b_name = var_b

    corr = temp[var_a].rolling(window=window_size).corr(temp[var_b_name])
    return corr


# =============================================================================
# Part 4: SARIMAX Forecasting
# =============================================================================
def run_sarimax_forecast(
    endog_series: pd.Series,
    exog_df: pd.DataFrame | None = None,
    periods: int = 24,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> pd.DataFrame:
    """
    Fit a SARIMAX model and forecast for 'periods' steps ahead.
    Returns a DataFrame with forecast mean and confidence intervals.
    """
    # Ensure DatetimeIndex
    if not isinstance(endog_series.index, pd.DatetimeIndex):
        raise TypeError("Endogenous series index must be a DatetimeIndex.")

    # Fit SARIMAX model
    model = SARIMAX(
        endog=endog_series,
        exog=exog_df,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    # Build forecast
    forecast = results.get_forecast(steps=periods, exog=exog_df.tail(periods) if exog_df is not None else None)
    forecast_df = forecast.summary_frame(alpha=0.05)

    # If index has no freq, create one for plotting convenience
    if endog_series.index.freq is None and endog_series.index.inferred_freq is None:
        # Assume hourly series
        last = endog_series.index[-1]
        forecast_index = pd.date_range(start=last, periods=periods + 1, freq="h")[1:]
        forecast_df.index = forecast_index

    return forecast_df


# =============================================================================
# Part 4: GeoJSON Loader (Elspot Price Areas)
# =============================================================================
def load_elspot_geojson(filepath: str | None = None):
    """
    Load GeoJSON file for Norwegian Elspot price areas.
    Default path: ../data/file.geojson (relative to this utils.py file).
    """
    if filepath is None:
        # utils.py is in streamlit_app/, data is in ../data/file.geojson
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "..", "data", "file.geojson")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"GeoJSON file not found at {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading GeoJSON: {e}")
