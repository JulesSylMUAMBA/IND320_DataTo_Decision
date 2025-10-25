# utils.py
# --- Utility helpers for MongoDB-backed Streamlit pages ---
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env if present (useful for local testing)
load_dotenv()

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

def fetch_price_areas():
    """Return all unique price areas."""
    db, coll = get_db_and_collection()
    areas = coll.distinct("price_area")
    return sorted([a for a in areas if a])

def fetch_groups():
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
