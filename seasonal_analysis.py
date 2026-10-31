"""
EcoSonic AI - Seasonal Analysis Module
Analyzes and compares wildlife sound activity across seasons.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_seasonal_data, get_all_recordings


def get_season_from_month(month: int) -> str:
    if month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Rainy"
    else:
        return "Winter"


def aggregate_seasonal_stats(location: Optional[str] = None) -> pd.DataFrame:
    recordings = get_all_recordings()
    if not recordings:
        return pd.DataFrame()
    df = pd.DataFrame(recordings)
    if location and location != "All":
        df = df[df["location"] == location]
    if df.empty:
        return pd.DataFrame()
    seasonal = df.groupby("season").agg(
        total_recordings=("filename", "count"),
        avg_confidence=("confidence_score", lambda x: round(x.mean(), 2) if x.notna().any() else 0),
    ).reset_index()
    for cat in ["Bird Calls", "Frog Calls", "Insect Sounds", "Unknown/Other"]:
        cat_counts = df[df["predicted_category"] == cat].groupby("season")["filename"].count()
        seasonal[cat] = seasonal["season"].map(cat_counts).fillna(0).astype(int)
    return seasonal


def get_monthly_activity(location: Optional[str] = None, year: Optional[int] = None) -> pd.DataFrame:
    recordings = get_all_recordings()
    if not recordings:
        return pd.DataFrame()
    df = pd.DataFrame(recordings)
    if location and location != "All":
        df = df[df["location"] == location]
    if "recording_date" not in df.columns or df["recording_date"].isna().all():
        return pd.DataFrame()
    df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")
    df = df.dropna(subset=["recording_date"])
    if year:
        df = df[df["recording_date"].dt.year == year]
    df["month"] = df["recording_date"].dt.month
    df["month_name"] = df["recording_date"].dt.strftime("%b")
    monthly = df.groupby(["month", "month_name"]).agg(
        total=("filename", "count"),
        birds=("predicted_category", lambda x: (x == "Bird Calls").sum()),
        frogs=("predicted_category", lambda x: (x == "Frog Calls").sum()),
        insects=("predicted_category", lambda x: (x == "Insect Sounds").sum()),
    ).reset_index().sort_values("month")
    return monthly


def get_location_comparison() -> pd.DataFrame:
    recordings = get_all_recordings()
    if not recordings:
        return pd.DataFrame()
    df = pd.DataFrame(recordings)
    location_stats = df.groupby("location").agg(
        total_recordings=("filename", "count"),
        birds=("predicted_category", lambda x: (x == "Bird Calls").sum()),
        frogs=("predicted_category", lambda x: (x == "Frog Calls").sum()),
        insects=("predicted_category", lambda x: (x == "Insect Sounds").sum()),
        avg_confidence=("confidence_score", lambda x: round(x.mean(), 2) if x.notna().any() else 0),
    ).reset_index()
    return location_stats


def get_category_distribution(season: Optional[str] = None, location: Optional[str] = None) -> Dict:
    recordings = get_all_recordings()
    if not recordings:
        return {}
    df = pd.DataFrame(recordings)
    if season and season != "All":
        df = df[df["season"] == season]
    if location and location != "All":
        df = df[df["location"] == location]
    if df.empty or "predicted_category" not in df.columns:
        return {}
    return df["predicted_category"].value_counts().to_dict()


def compare_seasons(season1: str, season2: str, location: Optional[str] = None) -> Dict:
    recordings = get_all_recordings()
    if not recordings:
        return {"error": "No recordings available"}
    df = pd.DataFrame(recordings)
    if location and location != "All":
        df = df[df["location"] == location]
    s1_data = df[df["season"] == season1]
    s2_data = df[df["season"] == season2]
    comparison = {
        "season1": season1, "season2": season2,
        "season1_count": len(s1_data), "season2_count": len(s2_data),
        "categories": {},
    }
    for cat in ["Bird Calls", "Frog Calls", "Insect Sounds", "Unknown/Other"]:
        s1_count = len(s1_data[s1_data["predicted_category"] == cat])
        s2_count = len(s2_data[s2_data["predicted_category"] == cat])
        change_pct = round(((s2_count - s1_count) / s1_count) * 100, 1) if s1_count > 0 else (100.0 if s2_count > 0 else 0.0)
        comparison["categories"][cat] = {
            "season1_count": s1_count, "season2_count": s2_count,
            "change_percent": change_pct,
            "trend": "increase" if change_pct > 10 else ("decrease" if change_pct < -10 else "stable"),
        }
    return comparison
