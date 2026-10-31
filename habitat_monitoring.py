"""
EcoSonic AI - Habitat Change Monitoring Module

IMPORTANT: Changes in sound patterns may be influenced by weather, season,
time of day, recording equipment, background noise, and natural fluctuations.
Sound pattern changes alone do not prove habitat destruction or biodiversity loss.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_all_recordings, get_locations
from config import STATUS_LABELS

DISCLAIMER = (
    "**Important**: Changes in wildlife sound patterns can be caused by many factors "
    "including weather conditions, seasonal cycles, time of day, recording equipment quality, "
    "background noise levels, and natural population fluctuations. Reduced sound activity "
    "alone does NOT prove habitat destruction or biodiversity loss. These indicators are "
    "intended to flag areas for professional ecological review, not to serve as definitive "
    "scientific conclusions."
)


def analyze_habitat_changes(location: Optional[str] = None) -> Dict:
    recordings = get_all_recordings()
    if not recordings:
        return {"status": "no_data", "message": "No recordings available for analysis."}
    df = pd.DataFrame(recordings)
    if location and location != "All":
        df = df[df["location"] == location]
    if df.empty:
        return {"status": "no_data", "message": f"No recordings for location: {location}"}

    df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")
    df = df.dropna(subset=["recording_date"])
    if df.empty:
        return {"status": "no_data", "message": "No valid recording dates found."}

    results = {
        "location": location or "All Locations",
        "total_recordings": len(df),
        "date_range": {
            "start": df["recording_date"].min().strftime("%Y-%m-%d"),
            "end": df["recording_date"].max().strftime("%Y-%m-%d"),
        },
        "disclaimer": DISCLAIMER,
        "indicators": [],
        "overall_status": "stable",
        "category_trends": {},
        "seasonal_patterns": {},
    }

    df["year_month"] = df["recording_date"].dt.to_period("M").astype(str)

    for cat in ["Bird Calls", "Frog Calls", "Insect Sounds"]:
        cat_data = df[df["predicted_category"] == cat]
        monthly_counts = cat_data.groupby("year_month").size()
        if len(monthly_counts) >= 2:
            values = monthly_counts.values.astype(float)
            mid = len(values) // 2
            first_half_avg = np.mean(values[:mid]) if mid > 0 else 0
            second_half_avg = np.mean(values[mid:]) if mid < len(values) else 0
            change = ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0
            trend = "increase" if change > 20 else ("decrease" if change < -20 else "stable")
            results["category_trends"][cat] = {
                "first_period_avg": round(first_half_avg, 1),
                "second_period_avg": round(second_half_avg, 1),
                "change_percent": round(change, 1),
                "trend": trend,
                "data_points": len(monthly_counts),
            }
            if trend == "decrease" and change < -30:
                results["indicators"].append({
                    "type": "decrease", "category": cat,
                    "change": round(change, 1),
                    "severity": "moderate" if change > -50 else "significant",
                    "message": f"{cat} activity decreased by {abs(round(change, 1))}% in recent period",
                    "note": "This may be due to seasonal variation, weather, or recording conditions.",
                })
            elif trend == "increase" and change > 30:
                results["indicators"].append({
                    "type": "increase", "category": cat,
                    "change": round(change, 1),
                    "message": f"{cat} activity increased by {round(change, 1)}% in recent period",
                })
        else:
            results["category_trends"][cat] = {
                "trend": "insufficient_data",
                "message": "Not enough data points for trend analysis",
            }

    for season in ["Summer", "Rainy", "Winter"]:
        season_data = df[df["season"] == season]
        if not season_data.empty:
            mode_vals = season_data["predicted_category"].mode()
            results["seasonal_patterns"][season] = {
                "total_recordings": len(season_data),
                "dominant_category": mode_vals.iloc[0] if not mode_vals.empty else "N/A",
                "avg_confidence": round(season_data["confidence_score"].mean(), 2) if season_data["confidence_score"].notna().any() else 0,
                "category_counts": season_data["predicted_category"].value_counts().to_dict(),
            }

    significant_changes = [i for i in results["indicators"] if i.get("severity") == "significant"]
    moderate_changes = [i for i in results["indicators"] if i.get("severity") == "moderate"]
    if significant_changes:
        results["overall_status"] = "review"
        results["status_label"] = STATUS_LABELS["review"]
    elif moderate_changes:
        results["overall_status"] = "variation"
        results["status_label"] = STATUS_LABELS["variation"]
    else:
        results["overall_status"] = "stable"
        results["status_label"] = STATUS_LABELS["stable"]
    return results


def get_change_summary() -> List[Dict]:
    locations = get_locations()
    summaries = []
    for loc in locations:
        analysis = analyze_habitat_changes(loc)
        summaries.append({
            "location": loc,
            "status": analysis.get("overall_status", "no_data"),
            "status_label": analysis.get("status_label", "No Data"),
            "total_recordings": analysis.get("total_recordings", 0),
            "indicators_count": len(analysis.get("indicators", [])),
            "category_trends": analysis.get("category_trends", {}),
        })
    return summaries
