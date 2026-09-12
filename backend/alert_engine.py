"""
backend/alert_engine.py
-----------------------
Implements business rules, threshold scoring, and automated maintenance recommendations
from Project2_manufacturing.ipynb (Cells 23, 24, and 26).
- High Failure Risk: Probability >= Operational Threshold (default 0.80)
- Moderate Warning: 0.50 <= Probability < Operational Threshold
- Normal Operation: Probability < 0.50
"""

import pandas as pd
import numpy as np


def compute_alerts(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Computes failure risk categories, badges, and automated technician recommendations
    based on predicted failure probabilities and underlying sensor telemetry.
    """
    res = df.copy()

    # Risk level classification
    res["is_critical"] = res["failure_probability"] >= threshold
    res["is_warning"] = (res["failure_probability"] >= 0.50) & (res["failure_probability"] < threshold)
    res["is_normal"] = res["failure_probability"] < 0.50

    def get_status_badge(row):
        if row["is_critical"]:
            return "🚨 Action Required: Critical Failure Risk"
        elif row["is_warning"]:
            return "⚠️ Attention: Elevating Wear Warning"
        return "✅ Normal Operation"

    res["alert_status"] = res.apply(get_status_badge, axis=1)

    def generate_recommendation(row):
        if row["is_normal"]:
            return "Normal Operation - No Action Required"

        recs = []
        # Check vibration drivers
        if row.get("vib_peak_12h", 0) > 1.25 or row.get("vibration_mm_s", 0) > 1.2:
            recs.append("High vibration detected: Inspect bearings, mounting bolts, and shaft alignment.")

        # Check cumulative wear
        if row.get("run_hours_since_maintenance", 0) >= 300:
            recs.append("High run-hours accumulated: Schedule routine lubricant change and filter check.")

        # Check thermal elevation
        if row.get("temperature_c", 0) >= 75.0 or row.get("is_temp_anomaly", 0) == 1:
            recs.append("Thermal escalation detected: Inspect cooling circuit, lubrication flow, and heat exchangers.")

        # Check rapid acceleration
        if abs(row.get("temp_delta_1h", 0)) >= 4.0 or abs(row.get("vib_delta_1h", 0)) >= 0.35:
            recs.append("Rapid sensor drift: Conduct immediate on-site diagnostic scan for mechanical looseness.")

        if not recs:
            recs.append("Elevated composite risk: Schedule preventative diagnostic scan during upcoming shift window.")

        return " | ".join(recs)

    res["recommendation"] = res.apply(generate_recommendation, axis=1)

    return res
