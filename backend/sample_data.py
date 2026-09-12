"""
backend/sample_data.py
----------------------
Provides realistic sample operational telemetry for manufacturing plant machinery:
- 15 Established Machines (MCH-200 to MCH-214) across Lines A, B, and C over 14 days of hourly telemetry.
- 2 Newly Commissioned Machines (MCH-300, MCH-301) with ~3 days (72h) of telemetry (Cold-Start).
Embedded with realistic industrial degradation patterns (vibration spikes, thermal creep, and maintenance wear).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_sample_telemetry(days_of_history: int = 14) -> pd.DataFrame:
    """
    Generates realistic hourly sensor telemetry matching project2_manufacturing_sensors.csv schema.
    Columns: [timestamp, machine_id, line, temperature_c, vibration_mm_s, run_hours_since_maintenance]
    """
    np.random.seed(42)
    end_date = datetime(2026, 4, 30, 23, 0, 0)
    start_date = end_date - timedelta(days=days_of_history)
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")

    lines = ["Line A", "Line B", "Line C"]
    established_machines = [f"MCH-{200 + i}" for i in range(15)]
    new_machines = ["MCH-300", "MCH-301"]

    records = []

    # Baseline operating conditions
    for idx, machine_id in enumerate(established_machines):
        line = lines[idx % len(lines)]
        base_temp = 62.0 + (idx % 4) * 1.5
        base_vib = 0.55 + (idx % 3) * 0.12
        initial_run_hours = int(np.random.randint(40, 240))

        # Designate specific machines to exhibit degradation patterns
        is_high_risk = machine_id in ["MCH-205", "MCH-211"]
        is_warning = machine_id in ["MCH-202", "MCH-208"]

        run_hours = initial_run_hours

        for step, ts in enumerate(date_range):
            hours_from_end = (end_date - ts).total_seconds() / 3600.0

            # Increment run hours
            run_hours += 1
            if not is_high_risk and run_hours > 360 and np.random.rand() < 0.15:
                # Maintenance reset
                run_hours = 0

            # Thermal and vibration noise
            hour_of_day = ts.hour
            ambient_factor = np.sin((hour_of_day - 6) / 24.0 * 2 * np.pi) * 2.0
            temp_noise = np.random.normal(0, 1.8)
            vib_noise = np.random.normal(0, 0.08)

            temp = base_temp + ambient_factor + temp_noise
            vib = max(0.1, base_vib + vib_noise)

            # Inject progressive degradation leading to the final hours
            if is_high_risk and hours_from_end < 72:
                # Degradation ramp: temperature and vibration climb
                progress = (72 - hours_from_end) / 72.0
                temp += progress * 14.5
                vib += progress * 1.85
                run_hours = 320 + int(progress * 40)
            elif is_warning and hours_from_end < 48:
                progress = (48 - hours_from_end) / 48.0
                vib += progress * 0.65
                run_hours = 280 + int(progress * 25)

            records.append({
                "timestamp": ts,
                "machine_id": machine_id,
                "line": line,
                "temperature_c": round(float(temp), 2),
                "vibration_mm_s": round(float(vib), 3),
                "run_hours_since_maintenance": int(run_hours)
            })

    # Add Cold-Start Machines (only last 72 hours active)
    cold_start_dates = [ts for ts in date_range if (end_date - ts).total_seconds() <= 72 * 3600]
    for idx, machine_id in enumerate(new_machines):
        line = "Line C" if idx == 0 else "Line A"
        base_temp = 64.0 + idx * 2.0
        base_vib = 0.45 + idx * 0.1
        run_hours = 0

        for ts in cold_start_dates:
            run_hours += 1
            temp = base_temp + np.random.normal(0, 2.0)
            vib = max(0.1, base_vib + np.random.normal(0, 0.06))

            records.append({
                "timestamp": ts,
                "machine_id": machine_id,
                "line": line,
                "temperature_c": round(float(temp), 2),
                "vibration_mm_s": round(float(vib), 3),
                "run_hours_since_maintenance": int(run_hours)
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
