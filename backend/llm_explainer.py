"""
backend/llm_explainer.py
------------------------
Implements the runtime LLM diagnostic explanation engine from Project2_manufacturing.ipynb
(Cells 31 and 34).
Uses 'ChatGoogleGenerativeAI' with model 'gemini-3.6-flash'.
Produces concise, technically grounded executive memorandums explaining the failure risk
and underlying sensor drivers (Vibration, Temperature, Run-Hours) to Plant Managers.
"""

import os
import json
from typing import Dict, Any, Optional

MAINTENANCE_EXPLAIN_PROMPT = """
You are a Senior Predictive Maintenance Engineer at Flour Mills of Nigeria.
Analyze the following industrial asset data and explain the failure risk to the Plant Manager.
Be specific about which telemetry features (Temperature, Vibration, Run-hours, Delta drifts) are driving the risk.
Keep the explanation clear, professional, grounded strictly in the numbers provided, and under 3 concise sentences.

Machine Data:
{machine_json}

Technical Explanation:"""


def explain_machine_risk(
    machine_record: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash",
    temperature: float = 0.2
) -> Dict[str, str]:
    """
    Invokes ChatGoogleGenerativeAI to generate a grounded explanation of machine health.

    Parameters:
        machine_record: Dict containing sensor telemetry, failure probability, and recommendation.
        api_key: Google Gemini API key or environment variable.
        model_name: Default 'gemini-3.6-flash'.
        temperature: Low temperature (0.2) for strictly factual grounding.

    Returns:
        dict: {"machine_id": ..., "status": "success"|"fallback"|"error", "explanation": ..., "message": ...}
    """
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("gemini_key")
    )

    machine_id = machine_record.get("machine_id", "MCH-Asset")
    line = machine_record.get("line", "Production Line")
    prob = round(float(machine_record.get("failure_probability", 0.0)) * 100, 1)
    temp = round(float(machine_record.get("temperature_c", 0.0)), 1)
    vib = round(float(machine_record.get("vibration_mm_s", 0.0)), 2)
    vib_peak = round(float(machine_record.get("vib_peak_12h", 0.0)), 2)
    run_hours = int(machine_record.get("run_hours_since_maintenance", 0))
    rec = machine_record.get("recommendation", "Normal Operation")

    # Clean payload for LLM injection
    payload = {
        "machine_id": machine_id,
        "line": line,
        "failure_probability_pct": f"{prob}%",
        "current_temperature_c": temp,
        "current_vibration_mm_s": vib,
        "peak_vibration_12h_mm_s": vib_peak,
        "run_hours_since_maintenance": run_hours,
        "temperature_6h_rolling_mean": round(float(machine_record.get("temp_rolling_mean_6h", temp)), 1),
        "recommended_action": rec
    }

    # Fallback simulation if no key is provided
    if not resolved_key:
        if prob >= 80.0:
            simulated = (
                f"Asset {machine_id} ({line}) exhibits severe failure risk ({prob}% probability) driven by elevated "
                f"12-hour peak vibrations ({vib_peak} mm/s) compounded by {run_hours} accumulated run-hours since overhaul. "
                f"Immediate mechanical inspection of bearings and rotating assemblies is mandated before catastrophic seizure."
            )
        elif prob >= 50.0:
            simulated = (
                f"Asset {machine_id} ({line}) has entered moderate wear warning status ({prob}% failure risk) with "
                f"accumulated operational time of {run_hours} hours and temperature levels reaching {temp}°C. "
                f"Pre-emptive lubrication and sensor recalibration should be scheduled during the next shift change."
            )
        else:
            simulated = (
                f"Asset {machine_id} ({line}) is operating reliably with a nominal failure probability of {prob}%. "
                f"Vibration levels ({vib} mm/s) and core temperature ({temp}°C) remain comfortably within design baselines."
            )

        return {
            "machine_id": machine_id,
            "status": "fallback",
            "explanation": simulated,
            "message": "Generated using rule-based plant heuristics. Enter a Gemini API Key in the sidebar for live LLM inference."
        }

    # Call Gemini via LangChain
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import PromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=resolved_key,
            temperature=temperature
        )

        prompt = PromptTemplate.from_template(MAINTENANCE_EXPLAIN_PROMPT)
        chain = prompt | llm

        response = chain.invoke({"machine_json": json.dumps(payload, indent=2)})
        content = response.content

        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and "text" in content[0]:
                text = content[0]["text"]
            else:
                text = str(content[0])
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        return {
            "machine_id": machine_id,
            "status": "success",
            "explanation": text.strip(),
            "message": f"Successfully generated via {model_name}"
        }

    except Exception as e:
        return {
            "machine_id": machine_id,
            "status": "error",
            "explanation": (
                f"Asset {machine_id} on {line} is evaluated at a {prob}% failure probability. "
                f"Operating metrics: {temp}°C, {vib_peak} mm/s peak vibration, and {run_hours} run-hours. Action: {rec}"
            ),
            "message": f"LLM Call Error: {e}"
        }
