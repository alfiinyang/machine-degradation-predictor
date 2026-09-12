"""
backend/plant_analyst_qa.py
---------------------------
Interactive Grounded Q&A Chat Engine for Plant Maintenance Analysts.
Replicates and extends the Plant Analyst Chat Interface from Project2_manufacturing.ipynb (Cell 33).
Uses 'ChatGoogleGenerativeAI' with model 'gemini-3.6-flash' to answer user queries grounded
directly in the live telemetry, degradation trends, and risk scores of all plant machines.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import pandas as pd


PLANT_ANALYST_SYSTEM_PROMPT = """
You are an expert Senior Predictive Maintenance & Reliability Analyst for Flour Mills of Nigeria (FMN).
Your goal is to answer the plant manager's or technician's question accurately, grounded strictly in the live fleet telemetry provided below.

Rules:
1. Always base your answers on the specific machine metrics, lines, failure probabilities, and recommendations provided in the Fleet Context.
2. If the user asks about a specific machine, cite its exact numbers: temperature, vibration, 12h peak vibration, run hours, and risk level.
3. If asked which machines need attention or maintenance, clearly prioritize Critical Risk (>= 80% probability) followed by Warning status (>= 50% probability).
4. Be concise, actionable, and engineer-focused. Use bullet points where appropriate.
5. If the telemetry data does not contain the answer, state that clearly rather than inventing numbers.

Live Fleet Context:
{fleet_context}
"""


def extract_fleet_context(snapshot_df: pd.DataFrame, query: str) -> str:
    """
    Extracts relevant machine metrics and summaries from the latest fleet snapshot
    to construct a compact, highly relevant grounding context for the LLM.
    """
    if snapshot_df is None or snapshot_df.empty:
        return "No machine telemetry currently loaded."

    # Identify if a specific machine is mentioned (e.g., MCH-205, MCH-300)
    mentioned_machines = re.findall(r"MCH-\d+", query.upper())

    context_blocks = []

    # 1. Fleet Overview Summary
    total_assets = len(snapshot_df)
    critical_count = int(snapshot_df["is_critical"].sum())
    warning_count = int(snapshot_df["is_warning"].sum())
    avg_prob = round(float(snapshot_df["failure_probability"].mean()) * 100, 1)

    summary_str = (
        f"Fleet Overview: {total_assets} total monitored machines across Line A, Line B, Line C.\n"
        f"Critical High-Risk: {critical_count} machines | Moderate Warning: {warning_count} machines | "
        f"Fleet Avg Risk: {avg_prob}%\n"
    )
    context_blocks.append(summary_str)

    # 2. Priority Risk Table (Machines with risk >= 0.50 or top 5 by risk)
    top_risk_df = snapshot_df.sort_values(by="failure_probability", ascending=False)
    high_priority = top_risk_df[top_risk_df["failure_probability"] >= 0.50]
    if high_priority.empty:
        high_priority = top_risk_df.head(3)

    fields_to_show = [
        "machine_id", "line", "failure_probability", "temperature_c",
        "vibration_mm_s", "vib_peak_12h", "run_hours_since_maintenance",
        "alert_status", "recommendation"
    ]
    avail_fields = [f for f in fields_to_show if f in high_priority.columns]
    context_blocks.append("High Priority Machines:\n" + high_priority[avail_fields].to_string(index=False))

    # 3. If specific machines were mentioned, extract their exact detailed record
    if mentioned_machines:
        specific_matches = snapshot_df[snapshot_df["machine_id"].isin(mentioned_machines)]
        if not specific_matches.empty:
            context_blocks.append(
                f"\nDetailed telemetry for queried machines ({', '.join(mentioned_machines)}):\n" +
                specific_matches[avail_fields].to_string(index=False)
            )

    return "\n\n".join(context_blocks)


def answer_plant_query(
    query: str,
    snapshot_df: pd.DataFrame,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash"
) -> Dict[str, Any]:
    """
    Answers free-text plant queries grounded in the latest machine data.
    """
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("gemini_key")
    )

    fleet_context = extract_fleet_context(snapshot_df, query)

    # If no API key, provide a high-fidelity deterministic grounded response
    if not resolved_key:
        top_risk = snapshot_df.sort_values(by="failure_probability", ascending=False)
        critical_machines = top_risk[top_risk["is_critical"]]
        warning_machines = top_risk[top_risk["is_warning"]]

        # Check for specific machine inquiry
        matches = re.findall(r"MCH-\d+", query.upper())
        if matches:
            mch = matches[0]
            mch_row = snapshot_df[snapshot_df["machine_id"] == mch]
            if not mch_row.empty:
                r = mch_row.iloc[0]
                prob_pct = round(r["failure_probability"] * 100, 1)
                text = (
                    f"**Asset Status: {mch} ({r['line']})**\n\n"
                    f"- **Failure Probability:** `{prob_pct}%` ({r['alert_status']})\n"
                    f"- **Vibration:** Current `{r['vibration_mm_s']} mm/s` (12h Peak: `{r['vib_peak_12h']} mm/s`)\n"
                    f"- **Temperature:** `{r['temperature_c']}°C` (6h Mean: `{r['temp_rolling_mean_6h']}°C`)\n"
                    f"- **Run-Hours Since Maintenance:** `{int(r['run_hours_since_maintenance'])} hrs`\n\n"
                    f"**Recommended Action:** {r['recommendation']}"
                )
            else:
                text = f"Machine `{mch}` was not found in the currently loaded fleet dataset."
        elif "attention" in query.lower() or "risk" in query.lower() or "fail" in query.lower() or "priority" in query.lower():
            if not critical_machines.empty:
                crit_list = ", ".join([f"`{row['machine_id']}` ({row['line']}, {row['failure_probability']*100:.1f}%)" for _, row in critical_machines.iterrows()])
                text = (
                    f"### 🚨 Machines Requiring Immediate Attention\n\n"
                    f"The following **{len(critical_machines)} asset(s)** exceed the critical failure threshold:\n"
                    f"- **Critical:** {crit_list}\n\n"
                )
                if not warning_machines.empty:
                    warn_list = ", ".join([f"`{row['machine_id']}` ({row['line']})" for _, row in warning_machines.iterrows()])
                    text += f"- **Moderate Warning:** {warn_list}\n\n"
                text += "**Key Priority:** Inspect the critical assets immediately for bearing vibration and thermal drift before line stoppage."
            else:
                text = "✅ All monitored machines are currently running within acceptable operational limits. No immediate critical failure risks detected."
        else:
            text = (
                f"### Fleet Telemetry Summary\n\n"
                f"- **Total Monitored Machines:** {len(snapshot_df)}\n"
                f"- **Critical Assets:** {len(critical_machines)}\n"
                f"- **Warning Assets:** {len(warning_machines)}\n"
                f"- **Fleet Average Failure Probability:** {snapshot_df['failure_probability'].mean()*100:.1f}%\n\n"
                f"Enter a Gemini API Key in the sidebar to ask arbitrary complex engineering questions powered by `gemini-3.6-flash`."
            )

        return {
            "answer": text,
            "status": "fallback",
            "message": "Answer grounded in fleet metrics using deterministic analyst logic. Add a Gemini API key for full generative LLM Q&A."
        }

    # Live call to Gemini via LangChain
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=resolved_key,
            temperature=0.2
        )

        messages = [
            SystemMessage(content=PLANT_ANALYST_SYSTEM_PROMPT.format(fleet_context=fleet_context))
        ]

        # Add recent conversation history for multi-turn context
        if chat_history:
            for msg in chat_history[-6:]:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))

        messages.append(HumanMessage(content=query))

        response = llm.invoke(messages)
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
            "answer": text.strip(),
            "status": "success",
            "message": f"Answer grounded live via {model_name}"
        }

    except Exception as e:
        return {
            "answer": f"Encountered an issue processing query: {e}. Please check your API key and connection.",
            "status": "error",
            "message": str(e)
        }
