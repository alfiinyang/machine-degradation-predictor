"""
app.py
------
Streamlit Dashboard for Project 2: Machine Degradation Predictor & Plant Reliability Intelligence.
Champion Model: alfiinyang/GBdegradation (Hugging Face)
Grounded LLM: gemini-3.6-flash via ChatGoogleGenerativeAI (LangChain)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Local backend modules
from backend.model_loader import load_artifacts
from backend.sample_data import get_sample_telemetry
from backend.telemetry_preprocessor import run_pipeline
from backend.alert_engine import compute_alerts
from backend.llm_explainer import explain_machine_risk
from backend.plant_analyst_qa import answer_plant_query

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Degradation & Plant Reliability Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .explanation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 14px 0;
        line-height: 1.6;
        color: #0C4A6E;
    }
    .rec-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
        font-size: 0.92rem;
        color: #78350F;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def load_and_process_telemetry(uploaded_file=None):
    artifacts = get_cached_artifacts()
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            raw_df = get_sample_telemetry(days_of_history=14)
    else:
        raw_df = get_sample_telemetry(days_of_history=14)

    processed_df = run_pipeline(raw_df, artifacts)
    return processed_df


# -----------------------------------------------------------------------------
# Sidebar: Controls, Key Configuration & Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏭 Plant Intelligence")
    st.caption("Predictive Maintenance & Degradation Early Warning System")

    st.markdown("---")
    st.markdown("#### 🔑 LLM Configuration")
    env_gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_gemini_key if env_gemini_key else "",
        type="password",
        help="Required for live grounded explanations and Q&A via gemini-3.6-flash."
    )

    if api_key_input:
        st.success("🟢 Gemini 3.6 Flash Active")
    else:
        st.info("🟡 Rule-based Fallback Active (Enter API Key above for live LLM)")

    st.markdown("---")
    st.markdown("#### 📂 Telemetry Data Source")
    data_source_mode = st.radio(
        "Select Data Input",
        ["Built-in Plant Telemetry (17 Assets)", "Upload Custom Sensor CSV"],
        index=0
    )

    uploaded_csv = None
    if data_source_mode == "Upload Custom Sensor CSV":
        uploaded_csv = st.file_uploader(
            "Upload Telemetry CSV",
            type=["csv"],
            help="Expected columns: timestamp, machine_id, line, temperature_c, vibration_mm_s, run_hours_since_maintenance"
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Operational Threshold")
    selected_threshold = st.slider(
        "Failure Risk Alert Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.80,
        step=0.05,
        help="Tuned at 0.80 to maintain 100% failure recall while suppressing nuisance alarms."
    )

    st.markdown("---")
    st.markdown("#### 🔗 Production Links")
    st.markdown("- [Live Web App (Render)](https://machine-degradation-predictor.onrender.com/)")
    st.markdown("- [Hugging Face Model Hub](https://huggingface.co/alfiinyang/GBdegradation)")
    st.markdown("- [GitHub Repository](https://github.com/alfiinyang/machine-degradation-predictor)")
    st.caption("FMN AI Internship — Project 2")



# -----------------------------------------------------------------------------
# Load and Process Data
# -----------------------------------------------------------------------------
with st.spinner("Processing telemetry through feature engineering & inference pipeline..."):
    df_telemetry = load_and_process_telemetry(uploaded_csv)

# Filter by line
all_lines = sorted(df_telemetry["line"].unique().tolist())
selected_lines = st.sidebar.multiselect("Filter by Production Line", all_lines, default=all_lines)
if not selected_lines:
    selected_lines = all_lines

df_filtered_telemetry = df_telemetry[df_telemetry["line"].isin(selected_lines)]

# Extract latest operational reading per machine
latest_snapshot = df_filtered_telemetry.groupby("machine_id").tail(1).sort_values(
    by="failure_probability", ascending=False
).reset_index(drop=True)

# Compute alerts and maintenance recommendations
df_alerts = compute_alerts(latest_snapshot, threshold=selected_threshold)


# -----------------------------------------------------------------------------
# Application Header & KPI Cards
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Machine Degradation & Plant Reliability Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Real-time industrial sensor telemetry, machine learning failure risk forecasting, and AI-grounded root cause diagnostics.'
    '</div>',
    unsafe_allow_html=True
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_assets = len(df_alerts)
crit_count = int(df_alerts["is_critical"].sum())
warn_count = int(df_alerts["is_warning"].sum())
norm_count = int(df_alerts["is_normal"].sum())
avg_risk = df_alerts["failure_probability"].mean() * 100

with kpi1:
    st.metric("Monitored Assets", f"{total_assets} Units", f"{len(selected_lines)} Lines")
with kpi2:
    st.metric("Critical High Risk", f"{crit_count} Assets", "Immediate Action", delta_color="inverse")
with kpi3:
    st.metric("Elevating Wear", f"{warn_count} Assets", "Schedule Service", delta_color="off")
with kpi4:
    st.metric("Normal Operations", f"{norm_count} Assets", f"{(norm_count/max(1,total_assets))*100:.0f}% Healthy")
with kpi5:
    st.metric("Fleet Average Risk", f"{avg_risk:.1f}%", f"Threshold: {selected_threshold*100:.0f}%")

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Tabs: Fleet Monitor, Visualizer, Plant Analyst Chat, Model Rationale
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Fleet Risk Monitor & Explain",
    "📈 Degradation Visualizer",
    "💬 Plant Analyst Q&A Chat",
    "🧠 Model Selection & Rationale"
])


# -----------------------------------------------------------------------------
# TAB 1: Fleet Risk Monitor & Click-to-Explain
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Current Fleet Risk Status (Latest Telemetry)")
    st.caption(f"Showing latest telemetry readings across {total_assets} monitored machines. Sorted by failure risk.")

    # Format table for display
    display_df = df_alerts[[
        "machine_id", "line", "failure_probability", "temperature_c",
        "vibration_mm_s", "vib_peak_12h", "run_hours_since_maintenance", "alert_status"
    ]].copy()

    display_df["failure_probability"] = (display_df["failure_probability"] * 100).round(1).astype(str) + "%"
    display_df.rename(columns={
        "machine_id": "Machine ID",
        "line": "Line",
        "failure_probability": "Failure Risk",
        "temperature_c": "Temp (°C)",
        "vibration_mm_s": "Vibration (mm/s)",
        "vib_peak_12h": "12h Peak Vib (mm/s)",
        "run_hours_since_maintenance": "Run-Hours",
        "alert_status": "Risk Status"
    }, inplace=True)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Asset Drill-down & Live LLM Diagnostic Memo")
    st.write("Select any machine below to inspect its operational metrics and receive an AI-generated technical explanation.")

    # Machine selector
    machine_options = df_alerts["machine_id"].tolist()
    default_index = 0
    # Pre-select highest risk machine if any critical
    if crit_count > 0:
        default_index = 0

    selected_machine = st.selectbox(
        "Select Machine for Deep Inspection:",
        machine_options,
        index=default_index
    )

    machine_record = df_alerts[df_alerts["machine_id"] == selected_machine].iloc[0]

    drill_col1, drill_col2, drill_col3, drill_col4 = st.columns(4)
    with drill_col1:
        st.metric("Failure Probability", f"{machine_record['failure_probability']*100:.1f}%", machine_record["alert_status"])
    with drill_col2:
        st.metric("Current Temperature", f"{machine_record['temperature_c']:.1f}°C", f"6h Mean: {machine_record['temp_rolling_mean_6h']:.1f}°C")
    with drill_col3:
        st.metric("Vibration (Current / 12h Peak)", f"{machine_record['vibration_mm_s']:.2f} mm/s", f"Peak: {machine_record['vib_peak_12h']:.2f} mm/s")
    with drill_col4:
        st.metric("Run-Hours Since Service", f"{int(machine_record['run_hours_since_maintenance'])} hrs", "Overhaul at 350h")

    # Action Recommendation Banner
    st.markdown(f"""
    <div class="rec-box">
        <b>Recommended Technician Action:</b> {machine_record['recommendation']}
    </div>
    """, unsafe_allow_html=True)

    # Explanation Trigger
    if f"memo_{selected_machine}" not in st.session_state:
        st.session_state[f"memo_{selected_machine}"] = None

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button("Generate Plant Engineer Memo", key=f"btn_{selected_machine}", type="primary")

    if generate_clicked or st.session_state[f"memo_{selected_machine}"] is None:
        with st.spinner("Invoking Gemini 3.6 Flash diagnostic analysis..."):
            memo_res = explain_machine_risk(
                machine_record=machine_record.to_dict(),
                api_key=api_key_input,
                model_name="gemini-3.6-flash"
            )
            st.session_state[f"memo_{selected_machine}"] = memo_res

    memo = st.session_state[f"memo_{selected_machine}"]

    st.markdown(f"""
    <div class="explanation-box">
        <b>Senior Reliability Engineer Assessment ({selected_machine} — {machine_record['line']}):</b><br><br>
        {memo['explanation']}
    </div>
    """, unsafe_allow_html=True)

    if memo.get("status") == "success":
        st.caption("⚡ Factual diagnostic memo generated live via `gemini-3.6-flash`.")
    else:
        st.caption(f"ℹ️ {memo.get('message', '')}")


# -----------------------------------------------------------------------------
# TAB 2: Telemetry & Degradation Visualizer
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Plant Telemetry & Degradation Visualizer")

    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        # Comparative Risk Bar Chart
        plot_df = df_alerts.sort_values(by="failure_probability", ascending=True).copy()
        plot_df["prob_pct"] = plot_df["failure_probability"] * 100

        colors = []
        for p in plot_df["failure_probability"]:
            if p >= selected_threshold:
                colors.append("#DC2626")  # Red
            elif p >= 0.50:
                colors.append("#F59E0B")  # Amber
            else:
                colors.append("#10B981")  # Green

        fig_bar = go.Figure(go.Bar(
            x=plot_df["prob_pct"],
            y=plot_df["machine_id"],
            orientation="h",
            marker=dict(color=colors),
            text=plot_df["prob_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside"
        ))
        fig_bar.add_vline(x=selected_threshold*100, line_dash="dash", line_color="#991B1B",
                          annotation_text=f"Alert Threshold ({selected_threshold*100:.0f}%)", annotation_position="top right")
        fig_bar.update_layout(
            title="Machine Failure Probability Ranking",
            xaxis_title="Failure Probability (%)",
            yaxis_title="Asset ID",
            height=420,
            margin=dict(l=20, r=30, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with v_col2:
        # Sensor Operating Boundary (Scatter: Temp vs Vibration vs Run-Hours)
        fig_scatter = px.scatter(
            df_alerts,
            x="temperature_c",
            y="vibration_mm_s",
            size="run_hours_since_maintenance",
            color="alert_status",
            hover_name="machine_id",
            color_discrete_map={
                "🚨 Action Required: Critical Failure Risk": "#DC2626",
                "⚠️ Attention: Elevating Wear Warning": "#F59E0B",
                "✅ Normal Operation": "#10B981"
            },
            title="Sensor Space: Vibration vs Temperature (Bubble Size = Run Hours)"
        )
        fig_scatter.update_layout(
            xaxis_title="Temperature (°C)",
            yaxis_title="Vibration (mm/s)",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Historical Sensor Degradation Trajectory: {selected_machine}")
    st.caption("Review chronological sensor trends leading up to the current operational reading.")

    history_machine = df_filtered_telemetry[df_filtered_telemetry["machine_id"] == selected_machine].sort_values("timestamp")

    if not history_machine.empty:
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            # Temperature trend
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=history_machine["timestamp"],
                y=history_machine["temperature_c"],
                name="Hourly Temperature",
                line=dict(color="#94A3B8", width=1.5)
            ))
            fig_temp.add_trace(go.Scatter(
                x=history_machine["timestamp"],
                y=history_machine["temp_rolling_mean_6h"],
                name="6h Rolling Mean",
                line=dict(color="#DC2626", width=2.5)
            ))
            fig_temp.update_layout(
                title=f"{selected_machine} — Thermal Trajectory (°C)",
                xaxis_title="Timestamp",
                yaxis_title="Temperature (°C)",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with t_col2:
            # Vibration trend
            fig_vib = go.Figure()
            fig_vib.add_trace(go.Scatter(
                x=history_machine["timestamp"],
                y=history_machine["vibration_mm_s"],
                name="Hourly Vibration",
                line=dict(color="#94A3B8", width=1.5)
            ))
            fig_vib.add_trace(go.Scatter(
                x=history_machine["timestamp"],
                y=history_machine["vib_peak_12h"],
                name="12h Peak Vibration",
                line=dict(color="#2563EB", width=2.5)
            ))
            fig_vib.update_layout(
                title=f"{selected_machine} — Vibration Trajectory (mm/s)",
                xaxis_title="Timestamp",
                yaxis_title="Vibration (mm/s)",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_vib, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Plant Analyst Q&A Chat (Grounded LLM)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("💬 Plant Maintenance Analyst Q&A")
    st.write(
        "Ask free-text questions about fleet risk status, specific machine health, degradation indicators, or maintenance priorities. "
        "All answers are synthesized from the live operational dataset."
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your Plant Maintenance Reliability Analyst. Ask me anything about current asset failure risks, sensor trends, or shift maintenance priorities."
            }
        ]

    # Quick prompt buttons
    st.markdown("**Suggested Questions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_query = None
    with q_col1:
        if st.button("🚨 Which machines need attention this shift?"):
            quick_query = "Which machines need attention this shift?"
    with q_col2:
        if st.button(f"🔍 What is driving risk on {selected_machine}?"):
            quick_query = f"What is driving risk on {selected_machine}?"
    with q_col3:
        if st.button("🆕 What is the status of newly commissioned machines?"):
            quick_query = "What is the status of newly commissioned machines MCH-300 and MCH-301?"

    # Display chat messages from history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about plant machine health...")

    query_to_send = quick_query or user_input

    if query_to_send:
        # Display user message
        st.session_state["chat_history"].append({"role": "user", "content": query_to_send})
        with st.chat_message("user"):
            st.markdown(query_to_send)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing plant fleet telemetry and reasoning with Gemini..."):
                qa_res = answer_plant_query(
                    query=query_to_send,
                    snapshot_df=df_alerts,
                    chat_history=st.session_state["chat_history"][:-1],
                    api_key=api_key_input,
                    model_name="gemini-3.6-flash"
                )
                answer_text = qa_res["answer"]
                st.markdown(answer_text)
                if qa_res.get("status") == "success":
                    st.caption("⚡ Live analysis grounded in telemetry via `gemini-3.6-flash`.")
                else:
                    st.caption(f"ℹ️ {qa_res.get('message', '')}")

        st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})


# -----------------------------------------------------------------------------
# TAB 4: Model Selection & Strategic Rationale
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Model Selection, Evaluation Rigor & Strategic Rationale")
    st.markdown(r"""
    #### 1. The Cost Asymmetry in Industrial Manufacturing

    In predictive maintenance for manufacturing operations (e.g. Flour Mills of Nigeria milling and packaging lines), the business cost structure is heavily asymmetric:
    - **False Negative (Missed Failure):** A machine fails without warning. Consequences include halted line throughput, ruined batch material, emergency mechanical overhaul, and high downtime costs.
    - **False Positive (False Alarm):** An inspection is triggered on a functional machine. Consequences are minor—a technician spends 10 minutes checking bearing lubrication and thermal readings.

    Therefore, **Recall** is the primary optimization objective. Any model missing failures is disqualified from production.

    ---

    #### 2. Benchmark Evaluation Across Paradigms
    Four model families were benchmarked on the 20% chronological holdout test split:

    | Candidate Model | Paradigm | Precision | Recall | F1-Score | Production Decision |
    | :--- | :--- | :---: | :---: | :---: | :--- |
    | **HistGradientBoosting** | Supervised Tree Ensemble | 0.300 | **1.000 (100%)** | **0.462** | **Selected Production Champion** |
    | **Random Forest** | Supervised Bagging Ensemble | **0.500** | 0.333 | 0.400 | Misses 67% of failures |
    | **Isolation Forest** | Unsupervised Anomaly Detection | 0.015 | 0.333 | 0.028 | High false alarm rate |
    | **PCA Reconstruction** | Unsupervised Subspace Projection | 0.012 | 0.333 | 0.023 | Insufficient degradation signal |

    **Key Takeaways:**
    - **HistGradientBoosting** was the only architecture to achieve **Recall = 1.0 (100%)**, successfully capturing every failure event in the evaluation window.
    - Unsupervised anomaly detectors (Isolation Forest, PCA) suffered from low precision ($<2\%$), generating prohibitive alarm fatigue without improving recall.

    ---

    #### 3. Operational Threshold Optimization
    Failure probabilities from the champion model were analyzed across multiple decision thresholds:

    | Threshold | Precision | Recall | F1-Score | False Alarms (Count) | Assessment |
    | :---: | :---: | :---: | :---: | :---: | :--- |
    | **0.50** | 0.300 | 1.000 | 0.462 | 7 | Baseline threshold |
    | **0.60** | 0.300 | 1.000 | 0.462 | 7 | Stable boundary |
    | **0.70** | 0.300 | 1.000 | 0.462 | 7 | High confidence |
    | **0.80** | **0.300** | **1.000** | **0.462** | **7** | **Optimal Operating Threshold** |
    | **0.90** | 0.667 | 0.667 | 0.667 | 1 | Unsafe (misses 33% of failures) |

    **Threshold Selection Decision:** An operational threshold of **`0.80`** maximizes diagnostic confidence while preserving perfect 100% recall.

    ---

    #### 4. Cold-Start Strategy for Newly Commissioned Assets
    Newly installed machines (`MCH-300`, `MCH-301`) lack the multi-month history needed to train asset-specific models. The system handles cold start by:
    1. **Fleet Population Baselines:** Computing population-level temperature mean ($63.2^\circ\text{C}$) and standard deviation ($5.23^\circ\text{C}$) to detect anomalies via z-score deviations immediately from day one.
    2. **Minimum-Period Rolling Windows:** Using adaptive rolling windows (`min_periods=1`) so rolling means and peaks accumulate gracefully without dropping initial records.

    ---

    #### 5. Artifact Packaging & Hugging Face Hub Deployment
    Trained weights, feature schemas, and baseline statistics are hosted publicly:
    - **Hugging Face Repository:** [`alfiinyang/GBdegradation`](https://huggingface.co/alfiinyang/GBdegradation)
    - **Artifacts:** `model.joblib`, `feature_list.joblib`, `pop_stats.joblib`, `preprocessing.py`
    """)
