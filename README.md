# Machine Degradation Predictor & Plant Reliability Intelligence

An interactive industrial predictive maintenance system that forecasts machine failure risks before breakdown, flags critical wear anomalies, and generates data-grounded engineering explanations and conversational Q&A using AI.

Presentation slides are here: [(see slides here)](https://docs.google.com/presentation/d/1FSWLbhVNmv5VGpJUxty-eArpCBssKPZFnhIaIDUm9p4/edit?usp=sharing).

---


## 1. Problem Understanding

In continuous manufacturing operations (such as Flour Mills of Nigeria milling and packaging lines), unexpected machinery downtime creates severe operational disruptions:
- **Catastrophic Line Stoppages:** When high-torque motors, bearings, or extruders fail unexpectedly, the entire line halts. This leads to wasted work-in-progress materials, missed delivery deadlines, and costly emergency technician call-outs.
- **Purely Reactive Maintenance:** Without real-time degradation visibility, maintenance teams only respond *after* a breakdown has occurred.
- **Alarm Fatigue & Opaque Warnings:** Traditional industrial telemetry systems either trigger nuisance threshold alarms or display an unexplained "red light" with no actionable diagnosis of what is failing or why.

**Core Objective:** Deliver an intuitive, self-service dashboard that the plant maintenance and reliability engineering team can use independently to:
1. Track the real-time health and failure probability of every operational asset across production lines.
2. Flag degrading machines with clear risk levels (**Critical**, **Warning**, **Normal**) before line stoppage occurs.
3. Provide plain-English, AI-generated engineering explanations diagnosing the root physical drivers (temperature drift, vibration peaks, cumulative run-hours) grounded strictly in telemetry.
4. Provide an interactive Plant Analyst Q&A chat interface where technicians and managers can ask free-text questions about fleet risk and maintenance priorities.

---

## 2. Approach: System Architecture and Methodology

The solution integrates supervised machine learning for degradation probability scoring, expert heuristic rules for physical maintenance recommendations, and Generative AI for grounded explanations and conversational intelligence.

```
       Industrial Telemetry (Hourly Temp, Vib, Run-Hours)
                                │
                                ▼
         Feature Engineering & Anomaly Detection Pipeline
      (6h Rolling Means, 1h Deltas, 12h Peak Vib, Z-Scores)
                                │
                                ▼
         Supervised Champion: HistGradientBoosting (Recall 1.0)
                                │
                                ▼
                Operational Decision Threshold (0.80)
               /                                     \
              ▼                                       ▼
  Heuristic Maintenance Actions          State Extraction & Prompt Injection
(Bearing check, Lube service)              (Vib peak, Temp mean, Run-hours)
              \                                       /
               ▼                                     ▼
                Streamlit Plant Dashboard & Grounded Chatbot
                       (Powered by Gemini 3.6 Flash)
```

---

### A. Failure Prediction & Model Selection

#### The Industrial Cost Asymmetry
In manufacturing operations, the business cost function is heavily asymmetric:
- **Cost of a False Negative (Missed Breakdown):** Catastrophic. Halts factory lines, ruins in-process grain/flour batches, damages downstream equipment, and incurs expensive emergency repairs.
- **Cost of a False Positive (False Alarm):** Minimal. A maintenance technician performs a brief 10-minute diagnostic check on bearings, lubrication, and belt tension.

Consequently, **Recall is the primary optimization metric**. Any candidate model that misses actual failure events cannot be deployed safely in plant operations.

#### Benchmark Across Paradigms
Four candidate models across supervised tree ensembles and unsupervised anomaly detection were evaluated on a chronological 20% holdout test split:

| Candidate Model | Paradigm | Precision | Recall | F1-Score | Production Decision |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **HistGradientBoosting** | Supervised Tree Ensemble | 0.300 | **1.000 (100%)** | **0.462** | **Selected Production Champion** |
| **Random Forest** | Supervised Bagging Ensemble | **0.500** | 0.333 | 0.400 | Disqualified (misses 67% of failures) |
| **Isolation Forest** | Unsupervised Anomaly Detection | 0.015 | 0.333 | 0.028 | High false alarm rate |
| **PCA Reconstruction** | Unsupervised Subspace Projection | 0.012 | 0.333 | 0.023 | Insufficient degradation signal |

**Evaluation Insights:**
1. **HistGradientBoosting Achieved 100% Recall:** Captured every single machine failure event in the evaluation window.
2. **Why Tree Boosting Outperformed Random Forest:** Degradation telemetry exhibits sequential wear accumulation. HistGradientBoosting iteratively refines residual errors on continuous trend features, whereas simple bagging averages away subtle precursor spikes.
3. **Failure of Unsupervised Baselines:** Isolation Forest and PCA generated extreme false alarm rates (Precision $< 2\%$) because normal operational noise (e.g. ambient shifts, motor spin-up) was frequently misclassified as failure anomalies.

---

### B. Operational Decision Threshold Optimization

Failure probabilities from the champion model were analyzed across decision thresholds:

| Threshold | Precision | Recall | F1-Score | False Alarms (Count) | Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50** | 0.300 | 1.000 | 0.462 | 7 | Baseline threshold |
| **0.60** | 0.300 | 1.000 | 0.462 | 7 | Stable boundary |
| **0.70** | 0.300 | 1.000 | 0.462 | 7 | High confidence |
| **0.80** | **0.300** | **1.000** | **0.462** | **7** | **Optimal Operating Threshold** |
| **0.90** | 0.667 | 0.667 | 0.667 | 1 | Unsafe (misses 33% of failures) |

**Decision:** A decision threshold of **`0.80`** was selected for production alerts. It maximizes model confidence and prevents alert fatigue while preserving perfect **100% recall**.

Trained model artifacts and preprocessing pipelines are hosted on Hugging Face: [`alfiinyang/GBdegradation`](https://huggingface.co/alfiinyang/GBdegradation).

---

### C. Feature Engineering & Cold-Start Strategy

#### Feature Engineering
Raw telemetry is grouped by `machine_id` and transformed into degradation signals:
- **Rolling Means (6h):** `temp_rolling_mean_6h`, `vib_rolling_mean_6h` (filters transient noise to detect sustained thermal/vibrational climb).
- **Rate of Change (1h Deltas):** `temp_delta_1h`, `vib_delta_1h` (identifies rapid mechanical drift or sudden unbalance).
- **Peak Vibration (12h Max):** `vib_peak_12h` (captures bearing impact spikes and resonance).
- **Cumulative Wear:** `run_hours_since_maintenance` (identifies components nearing lubrication exhaustion).

#### Cold-Start Handling for Newly Commissioned Machines
Newly installed machines (`MCH-300`, `MCH-301`) lack historical run data:
- **Fleet Population Baselines:** Population mean ($63.22^\circ\text{C}$) and standard deviation ($5.23^\circ\text{C}$) establish baseline operating bounds. Z-score anomaly tests (`is_temp_anomaly = |z| > 3`) flag extreme deviations from day one.
- **Adaptive Windowing:** Rolling statistics use `min_periods=1` so newly installed assets can be evaluated immediately without discarding early records.

---

### D. Grounded AI Explanations & Plant Analyst Chat

#### Grounded Root-Cause Explanations
When an asset is selected in the dashboard:
- The system extracts its operational state (failure risk, current & rolling temperature, vibration peak, run-hours, line, and maintenance status) into a structured JSON payload.
- It prompts **Google Gemini (`gemini-3.6-flash`)** via LangChain with low temperature (`0.2`) as a *Senior Reliability Engineer*.
- Gemini generates a concise executive memorandum explaining *why* the machine is at risk based strictly on the numerical telemetry.

#### Plant Analyst Grounded Q&A Chat
A dedicated conversational interface allows plant managers and shift supervisors to ask natural language questions such as:
- *"Which machines need immediate attention this shift?"*
- *"Why is MCH-205 flagged as critical?"*
- *"What is the condition of newly commissioned machine MCH-300?"*
- *"Which lines are running with high vibration?"*

The Q&A engine dynamically extracts fleet context and feeds it to Gemini to synthesize precise, data-grounded answers.

---

### E. User Interface (Streamlit Dashboard)

An interactive, responsive dashboard featuring:
- **Fleet KPI Metrics:** Total assets, critical risk count, warning count, normal fleet count, and average fleet risk.
- **Tab 1: Fleet Risk Monitor:** Sortable table of latest machine readings, status badges, drill-down metrics, and one-click LLM diagnostic memorandum.
- **Tab 2: Degradation Visualizer:** Comparative failure probability ranking chart, 2D sensor boundary scatter plot, and chronological multi-sensor degradation trajectories.
- **Tab 3: Plant Analyst Q&A Chat:** Conversational chat box with message history and quick-prompt suggestion buttons.
- **Tab 4: Model Selection & Strategic Rationale:** Comprehensive technical write-up, comparative benchmark tables, threshold curve, and Hugging Face artifact details.

---

## 3. How to Run and Use

### Live Cloud Deployment
The interactive dashboard is publicly hosted on Render and ready to use without local installation:
- **Live Application URL:** [https://machine-degradation-predictor.onrender.com/](https://machine-degradation-predictor.onrender.com/)

---

### Local Installation

#### Prerequisites
- Python 3.10 to 3.12 installed.

#### Step 1: Clone the Repository
```bash
git clone https://github.com/alfiinyang/machine-degradation-predictor.git
cd machine-degradation-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Gemini API Key (Optional)
Set the API key in your environment, or enter it directly into the dashboard sidebar:
```bash
# On Linux/macOS
export GEMINI_API_KEY="your-gemini-api-key"

# On Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key"
```
*(Note: If no API key is provided, the application automatically uses deterministic rule-based analyst heuristics so all features remain fully functional.)*

### Step 4: Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 4. Repository Structure

```
machine-degradation-predictor/
├── .streamlit/
│   └── config.toml                  # Streamlit headless server configuration
├── backend/
│   ├── __init__.py                  # Package marker
│   ├── model_loader.py              # Downloads & caches artifacts from Hugging Face
│   ├── sample_data.py               # Operational sensor telemetry generator (17 assets)
│   ├── telemetry_preprocessor.py     # Rolling feature engineering & inference pipeline
│   ├── alert_engine.py              # Threshold rules & heuristic maintenance actions
│   ├── llm_explainer.py             # Single-asset LLM diagnostic memos via Gemini
│   └── plant_analyst_qa.py          # Grounded conversational Q&A chat engine
├── models/                          # Cache directory for downloaded model artifacts
├── app.py                           # Main Streamlit web application & dashboard
├── Project2_manufacturing.ipynb     # Original research, EDA, modeling, and evaluation notebook
├── requirements.txt                 # Pinned project dependencies
├── render.yaml                      # Render cloud deployment blueprint
├── FMN Machine Degradation Predictor.pptx # Project presentation slide deck
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

---

## 5. Limitations & Future Work

1. **Failure Data Scarcity & Class Imbalance:** Real-world failure events in industrial manufacturing are naturally rare, creating severe class imbalance during training. Collecting more historical failure telemetry or generating realistic synthetic failure sequences (e.g. via Time-Series GANs or SMOTE) will provide a more balanced training distribution and enhance precision without sacrificing 100% recall.
2. **Remaining Useful Life (RUL) Regression:** Currently, the system predicts failure probability over a defined forward horizon. With longer run-to-failure datasets (e.g. NASA C-MAPSS or multi-year plant history), survival analysis (Weibull models) or direct RUL regression could estimate remaining operating hours.
3. **Lack of Dynamic Data Lookup / Tool Calling:** The conversational interface currently constructs prompts from pre-aggregated in-memory fleet context rather than equipping the LLM with an autonomous tool/function-calling layer (e.g., SQL/vector queries) to look up granular sensor telemetry on demand for specific machines, shifts, or anomaly intervals.
4. **Prompt Loading Overhead & Token RPM Consumption:** Injecting multi-asset operational state tables directly into system prompts inflates the token payload per user query. This increases input prompt latency and heightens the risk of exhausting Requests Per Minute (RPM) and Tokens Per Minute (TPM) rate limits during concurrent or extended conversational sessions.
5. **Automated CMMS Integration:** Automatically dispatching work orders to computerized maintenance management systems (SAP PM / Maximo) upon critical risk detection.
6. **Local SLM Edge Deployment:** Quantizing a lightweight model (e.g., Gemma 2 2B) for on-premise edge hardware to generate explanations in disconnected plant environments.


