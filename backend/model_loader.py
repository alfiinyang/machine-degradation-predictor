"""
backend/model_loader.py
-----------------------
Manages downloading, caching, and loading model artifacts from Hugging Face:
repository 'alfiinyang/GBdegradation'.
"""

import os
import urllib.request
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

HF_REPO_ID = "alfiinyang/GBdegradation"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

REQUIRED_FILES = [
    "model.joblib",
    "feature_list.joblib",
    "pop_stats.joblib"
]

# Baseline fallback population statistics matching the assessment dataset
DEFAULT_POP_STATS = {
    "temp_mean": 63.218276767342296,
    "temp_std": 5.229951739262637
}

DEFAULT_FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "run_hours_since_maintenance",
    "temp_rolling_mean_6h",
    "vib_rolling_mean_6h",
    "temp_delta_1h",
    "vib_delta_1h",
    "vib_peak_12h",
    "is_temp_anomaly"
]


def ensure_model_files(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, str]:
    """
    Ensures all required model artifacts are downloaded and available locally.
    Returns a dict mapping filename to local file path.
    """
    os.makedirs(model_dir, exist_ok=True)
    paths = {}

    for fname in REQUIRED_FILES:
        local_path = os.path.join(model_dir, fname)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{fname}"
            logger.info(f"Downloading {fname} from Hugging Face ({url})...")
            try:
                # Add User-Agent header to prevent 403 blocks
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(local_path, "wb") as out_f:
                    out_f.write(resp.read())
                logger.info(f"Successfully downloaded {fname} ({os.path.getsize(local_path)} bytes)")
            except Exception as e:
                logger.warning(f"Could not download {fname} from Hugging Face: {e}")
        paths[fname] = local_path

    return paths


def load_artifacts(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    """
    Loads and returns: dict with (model, features, pop_stats).
    Includes resilient fallback if artifacts are unreachable or version incompatibilities occur.
    """
    import joblib

    paths = ensure_model_files(model_dir)

    # 1. Load feature list
    features = DEFAULT_FEATURES
    feat_path = paths.get("feature_list.joblib")
    if feat_path and os.path.exists(feat_path) and os.path.getsize(feat_path) > 0:
        try:
            features = joblib.load(feat_path)
        except Exception as e:
            logger.warning(f"Note on feature_list loading: {e}")

    # 2. Load population statistics
    pop_stats = DEFAULT_POP_STATS
    stats_path = paths.get("pop_stats.joblib")
    if stats_path and os.path.exists(stats_path) and os.path.getsize(stats_path) > 0:
        try:
            pop_stats = joblib.load(stats_path)
        except Exception as e:
            logger.warning(f"Note on pop_stats loading: {e}")

    # 3. Load HistGradientBoosting model
    model = None
    model_path = paths.get("model.joblib")
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.warning(f"Note on model loading: {e}")

    return {
        "model": model,
        "features": features,
        "pop_stats": pop_stats
    }
