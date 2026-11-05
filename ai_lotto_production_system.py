"""
Premier Lotto Data Lab — Baba Ijebu ML service
Run: pip install flask flask-cors numpy pandas xgboost scikit-learn
      python ai_lotto_production_system.py

Environment:
  LOTTO_HISTORY_JSON=/path/to/saved_history.json  (optional)
  LOTTO_MODEL_DIR=./models                    (optional)
  PORT=5000                                   (optional)

Expected history JSON: a list of objects containing game/game_profile, date,
w (five Winning numbers), and m (five Machine numbers). The service accepts
that format directly; it does not seed production with demonstration draws.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(os.environ.get("LOTTO_MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE: dict[tuple[str, str], XGBClassifier] = {}
MODEL_LOCK = threading.RLock()
FEATURE_COLUMNS = [
    "hot_cold", "lap_flag", "mac_crossover",
    "base90_comp", "vtrac_group", "mirror_weight"
]
MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.03,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2,
)


def _five(values: Any, field: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError(f"{field} must contain exactly five numbers")
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{field} contains an invalid number")
        number = int(value)
        if number < 1 or number > 90:
            raise ValueError(f"{field} numbers must be between 1 and 90")
        result.append(number)
    return result


def normalize_history(records: list[dict]) -> list[dict]:
    """Convert Wix Saved Results rows or canonical records into newest-first draws."""
    normalized = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        profile = str(row.get("game_profile") or row.get("game") or "").strip()
        if not profile:
            continue
        winning = row.get("w") or row.get("winning") or [
            row.get(f"win_{i}") for i in range(1, 6)
        ]
        machine = row.get("m") or row.get("machine") or [
            row.get(f"mac_{i}") for i in range(1, 6)
        ]
        try:
            w = _five(winning, "Winning")
            m = _five(machine, "Machine")
        except (TypeError, ValueError):
            continue
        date = str(row.get("date") or row.get("draw_date") or "")
        normalized.append({
            "draw_id": str(row.get("draw_id") or row.get("id") or f"{profile}:{date}:{index}"),
            "game_profile": profile,
            "date": date,
            "win": w,
            "mac": m,
        })
    # ISO date strings sort chronologically. Wix dates are expected to be ISO-like.
    normalized.sort(key=lambda item: (item["date"], item["draw_id"]), reverse=True)
    return normalized


class LottoFeatureFactory:
    """Builds leakage-aware 90-candidate training rows and live inference rows."""

    @staticmethod
    def construct_feature_matrix(
        historical_draws: list[dict],
        game_profile: str,
        outcome: str = "win",
        calculation_depth: int = 100,
    ) -> pd.DataFrame:
        if outcome not in ("win", "mac"):
            raise ValueError("outcome must be 'win' or 'mac'")
        board = [d for d in historical_draws if d["game_profile"] == game_profile]
        rows = []
        # board is newest-first. Each active draw is older than its known next draw.
        for step in range(1, len(board)):
            active = board[step]
            target = board[step - 1]
            window = board[step:step + calculation_depth]
            freq = {n: 0.0 for n in range(1, 91)}
            for offset, historical in enumerate(window):
                decay = 1.0 / math.log2(offset + 2)
                for number in historical[outcome]:
                    freq[number] += decay

            current_pool = active[outcome]
            past_pool = board[step + 1][outcome] if step + 1 < len(board) else []
            crossover_pool = active["mac" if outcome == "win" else "win"]
            target_pool = target[outcome]
            current_center = current_pool[2]
            past_center = past_pool[2] if len(past_pool) == 5 else None

            for ball in range(1, 91):
                mirror_weight = 0.0
                if ball > 9 and ball % 10 != 0:
                    mirrored = int(str(ball)[::-1])
                    if 1 <= mirrored <= 90:
                        mirror_weight = freq[mirrored] * 0.45
                rows.append({
                    "draw_id": active["draw_id"],
                    "ball_number": ball,
                    "hot_cold": freq[ball],
                    "lap_flag": int(ball == current_center and current_center == past_center),
                    "mac_crossover": int(ball in crossover_pool),
                    "base90_comp": int(ball == (90 - current_center)),
                    "vtrac_group": ball % 5,
                    "mirror_weight": mirror_weight,
                    "is_winning_number": int(ball in target_pool),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def construct_live_inference_vector(
        historical_draws: list[dict],
        game_profile: str,
        outcome: str = "win",
        calculation_depth: int = 100,
    ) -> pd.DataFrame:
        if outcome not in ("win", "mac"):
            raise ValueError("outcome must be 'win' or 'mac'")
        board = [d for d in historical_draws if d["game_profile"] == game_profile]
        if len(board) < 2:
            raise ValueError("At least two saved draws are required for live inference")
        latest = board[0]
        window = board[:calculation_depth]
        freq = {n: 0.0 for n in range(1, 91)}
        for offset, historical in enumerate(window):
            decay = 1.0 / math.log2(offset + 2)
            for number in historical[outcome]:
                freq[number] += decay
        current_pool = latest[outcome]
        previous_pool = board[1][outcome]
        crossover_pool = latest["mac" if outcome == "win" else "win"]
        center = current_pool[2]
        previous_center = previous_pool[2]
        rows = []
        for ball in range(1, 91):
            mirror_weight = 0.0
            if ball > 9 and ball % 10 != 0:
                mirrored = int(str(ball)[::-1])
                if 1 <= mirrored <= 90:
                    mirror_weight = freq[mirrored] * 0.45
            rows.append({
                "ball_number": ball,
                "hot_cold": freq[ball],
                "lap_flag": int(ball == center and center == previous_center),
                "mac_crossover": int(ball in crossover_pool),
                "base90_comp": int(ball == (90 - center)),
                "vtrac_group": ball % 5,
                "mirror_weight": mirror_weight,
            })
        return pd.DataFrame(rows)


class LottoXGBoostTrainer:
    def __init__(self):
        self.model = XGBClassifier(**MODEL_PARAMS)

    def train(self, training_dataframe: pd.DataFrame) -> dict:
        if training_dataframe.empty:
            raise ValueError("Historical records are insufficient to build training frames.")
        labels = training_dataframe["is_winning_number"].astype(int)
        if labels.nunique() < 2:
            raise ValueError("Training data must contain both winning and non-winning examples.")
        X = training_dataframe[FEATURE_COLUMNS]
        self.model.fit(X, labels)
        return {"training_rows": int(len(training_dataframe)), "positive_labels": int(labels.sum())}

    def generate_predictions(self, inference_dataframe: pd.DataFrame) -> list[dict]:
        probabilities = self.model.predict_proba(inference_dataframe[FEATURE_COLUMNS])[:, 1]
        results = [
            {"number": int(inference_dataframe.iloc[i]["ball_number"]),
             "probability": float(probabilities[i])}
            for i in range(len(probabilities))
        ]
        return sorted(results, key=lambda item: (-item["probability"], item["number"]))

    def serialize_model(self, file_path: str | Path) -> None:
        with open(file_path, "wb") as handle:
            pickle.dump(self.model, handle)

    def deserialize_model(self, file_path: str | Path) -> None:
        with open(file_path, "rb") as handle:
            self.model = pickle.load(handle)


def load_history() -> list[dict]:
    path = os.environ.get("LOTTO_HISTORY_JSON")
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("History JSON must be a list of saved draw records")
    return normalize_history(payload)


def train_and_predict(history: list[dict], profile: str, outcome: str, depth: int) -> dict:
    training = LottoFeatureFactory.construct_feature_matrix(history, profile, outcome, depth)
    if training.empty:
        raise ValueError(f"Not enough historical draws for {profile} {outcome.upper()} training.")
    live = LottoFeatureFactory.construct_live_inference_vector(history, profile, outcome, depth)
    trainer = LottoXGBoostTrainer()
    training_summary = trainer.train(training)
    ranked = trainer.generate_predictions(live)
    model_path = MODEL_DIR / f"{profile.lower().replace(' ', '_')}_{outcome}.pkl"
    trainer.serialize_model(model_path)
    with MODEL_LOCK:
        MODEL_CACHE[(profile, outcome)] = trainer.model
    return {
        "ranked": ranked,
        "training": training_summary,
        "model_file": str(model_path),
        "model_type": "XGBoost",
        "hyperparameters": MODEL_PARAMS,
    }


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Baba Ijebu XGBoost", "model": "XGBoost"})


@app.route("/api/v1/ai/predict", methods=["GET", "POST"])
def execute_ai_prediction_flow():
    try:
        payload = request.get_json(silent=True) or {}
        profile = str(request.args.get("profile") or payload.get("profile") or "National").strip()
        depth = int(request.args.get("depth") or payload.get("depth") or 100)
        if depth < 2 or depth > 5000:
            return jsonify({"status": "error", "message": "depth must be between 2 and 5000"}), 400

        # Production accepts a supplied Saved Results snapshot or a configured history file.
        supplied = payload.get("historical_records")
        history = normalize_history(supplied) if isinstance(supplied, list) else load_history()
        profile_history = [row for row in history if row["game_profile"].casefold() == profile.casefold()]
        if len(profile_history) < 3:
            return jsonify({
                "status": "error",
                "code": "INSUFFICIENT_HISTORY",
                "message": f"At least 3 valid saved draws are required for {profile}.",
                "available_draws": len(profile_history),
            }), 400

        # Separate classifiers: Winning and Machine are trained and scored independently.
        winning = train_and_predict(history, profile, "win", depth)
        machine = train_and_predict(history, profile, "mac", depth)

        def shape(result):
            ranked = result["ranked"]
            return {
                "ranked_numbers": [x["number"] for x in ranked],
                "ranked_probabilities": ranked,
                "nap2_two_sure": [x["number"] for x in ranked[:2]],
                "nap3_three_direct": [x["number"] for x in ranked[:3]],
                "permutation_pool_top_10": [x["number"] for x in ranked[:10]],
                "training": result["training"],
                "model_type": result["model_type"],
                "hyperparameters": result["hyperparameters"],
            }

        return jsonify({
            "status": "success",
            "game_profile": profile,
            "calculation_depth": depth,
            "historical_draws_used": len(profile_history),
            "winning": shape(winning),
            "machine": shape(machine),
            "warning": "Lottery draws are random; model rankings are experimental scores, not guaranteed outcomes.",
        })
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction pipeline failed")
        return jsonify({"status": "error", "code": "MODEL_FAILURE", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
