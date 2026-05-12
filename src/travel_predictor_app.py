#!/usr/bin/env python3
"""Travel predictor web app.

Trains a regression model from a CSV with columns:
    fecha(or index), tramo_horario, viajes, viajes_km, tipo_code

The app predicts the number of trips (viajes) for a future date and time slot,
returning a central estimate, an uncertainty interval, a qualitative label
relative to the historical mean of that slot, and a chart with train/eval data.

This version also uses `tipo_code` as an input feature. At prediction time,
it looks up the `tipo_code` for the selected date from the CSV and feeds it
into the model together with the selected date, derived temporal features,
and the selected time slot.

Run:
    pip install -r requirements.txt
    python travel_predictor_app.py --csv viajes.csv --port 5000

Then open:
    http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATE_COL = "fecha"
SLOT_COL = "tramo_horario"
TARGET_COL = "viajes"
TYPE_COL = "tipo_code"

MODEL_MEAN = "model_mean_v4.joblib"
MODEL_LOW = "model_low_v4.joblib"
MODEL_HIGH = "model_high_v4.joblib"
META_JSON = "meta_v4.json"

SPLIT_RE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")


@dataclass
class ModelBundle:
    mean_model: GradientBoostingRegressor
    low_model: GradientBoostingRegressor
    high_model: GradientBoostingRegressor
    metadata: Dict


def parse_slot(slot: str) -> Tuple[int, int, int]:
    """Parse tramo_horario like '6-9' -> (6, 9, 3)."""
    if not isinstance(slot, str):
        return 0, 0, 0
    m = SPLIT_RE.match(slot)
    if not m:
        return 0, 0, 0
    start, end = int(m.group(1)), int(m.group(2))
    duration = max(0, end - start)
    return start, end, duration


def _unique_codes_in_order(values: pd.Series) -> list[str]:
    """Return unique string codes preserving first appearance order."""
    seen = set()
    out: list[str] = []
    for item in values.astype(str).fillna("").str.strip().tolist():
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Allow the CSV to come either with a `fecha` column or with the first
    # column named `index`, as in the example shared by the user.
    if DATE_COL not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": DATE_COL})

    required = {DATE_COL, SLOT_COL, TARGET_COL, TYPE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(missing)}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, SLOT_COL, TARGET_COL, TYPE_COL])
    df[SLOT_COL] = df[SLOT_COL].astype(str).str.strip()
    df[TYPE_COL] = df[TYPE_COL].astype(str).str.strip()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)
    df = df.sort_values([SLOT_COL, DATE_COL]).reset_index(drop=True)

    start_end = df[SLOT_COL].apply(parse_slot)
    df["slot_start"] = [x[0] for x in start_end]
    df["slot_end"] = [x[1] for x in start_end]
    df["slot_duration"] = [x[2] for x in start_end]

    slot_order = (
        df[[SLOT_COL, "slot_start", "slot_end"]]
        .drop_duplicates()
        .sort_values(["slot_start", "slot_end", SLOT_COL])
    )
    slot_to_rank = {slot: i for i, slot in enumerate(slot_order[SLOT_COL].tolist())}
    df["slot_rank"] = df[SLOT_COL].map(slot_to_rank).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["dayofweek"] = dt.dt.dayofweek
    out["dayofyear"] = dt.dt.dayofyear
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create supervised features with lagged information within each slot."""
    out = add_cyclical_features(df)
    out = out.sort_values([SLOT_COL, DATE_COL]).reset_index(drop=True)

    def _slot_lags(group: pd.DataFrame) -> pd.DataFrame:
        s = group[TARGET_COL].astype(float)
        return pd.DataFrame(
            {
                "lag_1": s.shift(1),
                "lag_7": s.shift(7),
                "roll_mean_4": s.shift(1).rolling(4, min_periods=1).mean(),
                "roll_std_4": s.shift(1).rolling(4, min_periods=2).std(),
            },
            index=group.index,
        )

    lag_df = out.groupby(SLOT_COL, group_keys=False).apply(_slot_lags)
    out = pd.concat([out, lag_df], axis=1)

    slot_mean = out.groupby(SLOT_COL)[TARGET_COL].transform("mean")
    global_mean = float(out[TARGET_COL].mean())
    for col in ["lag_1", "lag_7", "roll_mean_4", "roll_std_4"]:
        if col == "roll_std_4":
            out[col] = out[col].fillna(0.0)
        else:
            out[col] = out[col].fillna(slot_mean).fillna(global_mean)

    return out


def make_feature_matrix(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    out = add_cyclical_features(df)
    out["slot_start"] = pd.to_numeric(out.get("slot_start"), errors="coerce").fillna(0)
    out["slot_end"] = pd.to_numeric(out.get("slot_end"), errors="coerce").fillna(0)
    out["slot_duration"] = pd.to_numeric(out.get("slot_duration"), errors="coerce").fillna(0)
    out["slot_rank"] = pd.to_numeric(out.get("slot_rank"), errors="coerce").fillna(-1).astype(int)

    # These lag columns are required by the model. They will be present at
    # training time, and set externally for inference.
    for col in ["lag_1", "lag_7", "roll_mean_4", "roll_std_4"]:
        if col not in out.columns:
            out[col] = np.nan

    slot_categories = metadata["slot_categories"]
    type_categories = range(0,11)#metadata["type_categories"]

    slot_dummies = pd.get_dummies(out[SLOT_COL].astype(str), prefix="slot")
    for slot in slot_categories:
        key = f"slot_{slot}"
        if key not in slot_dummies.columns:
            slot_dummies[key] = 0
    slot_dummies = slot_dummies[[f"slot_{slot}" for slot in slot_categories]]

    type_dummies = pd.get_dummies(out[TYPE_COL].astype(str), prefix="type")
    for code in type_categories:
        key = f"type_{code}"
        if key not in type_dummies.columns:
            type_dummies[key] = 0
    type_dummies = type_dummies[[f"type_{code}" for code in type_categories]]

    features = pd.concat(
        [
            out[
                [
                    "year",
                    "month",
                    "day",
                    "dayofweek",
                    "dayofyear",
                    "weekofyear",
                    "is_weekend",
                    "dow_sin",
                    "dow_cos",
                    "doy_sin",
                    "doy_cos",
                    "slot_start",
                    "slot_end",
                    "slot_duration",
                    "slot_rank",
                    "lag_1",
                    "lag_7",
                    "roll_mean_4",
                    "roll_std_4",
                ]
            ],
            slot_dummies,
            type_dummies,
        ],
        axis=1,
    )

    return features.replace([np.inf, -np.inf], np.nan)


def train_models(df_raw: pd.DataFrame, model_dir: Path) -> ModelBundle:
    model_dir.mkdir(parents=True, exist_ok=True)

    df = build_training_frame(df_raw)
    unique_dates = sorted(pd.to_datetime(df[DATE_COL]).dt.normalize().unique())
    cut = max(1, int(len(unique_dates) * 0.8))
    train_dates = set(unique_dates[:cut])
    eval_dates = set(unique_dates[cut:])
    train_cutoff = pd.Timestamp(unique_dates[cut - 1]).date().isoformat()
    eval_start = pd.Timestamp(unique_dates[cut]).date().isoformat() if len(eval_dates) else train_cutoff

    # One code per date is expected; if the CSV contains more than one, we keep
    # the most frequent value for that date.
    type_code_by_date = (
        df.assign(date_key=df[DATE_COL].dt.strftime("%Y-%m-%d"))
        .groupby("date_key")[TYPE_COL]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.astype(str).iloc[0])
        .to_dict()
    )

    metadata = {
        "slot_categories": df[SLOT_COL].drop_duplicates().sort_values(
            key=lambda s: s.map(lambda x: parse_slot(x)[0])
        ).tolist(),
        "type_categories": _unique_codes_in_order(df[TYPE_COL]),
        "min_date": df[DATE_COL].min().date().isoformat(),
        "max_date": df[DATE_COL].max().date().isoformat(),
        "train_cutoff_date": train_cutoff,
        "eval_start_date": eval_start,
        "target_mean": float(df[TARGET_COL].mean()),
        "target_median": float(df[TARGET_COL].median()),
        "slot_means": df.groupby(SLOT_COL)[TARGET_COL].mean().to_dict(),
        "slot_std": df.groupby(SLOT_COL)[TARGET_COL].std().fillna(0.0).to_dict(),
        "slot_weekday_means": {
            f"{slot}__{dow}": float(val)
            for (slot, dow), val in df.groupby([SLOT_COL, "dayofweek"])[TARGET_COL].mean().items()
        },
        "type_code_by_date": type_code_by_date,
    }

    X = make_feature_matrix(df, metadata)
    y = np.log1p(df[TARGET_COL].astype(float).values)

    train_mask = df[DATE_COL].dt.normalize().isin(train_dates).values
    X_train, y_train = X.loc[train_mask].fillna(0.0), y[train_mask]
    X_val, y_val = X.loc[~train_mask].fillna(0.0), y[~train_mask]

    def fit_model(loss: str, alpha: Optional[float] = None) -> GradientBoostingRegressor:
        params = dict(
            loss=loss,
            learning_rate=0.05,
            n_estimators=350,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        )
        if alpha is not None:
            params["alpha"] = alpha
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        return model

    mean_model = fit_model("squared_error")
    low_model = fit_model("quantile", alpha=0.10)
    high_model = fit_model("quantile", alpha=0.90)

    val_pred = np.expm1(mean_model.predict(X_val)) if len(X_val) else np.array([])
    val_true = np.expm1(y_val) if len(y_val) else np.array([])
    report = {}
    if len(X_val):
        report = {
            "mae": float(mean_absolute_error(val_true, val_pred)),
            "rmse": float(np.sqrt(mean_squared_error(val_true, val_pred))),
            "r2": float(r2_score(val_true, val_pred)),
            "validation_rows": int(len(X_val)),
        }
    metadata["validation_report"] = report

    joblib.dump(mean_model, model_dir / MODEL_MEAN)
    joblib.dump(low_model, model_dir / MODEL_LOW)
    joblib.dump(high_model, model_dir / MODEL_HIGH)
    with open(model_dir / META_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return ModelBundle(mean_model=mean_model, low_model=low_model, high_model=high_model, metadata=metadata)


def load_bundle(model_dir: Path) -> Optional[ModelBundle]:
    meta_path = model_dir / META_JSON
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return ModelBundle(
        mean_model=joblib.load(model_dir / MODEL_MEAN),
        low_model=joblib.load(model_dir / MODEL_LOW),
        high_model=joblib.load(model_dir / MODEL_HIGH),
        metadata=metadata,
    )


def _slot_stats_for_prediction(history: pd.DataFrame, target_date: pd.Timestamp, slot: str) -> Dict[str, float]:
    history = history.copy()
    history[DATE_COL] = pd.to_datetime(history[DATE_COL])
    past = history[(history[DATE_COL] < target_date) & (history[SLOT_COL] == slot)].sort_values(DATE_COL)

    slot_mean = float(history.loc[history[SLOT_COL] == slot, TARGET_COL].mean()) if len(history) else 0.0
    if math.isnan(slot_mean):
        slot_mean = float(history[TARGET_COL].mean()) if len(history) else 0.0
    if math.isnan(slot_mean):
        slot_mean = 0.0

    lag_1 = slot_mean
    lag_7 = slot_mean
    roll_mean_4 = slot_mean
    roll_std_4 = 0.0

    if not past.empty:
        values = past[TARGET_COL].astype(float).values
        lag_1 = float(values[-1])
        lag_7 = float(values[-7]) if len(values) >= 7 else float(np.mean(values))
        recent = values[-4:]
        roll_mean_4 = float(np.mean(recent))
        roll_std_4 = float(np.std(recent, ddof=1)) if len(recent) >= 2 else 0.0

    return {
        "lag_1": lag_1,
        "lag_7": lag_7,
        "roll_mean_4": roll_mean_4,
        "roll_std_4": roll_std_4,
    }


def _lookup_type_code_for_date(history: pd.DataFrame, target_date: pd.Timestamp, metadata: Dict) -> str:
    """Get the type_code associated with the selected date from the CSV/history."""
    history = history.copy()
    history[DATE_COL] = pd.to_datetime(history[DATE_COL])
    key = pd.Timestamp(target_date).normalize().strftime("%Y-%m-%d")

    # First try the date->type_code map learned from the same CSV.
    mapped = metadata.get("type_code_by_date", {}).get(key)
    if mapped is not None and str(mapped).strip() != "":
        return str(mapped).strip()

    # Fallback: access the CSV rows directly for that date.
    same_day = history[history[DATE_COL].dt.normalize() == pd.Timestamp(target_date).normalize()]
    if same_day.empty:
        calendar_df = pd.read_csv("data\calendario_murcia_cartagena_22_25.csv")
        print(calendar_df.head())
        calendar_df['date']= pd.to_datetime(calendar_df['date'])
        calendar_df= calendar_df.set_index('date')
        print(calendar_df)
        return str(calendar_df.loc[target_date, 'tipo_code'])
        raise ValueError("No se ha encontrado un tipo_code asociado a la fecha seleccionada en el CSV.")

    values = same_day[TYPE_COL].astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        raise ValueError("No se ha podido recuperar un tipo_code válido para la fecha seleccionada.")

    mode = values.mode()
    return str(mode.iat[0] if not mode.empty else values.iloc[0]).strip()


def build_inference_row(history: pd.DataFrame, target_date: str, slot: str, metadata: Dict) -> pd.DataFrame:
    dt = pd.to_datetime(target_date, dayfirst=False, errors="coerce")
    if pd.isna(dt):
        raise ValueError("La fecha no es válida. Usa dd/mm/aaaa, yyyy-mm-dd o un selector de fecha válido.")

    if slot not in metadata["slot_categories"]:
        raise ValueError("El tramo horario seleccionado no existe en el CSV de entrenamiento.")

    type_code = _lookup_type_code_for_date(history, dt, metadata)

    start, end, duration = parse_slot(slot)
    base = pd.DataFrame(
        {
            DATE_COL: [dt],
            SLOT_COL: [slot],
            TYPE_COL: [type_code],
            "slot_start": [start],
            "slot_end": [end],
            "slot_duration": [duration],
            "slot_rank": [metadata["slot_categories"].index(slot)],
        }
    )
    base = add_cyclical_features(base)

    stats = _slot_stats_for_prediction(history, dt, slot)
    for k, v in stats.items():
        base[k] = v

    return base


def predict_one(history: pd.DataFrame, bundle: ModelBundle, target_date: str, slot: str) -> Dict[str, object]:
    row = build_inference_row(history, target_date, slot, bundle.metadata)
    print("rowww", row[TYPE_COL])
    metadata=  bundle.metadata
    metadata['type_categories'] = _unique_codes_in_order(row[TYPE_COL])
    metadata["slot_categories"]= history[SLOT_COL].drop_duplicates().sort_values(key=lambda s: s.map(lambda x: parse_slot(x)[0])).tolist()

    print(metadata['type_categories'])
    X = make_feature_matrix(row, metadata).fillna(0.0)
    print(X)
    for i in range(0,11):
        if i not in metadata['type_categories']:
            X[f'type_{i}']=False 
    

    pred = float(np.expm1(bundle.mean_model.predict(X)[0]))
    low = float(np.expm1(bundle.low_model.predict(X)[0]))
    high = float(np.expm1(bundle.high_model.predict(X)[0]))

    pred = max(0.0, pred)
    low = max(0.0, min(low, pred))
    high = max(low, max(high, pred))

    slot_mean = float(bundle.metadata.get("slot_means", {}).get(slot, bundle.metadata.get("target_mean", 0.0)))
    if math.isnan(slot_mean) or slot_mean <= 0:
        slot_mean = max(bundle.metadata.get("target_mean", 1.0), 1.0)

    ratio = pred / slot_mean if slot_mean else 1.0
    deviation_pct = (ratio - 1.0) * 100.0

    if ratio < 0.75:
        qualitative = "muy menor"
    elif ratio < 0.95:
        qualitative = "menor"
    elif ratio <= 1.05:
        qualitative = "similar"
    elif ratio <= 1.25:
        qualitative = "mayor"
    else:
        qualitative = "muy mayor"

    return {
        "estimate": pred,
        "low": low,
        "high": high,
        "slot_mean": slot_mean,
        "qualitative": qualitative,
        "deviation_pct": deviation_pct,
        "ratio": ratio,
        "type_code": str(metadata['type_categories'][0]),
    }


def build_boxplot_base64(history: pd.DataFrame, bundle: ModelBundle, target_date: str, slot: str) -> str:
    dt = pd.to_datetime(target_date, format="%Y-%m-%d")
    if pd.isna(dt):
        raise ValueError("La fecha no es válida para construir la gráfica.")

    type_code = _lookup_type_code_for_date(history, dt, bundle.metadata)

    h = history.copy()
    h[DATE_COL] = pd.to_datetime(h[DATE_COL])
    h["dayofweek"] = h[DATE_COL].dt.dayofweek

    # Prefer the same slot + same weekday + same type_code.
    filtered = h[
        (h[SLOT_COL] == slot)
        & (h["dayofweek"] == dt.dayofweek)
        & (h[TYPE_COL].astype(str).str.strip() == str(type_code))
    ].sort_values(DATE_COL)

    # Fallback if that subset is too small.
    if filtered.empty:
        filtered = h[(h[SLOT_COL] == slot) & (h["dayofweek"] == dt.dayofweek)].sort_values(DATE_COL)
    if filtered.empty:
        filtered = h[h[SLOT_COL] == slot].sort_values(DATE_COL)
    if filtered.empty:
        raise ValueError(
            "No hay suficientes datos para construir la gráfica de ese tramo horario y día de la semana."
        )

    values = filtered[TARGET_COL].astype(float).values
    pred = predict_one(history, bundle, target_date, slot)["estimate"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=140)

    bp = ax.boxplot(
        [values],
        vert=True,
        patch_artist=True,
        widths=0.38,
        showmeans=True,
        medianprops={"linewidth": 2},
        boxprops={"linewidth": 1.6},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
    )

    for box in bp["boxes"]:
        box.set_alpha(0.18)

    rng = np.random.default_rng(42)
    x_jitter = 1 + rng.normal(0, 0.03, size=len(values))
    ax.scatter(x_jitter, values, s=14, alpha=0.18, zorder=2)

    ax.scatter([1.18], [pred], s=110, marker="D", label=f"Predicción: {pred:.2f}", zorder=5)
    ax.annotate(
        f"{pred:.2f}",
        xy=(1.18, pred),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )

    ax.set_xticks([1])
    ax.set_xticklabels([f"{slot}\n{dt.day_name()}\n(tipo {type_code})"])
    ax.set_ylabel("Viajes")
    ax.set_title("Distribución histórica y predicción")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


HTML_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Predictor de viajes hacia el Campus Muralla del Mar</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #16202a;
      --muted: #5b6572;
      --accent: #2563eb;
      --accent2: #e8f0ff;
      --border: #d9e2f2;
      --shadow: 0 12px 30px rgba(20, 33, 61, 0.08);
    }
    body {
      margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: linear-gradient(180deg, var(--bg), #eef3fb); color: var(--text);
    }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 18px 44px; }
    .hero { display: grid; gap: 10px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 2rem; }
    p { margin: 0; color: var(--muted); line-height: 1.55; }
    .grid { display: grid; grid-template-columns: 1fr 1.15fr; gap: 18px; align-items: start; }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: 22px;
      box-shadow: var(--shadow); padding: 22px;
    }
    label { display: block; font-weight: 650; margin-bottom: 8px; }
    input, select, button {
      width: 100%; box-sizing: border-box; border-radius: 14px; border: 1px solid var(--border);
      padding: 12px 14px; font: inherit; background: #fff;
    }
    input:focus, select:focus {
      outline: 2px solid rgba(37, 99, 235, 0.18); border-color: var(--accent);
    }
    .field { margin-bottom: 16px; }
    button {
      background: var(--accent); color: white; font-weight: 700; border: none; cursor: pointer;
      transition: transform .05s ease, opacity .2s ease;
    }
    button:hover { opacity: .96; }
    button:active { transform: translateY(1px); }
    .result { display: grid; gap: 16px; }
    .big { font-size: 3rem; font-weight: 800; line-height: 1; }
    .pill {
      display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px;
      background: var(--accent2); color: var(--accent); font-weight: 700;
    }
    .statbox {
      background: #fafcff; border: 1px solid var(--border); border-radius: 18px; padding: 16px;
    }
    .muted { color: var(--muted); }
    .error { background: #fff2f2; border: 1px solid #ffd5d5; color: #a11; padding: 14px; border-radius: 14px; }
    .chart { width: 100%; border-radius: 18px; border: 1px solid var(--border); }
    .label-line { font-size: 1.15rem; font-weight: 700; }
    .small { font-size: 0.95rem; }
    .logo-wrap{
      display:flex;
      justify-content:center;
      align-items:center;
      margin-bottom:18px;
    }
    .logo{
      width: 240px;
      height:auto;
      object-fit:contain;
    }
    @media (max-width: 920px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">

    <div class="logo-wrap">
      <img src="https://www.upct.es/contenido/universidad/galeria/identidad-2021/logos/logos-upct/marca-upct/marca-principal/horizontal/azul.png"
           alt="Escudo"
           class="logo">
    </div>

    <div class="hero">
      <h1>Predictor de viajes de entrada al Campus Muralla del Mar</h1>
      <p>Introduce una fecha futura y un tramo horario. La app usa la fecha, sus variables derivadas, el tramo horario y el tipo de día lectivo asociado (`tipo_code`) para devolver una clasificación cualitativa, una estimación numérica con intervalo y una gráfica temporal.</p>
    </div>

    <div class="grid">
      <div class="card">
        <form method="post" action="{{ url_for('predict_route') }}">
          <div class="field">
            <label for="date">Fecha</label>
            <input id="date" name="date" type="date" required value="{{ form_date }}">
          </div>
          <div class="field">
            <label for="slot">Tramo horario</label>
            <select id="slot" name="slot" required>
              {% for s in slots %}
                <option value="{{ s }}" {% if s == form_slot %}selected{% endif %}>{{ s }}</option>
              {% endfor %}
            </select>
          </div>
          <button type="submit">Predecir viajes</button>
        </form>

        <div style="margin-top:18px" class="statbox small">
          <div class="muted">Rango temporal del entrenamiento</div>
          <div><strong>{{ min_date }}</strong> a <strong>{{ max_date }}</strong></div>
          <div style="margin-top:8px" class="muted">Validación rápida del modelo</div>
          <div>
            MAE: <strong>{{ report.get('mae', '—') }}</strong> ·
            RMSE: <strong>{{ report.get('rmse', '—') }}</strong> ·
            R²: <strong>{{ report.get('r2', '—') }}</strong>
          </div>
        </div>
      </div>

      <div class="card result">
        {% if error %}
          <div class="error">{{ error }}</div>
        {% endif %}

        {% if prediction %}
          <div>
            <div class="pill">Predicción cualitativa</div>
            <div class="label-line" style="margin-top:10px">{{ prediction['qualitative'] | upper }} respecto a la media del dia de la semana, tramo y calendario</div>
            <div class="muted">Desviación estimada: {{ prediction['deviation_pct'] | round(1) }}%</div>
            <div class="muted" style="margin-top:6px">Tipo de día asociado: <strong>{{ prediction['type_code'] }}</strong></div>
          </div>

          <div>
            <div class="pill">Estimación central</div>
            <div class="big">{{ prediction['estimate'] | round(2) }}</div>
            <div class="muted">viajes esperados</div>
          </div>

          <div class="statbox">
            <div><strong>Intervalo aproximado</strong></div>
            <div style="margin-top:8px">Entre <strong>{{ prediction['low'] | round(2) }}</strong> y <strong>{{ prediction['high'] | round(2) }}</strong> viajes.</div>
            <div class="muted" style="margin-top:8px">Media histórica del tramo: <strong>{{ prediction['slot_mean'] | round(2) }}</strong></div>
          </div>

          {% if chart_b64 %}
            <div class="statbox">
              <div><strong>Distribución histórica</strong></div>
              <img class="chart" alt="Boxplot de la distribución histórica" src="data:image/png;base64,{{ chart_b64 }}">
            </div>
          {% endif %}

          <div class="statbox muted small">
            La gráfica muestra la distribución histórica de viajes para el mismo tramo horario, el mismo día de la semana y el mismo `tipo_code` siempre que existan datos suficientes; el marcador indica la predicción.
          </div>
        {% else %}
          <div class="statbox muted">Aquí aparecerá el resultado de la predicción.</div>
        {% endif %}
      </div>
    </div>
  </div>
</body>
</html>
"""

mapa_tipo_invertido_es = {
    0: 'fin del curso academico',
    1: 'plazo administrativo',
    2: 'plazo academico',
    3: 'dia no lectivo',
    4: 'festivo',
    5: 'dia lectivo normal',
    6: 'evento universitario',
    7: 'evento academico',
    8: 'inicio de cuatrimestre',
    9: 'fin de cuatrimestre',
    10: 'periodo de examenes'
}

def create_app(csv_path: str, model_dir: str):
    from flask import Flask, render_template_string, request
    model_path = Path(model_dir)
    bundle = load_bundle(model_path)
    if bundle is None:
        raw = load_data(csv_path)
        bundle = train_models(raw, model_path)
    history = load_data(csv_path)

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        slots = bundle.metadata["slot_categories"]
        default_date = pd.to_datetime(bundle.metadata["max_date"]).date().isoformat()
        return render_template_string(
            HTML_TEMPLATE,
            slots=slots,
            form_date=default_date,
            form_slot=slots[0] if slots else "",
            prediction=None,
            error=None,
            chart_b64=None,
            min_date=bundle.metadata["min_date"],
            max_date=bundle.metadata["max_date"],
            report=bundle.metadata.get("validation_report", {}),
        )

    @app.route("/predict", methods=["POST"])
    def predict_route():
        slots = bundle.metadata["slot_categories"]
        target_date = request.form.get("date", "")
        slot = request.form.get("slot", "")
        error = None
        prediction = None
        chart_b64 = None
        try:
            prediction = predict_one(history, bundle, target_date, slot)
            print(prediction['type_code'])
            prediction['type_code']= mapa_tipo_invertido_es[int(prediction['type_code'])]
            chart_b64 = build_boxplot_base64(history, bundle, target_date, slot)
        except Exception as exc:
            error = str(exc)
        #slots = bundle.metadata["type_categories"]
        return render_template_string(
            HTML_TEMPLATE,
            slots=slots,
            form_date=target_date,
            form_slot=slot,
            prediction=prediction,
            error=error,
            chart_b64=chart_b64,
            min_date=bundle.metadata["min_date"],
            max_date=bundle.metadata["max_date"],
            report=bundle.metadata.get("validation_report", {}),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictor de viajes con interfaz web HTML.")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con viajes históricos.")
    parser.add_argument("--model-dir", default="./travel_model_artifacts", help="Directorio para guardar el modelo entrenado.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.csv, args.model_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
