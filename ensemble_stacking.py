from __future__ import annotations

import os
import sys
import gc
import json
import glob
import math
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# ----------------------------- Конфиг -----------------------------

DATA_DIR_FLOWS = "MachineLearningCVE"
DATA_DIR_LABELS = "TrafficLabelling"
ARTIFACTS_DIR = "artifacts_ensemble"


DEFAULT_SAMPLE_PER_FILE = int(os.environ.get("SAMPLE_PER_FILE", 400_000))
DEFAULT_MAX_PER_CLASS   = int(os.environ.get("MAX_PER_CLASS", 300_000))
RANDOM_STATE = 42
TEST_SIZE    = 0.3

REQUIRED_BASE_COLS = [
    "timestamp", "source ip", "destination ip", "source port", "destination port", "protocol"
]


CICIDS_REMAP = {
    "flow id": "flow id",
    "source ip": "source ip",
    "destination ip": "destination ip",
    "source port": "source port",
    "destination port": "destination port",
    "protocol": "protocol",
    "timestamp": "timestamp",
    "label": "label",
}

# --------------------------- Утилиты I/O --------------------------

console = Console()
warnings.filterwarnings("ignore", category=UserWarning)

def log_header():
    console.rule("Ансамблевый стекинг — запуск", style="cyan")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_csvs(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.csv")))

def try_read_csv(path: str, usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                on_bad_lines="skip",
                low_memory=False,
                dtype=None,
                engine="c",
            )
            return df
        except Exception as e:
            continue
    console.print(f"[yellow]WARNING[/] ️  {os.path.basename(path)} не прочитан; пропускаю.")
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    renames = {}
    for col in list(df.columns):
        c = col
        if c == "flow id" or c.endswith("flow id"):
            renames[col] = "flow id"
        elif c in ("src ip", "src_ip", "source address", "sourceip", " source ip"):
            renames[col] = "source ip"
        elif c in ("dst ip", "dst_ip", "destination address", " destination ip"):
            renames[col] = "destination ip"
        elif c in ("src port", "srcport", " source port"):
            renames[col] = "source port"
        elif c in ("dst port", "dstport", " destination port"):
            renames[col] = "destination port"
        elif c in ("protocol", " proto"):
            renames[col] = "protocol"
        elif c in ("timestamp", " flow start", "starttime", " time stamp", "time", "datetime"):
            renames[col] = "timestamp"
        elif c.strip() == "label":
            renames[col] = "label"
    if renames:
        df = df.rename(columns=renames)
    return df

def labelize(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        return df
    df["label"] = df["label"].astype(str).str.upper()
    df["y"] = np.where(df["label"].str.contains("BENIGN"), 0, 1)
    return df

def safe_parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def attach_labels_if_missing(df_flow: pd.DataFrame, label_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_flow is None or df_flow.empty:
        return df_flow
    if "label" in df_flow.columns:
        return df_flow

    if label_df is None or label_df.empty:
        return df_flow

    # нормализация
    df_flow = normalize_columns(df_flow)
    label_df = normalize_columns(label_df)
    df_flow = safe_parse_timestamp(df_flow)
    label_df = safe_parse_timestamp(label_df)

    # 1) по flow id
    if "flow id" in df_flow.columns and "flow id" in label_df.columns and "label" in label_df.columns:
        merged = df_flow.merge(label_df[["flow id", "label"]].drop_duplicates(), on="flow id", how="left")
        if "label" in merged.columns and merged["label"].notna().any():
            return merged

    # 2) по timestamp (одноимённый файл)
    if "timestamp" in df_flow.columns and "timestamp" in label_df.columns and "label" in label_df.columns:
        lf = label_df[["timestamp", "label"]].copy()
        lf["ts_round"] = lf["timestamp"].dt.floor("S")
        ff = df_flow.copy()
        ff["ts_round"] = ff["timestamp"].dt.floor("S")
        merged = ff.merge(lf.groupby("ts_round")["label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                          left_on="ts_round", right_index=True, how="left")
        if "label" in merged.columns and merged["label"].notna().any():
            merged = merged.drop(columns=["ts_round"])
            return merged

    return df_flow

# ------------------------ Загрузка & сборка -----------------------

def load_matching_label_file(flow_path: str, all_label_files: List[str]) -> Optional[str]:
    base = os.path.basename(flow_path)
    day = base.split(".pcap")[0]
    candidates = [p for p in all_label_files if os.path.basename(p).startswith(day)]
    return candidates[0] if candidates else None

def load_dataset(sample_per_file: int, max_per_class: int) -> pd.DataFrame:
    flow_files  = list_csvs(DATA_DIR_FLOWS)
    label_files = list_csvs(DATA_DIR_LABELS)

    if not flow_files:
        console.print("[red]Файлы в MachineLearningCVE не найдены.[/]")
        sys.exit(1)

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Загрузка CSV[/]"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as prog:
        task = prog.add_task("load", total=len(flow_files))

        chunks: List[pd.DataFrame] = []

        for fp in flow_files:
            prog.update(task, advance=1)
            df = try_read_csv(fp)
            if df is None or df.empty:
                continue
            df = normalize_columns(df)
            df = safe_parse_timestamp(df)

            if "label" not in df.columns:
                lbl_path = load_matching_label_file(fp, label_files)
                lbl_df = try_read_csv(lbl_path) if lbl_path else None
                if lbl_df is not None:
                    df = attach_labels_if_missing(df, lbl_df)

            if "label" not in df.columns:
                console.print(f"[yellow]WARNING[/]  {os.path.basename(fp)} без 'Label' — пропускаю.")
                continue

            if sample_per_file and len(df) > sample_per_file:
                df = df.sample(sample_per_file, random_state=RANDOM_STATE)

            chunks.append(df)

        if not chunks:
            console.print("[red]В объединённых данных нет 'Label' — нечего обучать.[/]")
            sys.exit(1)

    data = pd.concat(chunks, ignore_index=True)
    data = labelize(data)

    data = data.loc[data["label"].notna() & data["y"].isin([0, 1])]
    data = data.reset_index(drop=True)

    if max_per_class:
        parts = []
        for cls in (0, 1):
            part = data.loc[data["y"] == cls]
            if len(part) > max_per_class:
                part = part.sample(max_per_class, random_state=RANDOM_STATE)
            parts.append(part)
        data = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return data

# ----------------------------- Фичи -------------------------------

def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    drop_like = {"label", "y", "timestamp", "flow id", "source ip", "destination ip"}
    cols_num = []
    for c in df.columns:
        if c in drop_like:
            continue
        if any(x in c for x in ["ip", "mac", "address", "flow id", "timestamp"]):
            continue
        # числовые
        if pd.api.types.is_numeric_dtype(df[c]):
            cols_num.append(c)

    X = df[cols_num].copy()
    y = df["y"].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if X.shape[1] == 0:
        fallback = [c for c in df.columns if c.lower() in {
            "flow duration", "tot fwd pkts", "tot bwd pkts", "totlen fwd pkts",
            "totlen bwd pkts", "fwd pkt len mean", "bwd pkt len mean",
            "flow pkts/s", "flow bytes/s", "pkt len mean", "pkt len std",
            "down/up ratio", "subflow fwd pkts", "subflow bwd pkts"
        } and pd.api.types.is_numeric_dtype(df[c])]
        if fallback:
            X = df[fallback].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            raise RuntimeError("Не найдены числовые признаки для обучения.")

    return X, y

# --------------------------- Модель/оценка ------------------------

def build_model() -> StackingClassifier:
    base_estimators = [
        ("rf",  RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced_subsample"
        )),
        ("et",  ExtraTreesClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced_subsample"
        )),
        ("gb",  GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )),
    ]
    meta = LogisticRegression(max_iter=200, n_jobs=-1, random_state=RANDOM_STATE)
    clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        passthrough=False,
        stack_method="auto",
        n_jobs=-1
    )
    return clf

def evaluate_and_save(y_true, y_proba, y_pred, out_dir: str, threshold: float = 0.5):
    ensure_dir(out_dir)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_proba)
    prc = average_precision_score(y_true, y_proba)
    cm  = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc, "pr_auc": prc,
        "threshold": threshold,
        "positives": int(np.sum(y_true)),
        "total": int(len(y_true))
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    table = Table(title="=== Метрики (Test) ===")
    for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc","threshold","positives","total"]:
        table.add_row(k, f"{metrics[k]:.4f}" if isinstance(metrics[k], float) else str(metrics[k]))
    console.print(table)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["BENIGN","ATTACK"], yticklabels=["BENIGN","ATTACK"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.plot([0,1],[0,1], "--", color="orange")
    plt.title("ROC")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6,5))
    plt.hist(y_proba[y_true==0], bins=50, alpha=0.8, label="BENIGN")
    plt.hist(y_proba[y_true==1], bins=50, alpha=0.6, label="ATTACK")
    plt.legend()
    plt.title("Score distribution")
    plt.xlabel("Predicted probability (ATTACK)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_hist.png"), dpi=160)
    plt.close()

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble Stacking for CICIDS2017")
    parser.add_argument("--sample-per-file", type=int, default=DEFAULT_SAMPLE_PER_FILE, help="Максимум строк с каждого CSV (0 = все)")
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS, help="Потолок на класс после объединения (0 = без лимита)")
    parser.add_argument("--artifacts-dir", type=str, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    log_header()
    ensure_dir(args.artifacts_dir)

    data = load_dataset(sample_per_file=args.sample_per_file, max_per_class=args.max_per_class)
    console.print(f"[green]INFO[/]  Загружено строк: {len(data):,}".replace(",", " "))
    attack_share = 100.0 * data["y"].mean()
    console.print(f"[green]INFO[/]     Доля атак: {attack_share:.2f}%")

    X, y = select_features(data)
    console.print(f"[green]INFO[/]     Признаков: {X.shape[1]}, наблюдений: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    console.print(f"[green]INFO[/]     Train: {len(X_train):,},  Test: {len(X_test):,}".replace(",", " "))

    clf = build_model()

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Обучение ансамбля[/]"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as prog:
        task = prog.add_task("fit", total=1)
        clf.fit(X_train, y_train)
        prog.update(task, advance=1)

    t0 = time.time()
    y_proba = clf.predict_proba(X_test)[:, 1]
    infer_ms = (time.time() - t0) * 1000.0 / len(X_test)
    y_pred  = (y_proba >= 0.5).astype(int)

    evaluate_and_save(y_test, y_proba, y_pred, args.artifacts_dir, threshold=0.5)
    console.print(f"[green]INFO[/]     p95_inference_per_sample: {infer_ms:.3f} ms")

    import joblib
    joblib.dump({"model": clf, "feature_names": list(X.columns)}, os.path.join(args.artifacts_dir, "ensemble_stacking.joblib"))
    pd.DataFrame({"feature": X.columns}).to_csv(os.path.join(args.artifacts_dir, "feature_schema.csv"), index=False)

    console.print(f"\n[green][OK][/] Артефакты сохранены в: {os.path.abspath(args.artifacts_dir)}")
    console.print(f" - ensemble_stacking.joblib")
    console.print(f" - metrics.csv / metrics.json")
    console.print(f" - feature_schema.csv")
    console.print(f" - confusion_matrix.png / roc.png / pr.png / score_hist.png\n")

if __name__ == "__main__":
    main()
