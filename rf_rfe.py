from __future__ import annotations

import argparse
import os
import re
import time
from glob import glob
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

# Глушим шумные future‑варнинги pandas (floor("S") и т.п.)
warnings.filterwarnings("ignore", message=".*is deprecated and will be removed.*")
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------- Константы/утилиты --------------------------------- #

ATTACK_REGEX = re.compile(r"(dos|d[-\s]?dos)", re.IGNORECASE)

# Альтернативные имена колонок в разных выгрузках CIC‑IDS2017
ALT_NAMES = {
    "Timestamp": ["Timestamp", "Flow Timestamp", "Start Time", "StartTime", "Date first seen", "Date First Seen"],
    "Src IP": ["Src IP", "Source IP", "SrcIP", "SourceIP"],
    "Dst IP": ["Dst IP", "Destination IP", "DstIP", "DestinationIP"],
    "Src Port": ["Src Port", "Source Port", "SrcPort"],
    "Dst Port": ["Dst Port", "Destination Port", "DstPort"],
    "Protocol": ["Protocol", "Proto"],
    "Flow Duration": ["Flow Duration", "Flow_Duration"],
    "Tot Fwd Pkts": ["Tot Fwd Pkts", "Total Fwd Packets", "Total Forward Packets"],
    "Tot Bwd Pkts": ["Tot Bwd Pkts", "Total Backward Packets", "Total Bwd Packets"],
    "Flow Bytes/s": ["Flow Bytes/s", "Flow Bytes/s "],
    "Flow Pkts/s": ["Flow Packets/s", "Flow Pkts/s"],
    "Pkt Len Mean": ["Pkt Len Mean", "Packet Length Mean"],
    "Pkt Len Var": ["Pkt Len Var", "Packet Length Variance"],
    "Label": ["Label", "class", "Attack Label"],
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_timestamp(ts: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(ts, errors="coerce", infer_datetime_format=True)
    if parsed.isna().mean() <= 0.5:
        return parsed
    # fallback форматы
    fmts = [
        "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
    ]
    for fmt in fmts:
        parsed = pd.to_datetime(ts, format=fmt, errors="coerce")
        if parsed.isna().mean() <= 0.5:
            break
    return parsed


def shannon_entropy(values: Iterable) -> float:
    s = pd.Series(values)
    if s.empty:
        return 0.0
    p = s.value_counts(dropna=False).astype(float)
    p = p / p.sum()
    return float(-(p * np.log2(np.where(p > 0, p, 1.0))).sum())


def renyi_entropy(values: Iterable, alpha: float = 2.0) -> float:
    s = pd.Series(values)
    if s.empty:
        return 0.0
    p = s.value_counts(dropna=False).astype(float)
    p = p / p.sum()
    if alpha == 1.0:
        return shannon_entropy(values)
    return float((1.0 / (1.0 - alpha)) * np.log2(np.power(p, alpha).sum()))


def normalized_entropy(values: Iterable) -> float:
    s = pd.Series(values)
    if s.empty:
        return 0.0
    h = shannon_entropy(s)
    n_unique = max(s.nunique(dropna=False), 1)
    denom = np.log2(n_unique) if n_unique > 1 else 1.0
    return float(h / denom)


def conditional_entropy(x: Iterable, y: Iterable) -> float:
    xs = pd.Series(x).astype(str)
    ys = pd.Series(y).astype(str)
    if xs.empty or ys.empty:
        return 0.0
    df = pd.DataFrame({"x": xs, "y": ys})
    total = len(df)
    h = 0.0
    for _, grp in df.groupby("y"):
        py = len(grp) / total
        h += py * shannon_entropy(grp["x"])
    return float(h)


def mutual_info(a: Iterable, b: Iterable) -> float:
    sa = pd.Series(a).astype(str)
    sb = pd.Series(b).astype(str)
    if sa.empty or sb.empty:
        return 0.0
    return float(mutual_info_score(sa, sb))


def interarrival_entropy(timestamps: pd.Series) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    ts = pd.Series(timestamps).sort_values().view("int64")  # ns
    diffs = np.diff(ts) / 1e9
    diffs_q = np.round(diffs, 3)
    return shannon_entropy(diffs_q)


def safe_mean(x: Iterable) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(np.nanmean(s)) if len(s) else 0.0


def safe_std(x: Iterable) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(np.nanstd(s)) if len(s) else 0.0


def ratio_unique(series: Iterable, total: int) -> float:
    s = pd.Series(series)
    if total <= 0:
        return 0.0
    return float(s.nunique(dropna=False) / total)


# --------------------------------- Загрузка / нормализация --------------------------------- #

def discover_csvs(data_dir: str, patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(sorted(glob(os.path.join(data_dir, pat))))
    return out


def load_flows(csv_paths: List[str], limit_rows: int | None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in tqdm(csv_paths, desc="Загрузка CSV", unit="файл"):
        try:
            df = pd.read_csv(p, low_memory=False, nrows=limit_rows, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, nrows=limit_rows, encoding="latin-1")

        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
        df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=True).str.strip()
        frames.append(df)

    if not frames:
        raise RuntimeError("Не найдено подходящих CSV. Проверь --data-dir и --glob.")

    df = pd.concat(frames, ignore_index=True)

    cols_lc = {c.lower(): c for c in df.columns}

    def find(*names: str) -> str | None:
        for n in names:
            c = cols_lc.get(n.lower())
            if c:
                return c
        return None

    ts_col   = find(*ALT_NAMES["Timestamp"])
    src_ip   = find(*ALT_NAMES["Src IP"])
    dst_ip   = find(*ALT_NAMES["Dst IP"])
    src_prt  = find(*ALT_NAMES["Src Port"])
    dst_prt  = find(*ALT_NAMES["Dst Port"])
    proto    = find(*ALT_NAMES["Protocol"])
    flow_dur = find(*ALT_NAMES["Flow Duration"])
    fwd_pkts = find(*ALT_NAMES["Tot Fwd Pkts"])
    bwd_pkts = find(*ALT_NAMES["Tot Bwd Pkts"])
    flow_bs  = find(*ALT_NAMES["Flow Bytes/s"])
    flow_ps  = find(*ALT_NAMES["Flow Pkts/s"])
    pkt_mean = find(*ALT_NAMES["Pkt Len Mean"])
    pkt_var  = find(*ALT_NAMES["Pkt Len Var"])
    label    = find(*ALT_NAMES["Label"])

    if ts_col:
        df["Timestamp"] = parse_timestamp(df[ts_col])
    else:
        if not flow_dur:
            raise ValueError("[ERROR] Нет ни Timestamp, ни Flow Duration — невозможно построить окна.")
        dur_sec = pd.to_numeric(df[flow_dur], errors="coerce").fillna(0).clip(lower=0) / 1_000_000.0
        base = pd.Timestamp("2017-07-03 09:00:00")
        df["Timestamp"] = base + pd.to_timedelta(dur_sec.cumsum(), unit="s")

    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    rename = {}
    if src_ip:   rename[src_ip]   = "Src IP"
    if dst_ip:   rename[dst_ip]   = "Dst IP"
    if src_prt:  rename[src_prt]  = "Src Port"
    if dst_prt:  rename[dst_prt]  = "Dst Port"
    if proto:    rename[proto]    = "Protocol"
    if flow_dur: rename[flow_dur] = "Flow Duration"
    if fwd_pkts: rename[fwd_pkts] = "Tot Fwd Pkts"
    if bwd_pkts: rename[bwd_pkts] = "Tot Bwd Pkts"
    if flow_bs:  rename[flow_bs]  = "Flow Bytes/s"
    if flow_ps:  rename[flow_ps]  = "Flow Pkts/s"
    if pkt_mean: rename[pkt_mean] = "Pkt Len Mean"
    if pkt_var:  rename[pkt_var]  = "Pkt Len Var"
    if label:    rename[label]    = "Label"
    if rename:
        df = df.rename(columns=rename)

    if "Label" in df.columns:
        df["is_attack_flow"] = df["Label"].astype(str).str.contains(ATTACK_REGEX).astype(int)
    else:
        df["is_attack_flow"] = 0

    return df.reset_index(drop=True)


# --------------------------------- Окна и признаки --------------------------------- #

def window_iter(df: pd.DataFrame, window_sec: int):
    start = df["Timestamp"].min().floor(f"{window_sec}S")
    end = df["Timestamp"].max().ceil(f"{window_sec}S")
    cur = start
    dt = pd.Timedelta(seconds=window_sec)
    while cur < end:
        nxt = cur + dt
        mask = (df["Timestamp"] >= cur) & (df["Timestamp"] < nxt)
        yield cur, df.loc[mask]
        cur = nxt


def build_http_cols_mapping(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    def has(name: str) -> str | None: return cols.get(name.lower())
    return {
        "url":             has("url"),
        "query_params":    has("query_params"),
        "user_agent":      has("user_agent"),
        "referer":         has("referer"),
        "accept_language": has("accept_language"),
        "cookies":         has("cookies"),
        "content_type":    has("content_type"),
        "method":          has("method"),
        "response_code":   has("response_code"),
        "content_length":  has("content_length"),
        "session_id":      has("session_id"),
        "headers":         has("headers"),
    }


def compute_http_like_features(g: pd.DataFrame, http_cols: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    def col(name: str) -> pd.Series:
        c = http_cols.get(name)
        return g[c] if (c and c in g.columns) else pd.Series(dtype=object)

    out["entropy_url_paths"]        = shannon_entropy(col("url"))
    out["entropy_query_params"]     = shannon_entropy(col("query_params"))
    out["entropy_user_agents"]      = shannon_entropy(col("user_agent"))
    out["entropy_referers"]         = shannon_entropy(col("referer"))
    out["entropy_accept_languages"] = shannon_entropy(col("accept_language"))
    out["entropy_cookies"]          = shannon_entropy(col("cookies"))
    out["entropy_content_types"]    = shannon_entropy(col("content_type"))
    out["entropy_methods"]          = shannon_entropy(col("method"))
    out["entropy_response_codes"]   = shannon_entropy(col("response_code"))
    out["entropy_content_lengths"]  = shannon_entropy(col("content_length"))

    out["normalized_entropy_urls"]     = normalized_entropy(col("url"))
    out["normalized_entropy_ips"]      = normalized_entropy(g.get("Src IP"))
    out["normalized_entropy_sessions"] = normalized_entropy(col("session_id"))
    out["normalized_entropy_headers"]  = normalized_entropy(col("headers"))

    out["conditional_entropy_url_given_ip"]     = conditional_entropy(col("url"), g.get("Src IP"))
    out["conditional_entropy_method_given_url"] = conditional_entropy(col("method"), col("url"))
    out["conditional_entropy_response_given_request"] = conditional_entropy(col("response_code"), col("method"))

    ts_floor_s = pd.to_datetime(g.get("Timestamp")).dt.floor("s")
    out["mutual_info_ip_url"]        = mutual_info(g.get("Src IP"), col("url"))
    out["mutual_info_method_response"]= mutual_info(col("method"), col("response_code"))
    out["mutual_info_time_url"]      = mutual_info(ts_floor_s, col("url"))

    return out


def compute_behavioral_temporal_stat_features(g: pd.DataFrame, window_sec: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(g)

    # Поведенческие
    out["request_rate"]      = float(n) / max(window_sec, 1)
    out["unique_ip_ratio"]   = ratio_unique(g.get("Src IP"), n)
    out["session_diversity"] = ratio_unique(g.get("Src IP").astype(str) + "→" + g.get("Dst IP").astype(str), n)
    out["error_rate"]        = 0.0  # без HTTP status-кодов
    out["avg_response_time"] = safe_mean(g.get("Flow Duration")) / 1000.0  # мс
    out["std_response_time"] = safe_std(g.get("Flow Duration")) / 1000.0
    out["get_post_ratio"]    = 0.0  # нет методов — 0
    key = g.get("Src IP").astype(str) + "|" + g.get("Dst IP").astype(str) + "|" + g.get("Dst Port").astype(str)
    out["repeat_request_ratio"] = 0.0 if n == 0 else float((key.value_counts().max() if not key.empty else 0) / n)

    # Временные
    out["request_interval_entropy"] = interarrival_entropy(g.get("Timestamp"))
    if n > 0:
        per_sec = g["Timestamp"].dt.floor("s").value_counts().astype(float)
        out["burst_intensity"] = float(per_sec.max() / max(per_sec.mean(), 1.0))
    else:
        out["burst_intensity"] = 0.0
    if n >= 3:
        ts = g["Timestamp"].sort_values().view("int64")
        diffs = np.diff(ts) / 1e9
        cv = (np.std(diffs) / max(np.mean(diffs), 1e-9)) if diffs.size > 0 else 0.0
        out["temporal_regularity"] = float(1.0 / (1.0 + cv))
    else:
        out["temporal_regularity"] = 0.0

    # Статистические
    out["traffic_volume"]        = float(safe_mean(g.get("Flow Bytes/s"))) * window_sec
    out["packet_size_variance"]  = safe_mean(g.get("Pkt Len Var"))
    out["connection_duration"]   = safe_mean(g.get("Flow Duration")) / 1000.0

    # Сетевые энтропии
    out["entropy_src_ip"]        = shannon_entropy(g.get("Src IP"))
    out["entropy_dst_ip"]        = shannon_entropy(g.get("Dst IP"))
    out["entropy_dst_port"]      = shannon_entropy(g.get("Dst Port"))
    out["entropy_protocol"]      = shannon_entropy(g.get("Protocol"))
    out["norm_entropy_src_ip"]   = normalized_entropy(g.get("Src IP"))
    out["norm_entropy_dst_ip"]   = normalized_entropy(g.get("Dst IP"))
    out["norm_entropy_dst_port"] = normalized_entropy(g.get("Dst Port"))
    out["renyi_entropy_src_ip"]  = renyi_entropy(g.get("Src IP"))
    out["renyi_entropy_dst_ip"]  = renyi_entropy(g.get("Dst IP"))

    # Взаимная информация (сетевая)
    out["mi_srcip_dstip"]    = mutual_info(g.get("Src IP"), g.get("Dst IP"))
    out["mi_srcip_dstport"]  = mutual_info(g.get("Src IP"), g.get("Dst Port"))

    # Агрегаты по потокам
    out["mean_flow_bytes_s"] = safe_mean(g.get("Flow Bytes/s"))
    out["std_flow_bytes_s"]  = safe_std(g.get("Flow Bytes/s"))
    out["mean_flow_pkts_s"]  = safe_mean(g.get("Flow Pkts/s"))
    out["std_flow_pkts_s"]   = safe_std(g.get("Flow Pkts/s"))
    out["mean_pkt_len"]      = safe_mean(g.get("Pkt Len Mean"))
    out["std_pkt_len"]       = safe_std(g.get("Pkt Len Mean"))
    out["pkt_len_var_mean"]  = safe_mean(g.get("Pkt Len Var"))
    out["mean_tot_fwd_pkts"] = safe_mean(g.get("Tot Fwd Pkts"))
    out["mean_tot_bwd_pkts"] = safe_mean(g.get("Tot Bwd Pkts"))

    # Служебные для разметки окна
    out["_attack_flows"] = int(g.get("is_attack_flow", pd.Series(dtype=int)).sum())
    out["_total_flows"]  = int(n)
    return out


def compute_window_features(g: pd.DataFrame, window_sec: int, http_cols: Dict[str, str]) -> Dict[str, float]:
    f_http = compute_http_like_features(g, http_cols)
    f_core = compute_behavioral_temporal_stat_features(g, window_sec)
    f_http.update(f_core)
    return f_http


def build_dataset(
    df: pd.DataFrame,
    window_sec: int,
    attack_ratio_threshold: float,
    http_cols: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.Series]:
    rows: List[Tuple[pd.Timestamp, Dict[str, float], int]] = []
    total_windows = int(
        np.ceil(
            (df["Timestamp"].max().ceil(f"{window_sec}S") - df["Timestamp"].min().floor(f"{window_sec}S")
        ).total_seconds() / window_sec)
    )
    for w_start, g in tqdm(window_iter(df, window_sec), total=total_windows, desc="Агрегация по окнам", unit="окно"):
        feats = compute_window_features(g, window_sec, http_cols)
        if feats["_total_flows"] == 0:
            continue
        attack_ratio = feats["_attack_flows"] / max(feats["_total_flows"], 1)
        y = 1 if attack_ratio >= attack_ratio_threshold else 0
        del feats["_attack_flows"]; del feats["_total_flows"]
        rows.append((w_start, feats, y))

    if not rows:
        raise RuntimeError("Не удалось сформировать окна: пустые данные?")

    idx = [r[0] for r in rows]
    X = pd.DataFrame([r[1] for r in rows], index=pd.Index(idx, name="window_start")).sort_index()
    y = pd.Series([r[2] for r in rows], index=X.index, name="label")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-1e12, upper=1e12)
    return X, y


# --------------------------------- RFE / RF / Метрики --------------------------------- #

def time_based_split(X: pd.DataFrame, y: pd.Series, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * (1.0 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_rf_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int,
    random_state: int = 42,
    calibrate: bool = True,
) -> Bunch:
    base_rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, class_weight="balanced", random_state=random_state,
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_train.values.astype(float))

    rfe = RFE(estimator=base_rf, n_features_to_select=n_features, step=1)
    tqdm.write("[INFO] Запуск RFE (RandomForest -> отбор признаков)...")
    rfe.fit(X_scaled, y_train.values.astype(int))

    support_mask = rfe.support_
    selected_features = X_train.columns[support_mask].tolist()

    final_rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, class_weight="balanced", random_state=random_state,
    )
    X_sel = X_train[selected_features].values.astype(float)
    X_sel = scaler.fit_transform(X_sel)
    if calibrate:
        clf = CalibratedClassifierCV(final_rf, method="sigmoid", cv=3)
        clf.fit(X_sel, y_train.values.astype(int))
        model = clf
    else:
        final_rf.fit(X_sel, y_train.values.astype(int))
        model = final_rf

    return Bunch(
        scaler=scaler,
        rfe=rfe,
        model=model,
        selected_features=selected_features,
        support_mask=support_mask,
    )


def evaluate(model: Bunch, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float | list]:
    X_sel = X_test[model.selected_features].values.astype(float)
    X_sel = model.scaler.transform(X_sel)
    proba = model.model.predict_proba(X_sel)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def benchmark_latency(model: Bunch, X_test: pd.DataFrame, n_runs: int = 5) -> float:
    X_sel = X_test[model.selected_features].values.astype(float)
    X_sel = model.scaler.transform(X_sel)
    _ = model.model.predict_proba(X_sel[:100])  # прогрев
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.model.predict_proba(X_sel)
        dt = time.perf_counter() - t0
        times.append((dt / max(len(X_sel), 1)) * 1000.0)
    return float(np.percentile(times, 95)) if times else 0.0


def save_artifacts(model: Bunch, X: pd.DataFrame, metrics: Dict[str, float | list], out_dir: str) -> None:
    ensure_dir(out_dir)
    joblib.dump(
        {
            "scaler": model.scaler,
            "rfe_support": model.support_mask,
            "selected_features": model.selected_features,
            "clf": model.model,
        },
        os.path.join(out_dir, "rf_rfe_cicids2017.joblib"),
    )
    pd.Series(model.selected_features, name="selected_features").to_csv(
        os.path.join(out_dir, "selected_features.csv"), index=False
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    pd.Series(X.columns, name="feature_schema").to_csv(
        os.path.join(out_dir, "feature_schema.csv"), index=False
    )

def main():
    ap = argparse.ArgumentParser(description="RF‑RFE для DoS/DDoS на CIC‑IDS2017 (flow CSV)")
    ap.add_argument("--data-dir", required=True, help="Папка с CSV")
    ap.add_argument("--glob", nargs="+", default=["*.csv"], help="Маски файлов (можно несколько)")
    ap.add_argument("--limit-rows", type=int, default=None, help="Ограничение строк/файл (для быстрого прогона)")
    ap.add_argument("--window", type=int, default=10, help="Размер окна, сек")
    ap.add_argument("--attack-threshold", type=float, default=0.2, help="Доля атакующих flows в окне для метки=1")
    ap.add_argument("--test-size", type=float, default=0.3, help="Доля теста (time‑based split)")
    ap.add_argument("--features", type=int, default=8, help="Сколько фич оставить после RFE")
    ap.add_argument("--no-calibration", action="store_true", help="Отключить калибровку вероятностей")
    ap.add_argument("--output", default="./artifacts_rf_rfe", help="Папка для артефактов")
    args = ap.parse_args()

    ensure_dir(args.output)

    csvs = discover_csvs(args.data_dir, args.glob)
    if not csvs:
        raise SystemExit("CSV не найдены. Проверь путь и маски.")
    tqdm.write(f"[INFO] Файлов CSV: {len(csvs)}")
    df = load_flows(csvs, limit_rows=args.limit_rows)
    tqdm.write(f"[INFO] Загружено потоков: {len(df):,}")
    tqdm.write(f"[INFO] Диапазон времени: {df['Timestamp'].min()} .. {df['Timestamp'].max()}")

    http_cols = build_http_cols_mapping(df)
    if all(v is None for v in http_cols.values()):
        tqdm.write("[INFO] HTTP‑колонки не найдены (ожидаемо для CIC‑flows). L7‑энтропии будут 0.")

    tqdm.write(f"[INFO] Формируем окна: {args.window}s, порог атаки: {args.attack_threshold}")
    X, y = build_dataset(df, window_sec=args.window, attack_ratio_threshold=args.attack_threshold, http_cols=http_cols)

    X.to_csv(os.path.join(args.output, "window_features.csv"))
    y.to_csv(os.path.join(args.output, "window_labels.csv"))
    tqdm.write(f"[INFO] Получено окон: {len(X):,}. Доля атакующих окон: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = time_based_split(X, y, test_size=args.test_size)
    tqdm.write(f"[INFO] Train: {len(X_train):,} окон,  Test: {len(X_test):,} окон")

    model = train_rf_rfe(
        X_train, y_train, n_features=args.features, random_state=42, calibrate=not args.no_calibration,
    )
    tqdm.write("[INFO] Выбранные RFE признаки:")
    for i, f in enumerate(model.selected_features, 1):
        tqdm.write(f"  {i:2d}. {f}")

    metrics = evaluate(model, X_test, y_test)
    p95_ms = benchmark_latency(model, X_test)
    metrics_out = dict(metrics)
    metrics_out["p95_ms"] = p95_ms

    tqdm.write("\n=== Метрики (Test) ===")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        tqdm.write(f"{k:>10s}: {metrics.get(k):.4f}")
    tqdm.write(f"confusion_matrix: {metrics['confusion_matrix']}")
    tqdm.write(f"p95_inference_per_window: {p95_ms:.3f} ms")

    save_artifacts(model, X, metrics_out, args.output)
    tqdm.write(f"\n[OK] Артефакты сохранены в: {os.path.abspath(args.output)}")
    tqdm.write(" - rf_rfe_cicids2017.joblib")
    tqdm.write(" - selected_features.csv")
    tqdm.write(" - metrics.csv")
    tqdm.write(" - feature_schema.csv")
    tqdm.write(" - window_features.csv / window_labels.csv")


if __name__ == "__main__":
    main()
