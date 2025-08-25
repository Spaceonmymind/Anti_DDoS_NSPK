# post_eval_full.py
# -*- coding: utf-8 -*-
"""
Пост‑оценка RF‑RFE модели для DoS/DDoS (CIC‑IDS2017):
- загрузка обученных артефактов (rf_rfe.py)
- расчёт вероятностей, сдвиг порога, метрики, sweep по порогам
- автоподбор порога по F‑β (с ограничением min precision)
- ROC / PR / Confusion Matrix / распределение вероятностей
- (опц.) полная IP‑аналитика по исходным CSV: топ Src IP, пары Src→Dst, топ Dst‑портов
- тепловые карты (src→dst), (src→dst_port) по топ‑N
- формирование HTML‑отчёта со сводкой и ссылками на артефакты

Зависимости: numpy, pandas, joblib, matplotlib, tqdm, scikit‑learn
"""

from __future__ import annotations

import argparse
import os
import json
from glob import glob
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import warnings

import matplotlib
matplotlib.use("Agg")  # headless-рендеринг
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

warnings.filterwarnings("ignore", message=".*is deprecated and will be removed.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ========================= УТИЛИТЫ I/O =========================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_csv_smart(path: str, **kw) -> pd.DataFrame:
    """
    Маленькие служебные CSV читаем с fallback по кодировкам.
    """
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        for enc in ("utf-8-sig", "cp1251", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, **kw)
            except UnicodeDecodeError:
                continue
        # последняя броня — съедаем битые символы
        try:
            return pd.read_csv(path, encoding="utf-8", encoding_errors="ignore", **kw)
        except TypeError:
            return pd.read_csv(path, encoding="latin-1", **kw)


def iter_csv_chunks(
    path: str,
    chunksize: int,
    encoding: str = "auto",
    on_bad_lines: str = "skip",
) -> Iterable[pd.DataFrame]:
    """
    Надёжный итератор по чанкам большого CSV.

    Поведение:
      - Если `encoding` != "auto" — используем ровно эту кодировку.
      - Иначе пробуем по очереди: utf-8, utf-8-sig, cp1251, cp1252, latin-1.
      - Валидируем выбранную кодировку на первом `next(reader)`.
      - Пропускаем битые строки (`on_bad_lines='skip'`).
      - Финальный fallback — `encoding_errors='ignore'` (если доступно в pandas).

    Возвращает генератор DataFrame-чанков.
    """
    def _yield_all(_reader):
        for _chunk in _reader:
            yield _chunk

    # Явно заданная кодировка
    if encoding and encoding.lower() != "auto":
        try:
            reader = pd.read_csv(
                path,
                low_memory=False,
                chunksize=chunksize,
                encoding=encoding,
                on_bad_lines=on_bad_lines,
            )
            first = next(reader)  # валидируем кодировку здесь
            yield first
            yield from _yield_all(reader)
            return
        except StopIteration:
            return
        except UnicodeDecodeError:
            # если пользователь указал неверно — упадём ровно как есть
            raise

    # Авто‑перебор
    candidates = ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1")
    for enc in candidates:
        try:
            reader = pd.read_csv(
                path,
                low_memory=False,
                chunksize=chunksize,
                encoding=enc,
                on_bad_lines=on_bad_lines,
            )
            first = next(reader)  # здесь и проявляется неверная кодировка
            yield first
            yield from _yield_all(reader)
            return
        except UnicodeDecodeError:
            continue
        except StopIteration:
            return

    # Бронебойный вариант: съедаем битые байты
    try:
        reader = pd.read_csv(
            path,
            low_memory=False,
            chunksize=chunksize,
            encoding="utf-8",
            encoding_errors="ignore",
            on_bad_lines=on_bad_lines,
        )
    except TypeError:
        reader = pd.read_csv(
            path,
            low_memory=False,
            chunksize=chunksize,
            encoding="latin-1",
            on_bad_lines=on_bad_lines,
        )
    yield from _yield_all(reader)


# ========================= ЗАГРУЗКА АРТЕФАКТОВ =========================

def load_artifacts(art_dir: str):
    model_path = os.path.join(art_dir, "rf_rfe_cicids2017.joblib")
    if not os.path.exists(model_path):
        raise SystemExit(f"[ERROR] Не найден {model_path}. Сначала запусти rf_rfe.py.")
    bundle = joblib.load(model_path)

    X = read_csv_smart(os.path.join(art_dir, "window_features.csv"), index_col=0)
    # индекс окна – это строка с датой → приведём к datetime (мягко)
    try:
        X.index = pd.to_datetime(X.index, errors="coerce", infer_datetime_format=True)
    except Exception:
        pass

    y_path = os.path.join(art_dir, "window_labels.csv")
    y = read_csv_smart(y_path, index_col=0)
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    y.index = X.index
    return bundle, X, y


def predict_proba(bundle, X: pd.DataFrame) -> np.ndarray:
    feats = bundle["selected_features"]
    scaler = bundle["scaler"]
    clf = bundle["clf"]
    X_sel = X[feats].values.astype(float)
    X_sel = scaler.transform(X_sel)
    return clf.predict_proba(X_sel)[:, 1]


# ========================= МЕТРИКИ / ПОРОГ =========================

def metrics_at_threshold(y_true, proba, thr: float) -> Dict:
    pred = (proba >= thr).astype(int)
    out = dict(
        threshold=float(thr),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        pr_auc=float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        confusion_matrix=confusion_matrix(y_true, pred).tolist(),
        positives=int(pred.sum()),
        total=int(len(pred)),
    )
    return out


def sweep_thresholds(y_true, proba, grid: List[float]) -> pd.DataFrame:
    rows = []
    for thr in grid:
        rows.append(metrics_at_threshold(y_true, proba, thr))
    return pd.DataFrame(rows)


def auto_threshold(
    y_true,
    proba,
    beta: float = 2.0,
    min_precision: float = 0.8,
    grid: Optional[List[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Подбирает порог, максимизируя F‑β среди точек, где precision >= min_precision.
    Возвращает (best_thr, sweep_df).
    """
    if grid is None:
        grid = list(np.round(np.linspace(0.05, 0.95, 19), 2))
    rows = []
    best_thr, best_fb = None, -1.0
    for thr in grid:
        pred = (proba >= thr).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if prec >= min_precision:
            beta2 = beta * beta
            denom = beta2 * prec + rec
            fb = (1 + beta2) * prec * rec / denom if denom > 0 else 0.0
        else:
            fb = -1.0  # вне ограничений
        rows.append(
            dict(
                threshold=thr,
                precision=prec,
                recall=rec,
                f1=f1_score(y_true, pred, zero_division=0),
                fbeta=fb,
            )
        )
        if fb > best_fb:
            best_fb, best_thr = fb, thr
    return (best_thr if best_thr is not None else 0.5), pd.DataFrame(rows)


# ========================= ГРАФИКИ =========================

def plot_roc(y_true, proba, outpath: str):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], ls="--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR (Recall)")
    plt.title("ROC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_pr(y_true, proba, outpath: str, mark_thr: float | None = None, pred=None):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(True, alpha=0.3)
    if mark_thr is not None:
        if pred is None:
            pred = (proba >= mark_thr).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        plt.scatter([r], [p], s=60, marker="o")
        plt.text(r, p, f"  thr={mark_thr:.2f}\n  P={p:.2f}, R={r:.2f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_confmat(cm: np.ndarray, outpath: str, labels=("BENIGN", "ATTACK")):
    plt.figure(figsize=(4.5, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_score_hist(proba: np.ndarray, y_true: np.ndarray, outpath: str, bins: int = 40):
    plt.figure(figsize=(6, 5))
    plt.hist(proba[y_true == 0], bins=bins, alpha=0.6, label="BENIGN")
    plt.hist(proba[y_true == 1], bins=bins, alpha=0.6, label="ATTACK")
    plt.xlabel("Predicted probability (ATTACK)")
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ========================= IP‑АНАЛИТИКА (STREAM) =========================

ALT_NAMES = {
    "Timestamp": ["Timestamp", "Flow Timestamp", "Start Time", "StartTime", "Date first seen", "Date First Seen"],
    "Src IP": ["Src IP", "Source IP", "SrcIP", "SourceIP"],
    "Dst IP": ["Dst IP", "Destination IP", "DstIP", "DestinationIP"],
    "Dst Port": ["Dst Port", "Destination Port", "DstPort"],
}

def _find_col(cols, names):
    cl = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in cl:
            return cl[n.lower()]
    return None


def _iter_csv_files(data_dir: str, globs: List[str]) -> List[str]:
    out = []
    for pat in globs:
        out.extend(sorted(glob(os.path.join(data_dir, pat))))
    return out


def ip_analytics_stream(
    data_dir: str,
    globs: List[str],
    pred_df: pd.DataFrame,
    window_sec: int,
    outdir: str,
    top_n_heatmap: int = 30,
    chunksize: int = 500_000,
    encoding: str = "auto",
):
    """
    Стримим исходные CSV кусками, чтобы не съесть память.
    Отбираем только те строки, чья Timestamp.floor(window) попадает в множество предсказанных атакующих окон.
    Считаем:
      - top_src_ips.csv (agg по всем атакующим окнам)
      - top_src_dst_pairs.csv
      - top_dst_ports.csv
      - heatmap_src_dst.png, heatmap_src_dstport.png (по топ‑N)
    """
    ensure_dir(outdir)
    if pred_df.empty or (pred_df["pred"] == 1).sum() == 0:
        # нет предсказанных атак — создадим пустые файлы, чтобы отчёт открывался без ошибок
        pd.DataFrame(columns=["src_ip", "total"]).to_csv(os.path.join(outdir, "top_src_ips.csv"), index=False)
        pd.DataFrame(columns=["src_ip", "dst_ip", "total"]).to_csv(os.path.join(outdir, "top_src_dst_pairs.csv"), index=False)
        pd.DataFrame(columns=["dst_port", "total"]).to_csv(os.path.join(outdir, "top_dst_ports.csv"), index=False)
        return

    # множество атакующих окон (по старту)
    attack_windows = pd.to_datetime(pred_df.loc[pred_df["pred"] == 1, "window_start"]).dt.floor(f"{window_sec}s")
    win_set = set(attack_windows.astype("int64").values)  # ns-ints для быстрого lookup

    # счётчики
    c_src: Dict[str, int] = {}
    c_pair: Dict[Tuple[str, str], int] = {}
    c_port: Dict[int, int] = {}

    files = _iter_csv_files(data_dir, globs)
    for fp in tqdm(files, desc="IP-аналитика: чтение CSV", unit="файл"):
        for chunk in iter_csv_chunks(fp, chunksize, encoding=encoding, on_bad_lines="skip"):
            # чистим странные колонки и BOM
            chunk = chunk.loc[:, ~chunk.columns.astype(str).str.contains(r"^Unnamed", na=False)]
            chunk.columns = chunk.columns.astype(str).str.replace("\ufeff", "", regex=True).str.strip()

            ts_col = _find_col(chunk.columns, ALT_NAMES["Timestamp"])
            if ts_col is None:
                continue
            ts = pd.to_datetime(chunk[ts_col], errors="coerce", infer_datetime_format=True)
            chunk = chunk.assign(__ts_floor=ts.dt.floor(f"{window_sec}s").astype("int64"))
            # оставляем строки, которые попали в атакующие окна
            chunk = chunk[chunk["__ts_floor"].isin(win_set)]
            if chunk.empty:
                continue

            src_col = _find_col(chunk.columns, ALT_NAMES["Src IP"])
            dst_col = _find_col(chunk.columns, ALT_NAMES["Dst IP"])
            dpt_col = _find_col(chunk.columns, ALT_NAMES["Dst Port"])

            if src_col is not None:
                vc = chunk[src_col].astype(str).value_counts()
                for ip, cnt in vc.items():
                    c_src[ip] = c_src.get(ip, 0) + int(cnt)

            if src_col is not None and dst_col is not None:
                pair = (chunk[src_col].astype(str) + " → " + chunk[dst_col].astype(str)).value_counts()
                for k, cnt in pair.items():
                    c_pair[k] = c_pair.get(k, 0) + int(cnt)

            if dpt_col is not None:
                vcp = pd.to_numeric(chunk[dpt_col], errors="coerce").dropna().astype(int).value_counts()
                for port, cnt in vcp.items():
                    c_port[int(port)] = c_port.get(int(port), 0) + int(cnt)

    # сохраняем топы
    top_src_df = (
        pd.Series(c_src, name="total")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "src_ip"})
    )
    top_src_df.to_csv(os.path.join(outdir, "top_src_ips.csv"), index=False)

    top_pair_df = (
        pd.Series(c_pair, name="total")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "src_to_dst"})
    )
    top_pair_df[["src_ip", "dst_ip"]] = top_pair_df["src_to_dst"].str.split(" → ", n=1, expand=True)
    top_pair_df.to_csv(os.path.join(outdir, "top_src_dst_pairs.csv"), index=False)

    top_port_df = (
        pd.Series(c_port, name="total")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "dst_port"})
    )
    top_port_df.to_csv(os.path.join(outdir, "top_dst_ports.csv"), index=False)

    # ---- второй проход: для теплокарты src × dst_port нужен совместный счётчик ----
    c_src_port: Dict[Tuple[str, int], int] = {}
    files = _iter_csv_files(data_dir, globs)
    for fp in tqdm(files, desc="IP‑аналитика: src×dst_port heatmap", unit="файл"):
        for chunk in iter_csv_chunks(fp, chunksize, encoding=encoding, on_bad_lines="skip"):
            chunk = chunk.loc[:, ~chunk.columns.astype(str).str.contains(r"^Unnamed", na=False)]
            chunk.columns = chunk.columns.astype(str).str.replace("\ufeff", "", regex=True).str.strip()
            ts_col = _find_col(chunk.columns, ALT_NAMES["Timestamp"])
            if ts_col is None:
                continue
            ts = pd.to_datetime(chunk[ts_col], errors="coerce", infer_datetime_format=True)
            chunk = chunk.assign(__ts_floor=ts.dt.floor(f"{window_sec}s").astype("int64"))
            chunk = chunk[chunk["__ts_floor"].isin(win_set)]
            if chunk.empty:
                continue
            src_col = _find_col(chunk.columns, ALT_NAMES["Src IP"])
            dpt_col = _find_col(chunk.columns, ALT_NAMES["Dst Port"])
            if src_col is None or dpt_col is None:
                continue
            g = chunk[[src_col, dpt_col]].copy()
            g[dpt_col] = pd.to_numeric(g[dpt_col], errors="coerce").astype("Int64")
            g = g.dropna()
            vc = (g[src_col].astype(str) + "||" + g[dpt_col].astype(str)).value_counts()
            for k, cnt in vc.items():
                s, p = k.split("||")
                c_src_port[(s, int(p))] = c_src_port.get((s, int(p)), 0) + int(cnt)

    # теперь строим теплокарту src×dst_port
    if c_src_port:
        ssp = pd.Series(c_src_port, name="total").reset_index()
        ssp.columns = ["src_ip", "dst_port", "total"]
        # top‑N срезы
        top_srcs = top_src_df.head(top_n_heatmap)["src_ip"].tolist()
        top_ports = top_port_df.head(top_n_heatmap)["dst_port"].tolist()
        m2 = (
            ssp[ssp["src_ip"].isin(top_srcs) & ssp["dst_port"].isin(top_ports)]
            .pivot_table(index="src_ip", columns="dst_port", values="total", aggfunc="sum", fill_value=0)
        )
        if not m2.empty:
            plt.figure(figsize=(max(6, 0.25 * len(m2.columns)), max(6, 0.25 * len(m2.index))))
            plt.imshow(m2.values, aspect="auto", cmap="Oranges")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(range(len(m2.columns)), m2.columns, rotation=90)
            plt.yticks(range(len(m2.index)), m2.index)
            plt.title(f"Heatmap: Src IP → Dst Port (top {top_n_heatmap})")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "heatmap_src_dstport.png"), dpi=150)
            plt.close()

    # теплокарта src→dst
    if not top_pair_df.empty:
        top_srcs = top_src_df.head(top_n_heatmap)["src_ip"].tolist()
        top_dsts = (
            top_pair_df.groupby("dst_ip")["total"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n_heatmap)
            .index.tolist()
        )
        m = (
            top_pair_df[top_pair_df["src_ip"].isin(top_srcs) & top_pair_df["dst_ip"].isin(top_dsts)]
            .pivot_table(index="src_ip", columns="dst_ip", values="total", aggfunc="sum", fill_value=0)
        )
        if not m.empty:
            plt.figure(figsize=(max(6, 0.25 * len(top_dsts)), max(6, 0.25 * len(top_srcs))))
            plt.imshow(m.values, aspect="auto", cmap="Reds")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.xticks(range(len(m.columns)), m.columns, rotation=90)
            plt.yticks(range(len(m.index)), m.index)
            plt.title(f"Heatmap: Src IP → Dst IP (top {top_n_heatmap})")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "heatmap_src_dst.png"), dpi=150)
            plt.close()


# ========================= HTML‑ОТЧЁТ =========================

def write_html_report(outdir: str, summary: Dict, figs: Dict[str, str], tables: Dict[str, str]):
    """
    outdir: папка артефактов
    summary: словарь с ключевыми числами (threshold, metrics, counts)
    figs: имена png файлов (локальные пути) {'roc':'roc.png', ...}
    tables: имена csv таблиц {'sweep':'threshold_sweep.csv', 'top_src':'top_src_ips.csv', ...}
    """
    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<title>Anti DDoS — Post‑Eval Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
h1,h2 {{ margin: 0.2em 0; }}
.container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.card {{ border: 1px solid #e3e3e3; border-radius: 10px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
.kv {{ display:grid; grid-template-columns: 220px 1fr; gap: 6px 12px; align-items: baseline; }}
small {{ color:#666; }}
table {{ border-collapse: collapse; width:100%; font-size: 14px; }}
td, th {{ border:1px solid #eee; padding:6px 8px; text-align:left; }}
th {{ background:#fafafa; }}
img {{ max-width:100%; height:auto; border:1px solid #eee; border-radius:8px; }}
.code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; background:#f7f7f7; padding:4px 6px; border-radius:6px; }}
</style>
</head>
<body>
<h1>Отчёт по пост‑оценке RF‑RFE (CIC‑IDS2017)</h1>
<p><small>Папка артефактов: <span class="code">{outdir}</span></small></p>

<div class="card">
  <h2>Итог на выбранном пороге</h2>
  <div class="kv">
    <div>Порог (threshold):</div><div><b>{summary.get('threshold')}</b></div>
    <div>Accuracy:</div><div>{summary.get('accuracy'):.4f}</div>
    <div>Precision:</div><div>{summary.get('precision'):.4f}</div>
    <div>Recall:</div><div>{summary.get('recall'):.4f}</div>
    <div>F1:</div><div>{summary.get('f1'):.4f}</div>
    <div>ROC‑AUC:</div><div>{summary.get('roc_auc'):.4f}</div>
    <div>PR‑АUC:</div><div>{summary.get('pr_auc'):.4f}</div>
    <div>Confusion Matrix [ [TN, FP], [FN, TP] ]:</div><div>{json.dumps(summary.get('confusion_matrix'))}</div>
    <div>Положительных предсказаний:</div><div>{summary.get('positives')} из {summary.get('total')}</div>
  </div>
</div>

<div class="container">
  <div class="card">
    <h2>ROC</h2>
    <img src="{figs.get('roc','roc.png')}" alt="ROC"/>
  </div>
  <div class="card">
    <h2>Precision‑Recall</h2>
    <img src="{figs.get('pr','pr.png')}" alt="PR"/>
  </div>
  <div class="card">
    <h2>Confusion Matrix</h2>
    <img src="{figs.get('cm','cm.png')}" alt="Confusion Matrix"/>
  </div>
  <div class="card">
    <h2>Распределение вероятностей</h2>
    <img src="{figs.get('score_hist','score_hist.png')}" alt="Score hist"/>
  </div>
</div>

<div class="card">
  <h2>Пороговая сводка (sweep)</h2>
  <p>Файл: <a href="{tables.get('sweep','threshold_sweep.csv')}">threshold_sweep.csv</a></p>
  <p>Метрики на сетке порогов.</p>
</div>

<div class="card">
  <h2>IP‑аналитика (по предсказанным атакующим окнам)</h2>
  <ul>
    <li><a href="{tables.get('top_src','top_src_ips.csv')}">top_src_ips.csv</a></li>
    <li><a href="{tables.get('top_pair','top_src_dst_pairs.csv')}">top_src_dst_pairs.csv</a></li>
    <li><a href="{tables.get('top_ports','top_dst_ports.csv')}">top_dst_ports.csv</a></li>
  </ul>
  <div class="container">
    <div class="card">
      <h3>Heatmap: Src → Dst</h3>
      <img src="{figs.get('hm_src_dst','heatmap_src_dst.png')}" alt="Heatmap src→dst"/>
    </div>
    <div class="card">
      <h3>Heatmap: Src → DstPort</h3>
      <img src="{figs.get('hm_src_dstport','heatmap_src_dstport.png')}" alt="Heatmap src→dst_port"/>
    </div>
  </div>
</div>

</body>
</html>
"""
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ========================= CLI MAIN =========================

def main():
    ap = argparse.ArgumentParser(description="Post‑Eval RF‑RFE: threshold, metrics, plots, IP‑analytics, HTML report")
    ap.add_argument("--artifacts", required=True, help="Папка с артефактами от rf_rfe.py")
    ap.add_argument("--threshold", type=float, default=None, help="Фиксированный порог (если не задан — автоподбор по F‑β)")
    ap.add_argument(
        "--grid",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Сетка порогов для сводки (через запятую)",
    )
    ap.add_argument("--auto-beta", type=float, default=2.0, help="β для автоподбора порога по F‑β (если threshold не задан)")
    ap.add_argument("--min-precision", type=float, default=0.80, help="Мин. precision для автоподбора")
    ap.add_argument("--data-dir", default=None, help="(опц.) Папка сырых CSV для IP‑аналитики")
    ap.add_argument("--glob", nargs="+", default=["*.pcap_ISCX.csv"], help="(опц.) Маски файлов CSV")
    ap.add_argument("--window", type=int, default=10, help="Размер окна (сек) — нужен для IP‑аналитики")
    ap.add_argument("--heatmap-top", type=int, default=30, help="Топ‑N сущностей в теплокартах")
    ap.add_argument("--chunksize", type=int, default=500_000, help="Chunk size для стрим‑чтения CSV")
    ap.add_argument("--encoding", default="auto", help="Кодировка исходных CSV: 'auto' | cp1251 | cp1252 | utf-8 | utf-8-sig | latin-1")
    args = ap.parse_args()

    ensure_dir(args.artifacts)

    # 1) загрузка модели и окон
    bundle, X, y = load_artifacts(args.artifacts)
    proba = predict_proba(bundle, X)

    # 2) выбор порога
    thr_list = [float(t) for t in args.grid.split(",") if t.strip()]
    if args.threshold is None:
        best_thr, sweep_fb = auto_threshold(
            y.values, proba, beta=args.auto_beta, min_precision=args.min_precision, grid=thr_list
        )
        threshold = float(best_thr)
        # сохраним таблицу с fbeta
        sweep_fb.sort_values(["fbeta", "recall"], ascending=[False, False]).to_csv(
            os.path.join(args.artifacts, "threshold_sweep_fbeta.csv"), index=False
        )
    else:
        threshold = float(args.threshold)

    # 3) метрики @ threshold + sweep
    met = metrics_at_threshold(y.values, proba, threshold)
    sweep = sweep_thresholds(y.values, proba, thr_list)
    sweep.to_csv(os.path.join(args.artifacts, "threshold_sweep.csv"), index=False)
    pd.DataFrame([met]).to_csv(os.path.join(args.artifacts, "metrics_at_threshold.csv"), index=False)

    # 4) predictions.csv (для аудита)
    preds = (proba >= threshold).astype(int)
    pred_df = pd.DataFrame(
        {
            "window_start": X.index,
            "proba": proba,
            "pred": preds,
            "true": y.values,
        }
    )
    pred_df.to_csv(os.path.join(args.artifacts, "predictions.csv"), index=False)

    # 5) графики
    plot_roc(y.values, proba, os.path.join(args.artifacts, "roc.png"))
    plot_pr(y.values, proba, os.path.join(args.artifacts, "pr.png"), mark_thr=threshold, pred=preds)
    plot_confmat(np.array(met["confusion_matrix"]), os.path.join(args.artifacts, "cm.png"))
    plot_score_hist(proba, y.values, os.path.join(args.artifacts, "score_hist.png"))

    # 6) (опц.) IP‑аналитика
    figs = {
        "roc": "roc.png",
        "pr": "pr.png",
        "cm": "cm.png",
        "score_hist": "score_hist.png",
        "hm_src_dst": "heatmap_src_dst.png",
        "hm_src_dstport": "heatmap_src_dstport.png",
    }
    tables = {
        "sweep": "threshold_sweep.csv",
        "top_src": "top_src_ips.csv",
        "top_pair": "top_src_dst_pairs.csv",
        "top_ports": "top_dst_ports.csv",
    }
    if args.data_dir:
        ip_analytics_stream(
            data_dir=args.data_dir,
            globs=args.glob,
            pred_df=pred_df,
            window_sec=args.window,
            outdir=args.artifacts,
            top_n_heatmap=args.heatmap_top,
            chunksize=args.chunksize,
            encoding=args.encoding,
        )
    else:
        # создадим «пустые» файлы‑заглушки, чтобы отчёт открылся без ошибок
        for k in ["top_src_ips.csv", "top_src_dst_pairs.csv", "top_dst_ports.csv"]:
            f = os.path.join(args.artifacts, k)
            if not os.path.exists(f):
                pd.DataFrame().to_csv(f, index=False)

    # 7) HTML‑отчёт
    write_html_report(args.artifacts, met, figs, tables)

    # 8) консольная сводка
    print("\n=== METRICS @ threshold={:.2f} ===".format(threshold))
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "positives", "total"]:
        print(f"{k:>10s}: {met[k]}")
    print("confusion_matrix:", met["confusion_matrix"])
    print(f"\n[OK] Отчёт: {os.path.abspath(os.path.join(args.artifacts,'index.html'))}")
    print("     Таблица порогов: threshold_sweep.csv")
    if os.path.exists(os.path.join(args.artifacts, "threshold_sweep_fbeta.csv")):
        print("     Автоподбор (F‑beta): threshold_sweep_fbeta.csv")
    if args.data_dir:
        print("     IP‑аналитика: top_src_ips.csv, top_src_dst_pairs.csv, top_dst_ports.csv, heatmap_*")
    else:
        print("     IP‑аналитика пропущена (не задан --data-dir).")


if __name__ == "__main__":
    main()
