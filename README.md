```markdown
# Anti_DDoS_NSPK

Мини-фреймворк для обнаружения DDoS-атак на датасете **CICIDS2017**. Реализованы два рабочих пайплайна:
- **RF-RFE**: RandomForest + Recursive Feature Elimination (обучение, пост-оценка, отчёт)
- **Ансамблевый стекинг**: Stacking (RandomForest + ExtraTrees + GradientBoosting → LogisticRegression)



## Установка
```bash
# Клонируем репозиторий
git clone https://github.com/Spaceonmymind/Anti_DDoS_NSPK.git
cd Anti_DDoS_NSPK

# Виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Зависимости
pip install -r requirements.txt
````

---

## Подготовка данных

Данные CICIDS2017 в репозитории **не хранятся**. Скачайте CSV и разложите локально:

```
MachineLearningCVE/     # флоу CSV из CICIDS2017
TrafficLabelling/       # CSV c разметкой (если используете пост-оценку/аналитику)
```

> Большие файлы не коммитим. Добавьте в .gitignore каталоги с данными и артефактами.

---

## 1) RF-RFE: обучение

```bash
python rf_rfe.py \
  --data-dir ./TrafficLabelling \
  --artifacts ./artifacts_full
```

Выход:

* `artifacts_full/rf_rfe_cicids2017.joblib` — модель + scaler + выбранные признаки
* `artifacts_full/window_{features,labels}.csv` — оконные признаки/метки
* `artifacts_full/metrics.csv` — сводка по train/test

---

## 2) RF-RFE: пост-оценка + отчёт + IP-аналитика

```bash
python post_eval_full.py \
  --artifacts ./artifacts_full \
  --data-dir ./TrafficLabelling \
  --glob "*.pcap_ISCX.csv" \
  --window 10 \
  --auto-beta 2.0 \
  --min-precision 0.80 \
  --grid "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" \
  --heatmap-top 30
```

Что создаётся в `./artifacts_full/`:

* Графики: `roc.png`, `pr.png`, `cm.png`, `score_hist.png`
* Таблицы: `metrics_at_threshold.csv`, `threshold_sweep.csv`, `threshold_sweep_fbeta.csv`, `predictions.csv`
* IP-аналитика: `top_src_ips.csv`, `top_src_dst_pairs.csv`, `top_dst_ports.csv`, `heatmap_src_dst.png`, `heatmap_src_dstport.png`
* HTML-отчёт: **`index.html`** (открыть в браузере)

---

## 3) Ансамблевый стекинг

```bash
python ensemble_stacking.py \
  --sample-per-file 500000 \
  --max-per-class 300000 \
  --artifacts-dir ./artifacts_ensemble
```

Выход:

* `artifacts_ensemble/ensemble_stacking.joblib`
* `artifacts_ensemble/metrics.{csv,json}`
* `artifacts_ensemble/feature_schema.csv`
* Графики: `confusion_matrix.png`, `roc.png`, `pr.png`, `score_hist.png`

Параметры `--sample-per-file` и `--max-per-class` помогут ограничить объём для ноутбука. Поставьте `0`, чтобы обучать на всех строках.

---

## Референс-результаты (RF-RFE, Test)

```
Окон: 1,330   Атакующих окон: 5.04%   Train: 930   Test: 400
Выбранные признаки (RFE, 8): session_diversity, avg_response_time, std_response_time,
repeat_request_ratio, packet_size_variance, connection_duration,
entropy_protocol, norm_entropy_dst_ip

Accuracy: 0.9975
Precision: 0.9545
Recall:    1.0000
F1-score:  0.9767
ROC-AUC:   0.9999
PR-AUC:    0.9978
p95 inference per window: ~0.119 ms

Confusion Matrix:
[[378, 1],
 [  0, 21]]
```
