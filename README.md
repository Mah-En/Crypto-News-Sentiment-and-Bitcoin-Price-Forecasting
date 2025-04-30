# Crypto-News Sentiment → Bitcoin Price Forecasting  
*A reproducible end-to-end ML pipeline*

---

## Project snapshot
| &nbsp; | **Metric** | **Score** |
|---|---|---|
| Macro-F1 | **0.684** |
| ROC-AUC | **0.712** |
| Horizon | 1 hour |
| Best model | XGBoost (Bayesian-tuned) |

> This repository ingests live crypto-news, extracts sentiment features, and predicts whether Bitcoin will **rise (↑)**, **fall (↓)**, or stay **flat (→)** in the next hour.  
> All experiments are fully reproducible from raw data to trained models and a Gradio demo.

---

## Table of contents
1. [Folder layout](#folder-layout)  
2. [Datasets](#datasets)  
3. [Pipeline overview](#pipeline-overview)  
4. [Feature engineering](#feature-engineering)  
5. [Modelling & results](#modelling--results)  
6. [Quick start](#quick-start)  
7. [Re-training](#re-training)  
8. [Gradio demo](#gradio-demo)  
9. [Limitations](#limitations)  
10. [License & citation](#license--citation)

---

## Folder layout
```
.
├── data_raw/            # downloaded CSV / JSON
├── data_processed/      # hourly Parquet feature store
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── collect.py       # fetch news & OHLCV
│   ├── merge.py
│   └── train.py
├── models/
│   └── xgb_best.json
├── figures/             # .png/.pdf used in the report
└── README.md
```

---

## Datasets
| Dataset | Period | Size | Link |
|---------|--------|------|------|
| **Crypto News** | 2013-02-11 → 2024-04-18 | 44 938 articles | kaggle.com/oliviervha/crypto-news |
| **Bitcoin OHLCV** (Binance API) | 2018-01-01 → 2024-04-18 | 3.3 M rows (1-min) | api.binance.com |

Both are pulled automatically by `src/collect.py`.

---

## Pipeline overview
![Pipeline](figures/pipeline.pdf)

1. **Collect** raw news + minute OHLCV  
2. **Clean** timestamps, drop nulls  
3. **Resample** OHLCV → hourly bars  
4. **Merge** each article with the *next* hour’s candle (30-min buffer)  
5. **Feature store** written as compressed Parquet  

---

## Feature engineering
| Tag | Description |
|-----|-------------|
| **F1** | TF–IDF (1–2 grams) of lemmatised headlines |
| **F2** | Sentiment: VADER polarity, RoBERTa polarity, TextBlob subjectivity |
| **F3** | 768-dim Sentence-BERT embedding (`all-mpnet-base-v2`) |
| **F4** | *Source-Credibility Index* (target-encoded outlet) |
| **F5** | Cyclic hour, weekend flag, time-since-prev-news |
| **F6** | Market context: lagged returns, Bollinger-band width |

---

## Modelling & results
| Model | Accuracy | Macro-F1 | ROC-AUC |
|-------|----------|----------|---------|
| Persistence | 0.492 | 0.333 | 0.500 |
| LogReg (TF–IDF) | 0.628 | 0.571 | 0.643 |
| Random Forest | 0.657 | 0.602 | 0.679 |
| **XGBoost** | **0.694** | **0.684** | **0.712** |
| BERT + MLP | 0.673 | 0.658 | 0.701 |

Confusion matrix  

![Confusion matrix](figures/confusion_matrix.pdf)

---

## Quick start
```bash
# clone repo
git clone https://github.com/your-handle/crypto-news-btc-forecast.git
cd crypto-news-btc-forecast

# create env
conda env create -f environment.yml
conda activate btc-sentiment

# download & preprocess
python src/collect.py           # raw data → data_raw/
python src/merge.py             # raw → feature store

# train model
python src/train.py --model xgb --optuna 100
```

---

## Re-training
To update the model with fresh data, schedule `collect.py` daily
(e.g. cron) and re-run `train.py` weekly.  
All hyper-parameter settings are saved in `models/xgb_best.json`.

---

## Gradio demo
```bash
python -m gradio app.py
```
A browser tab will open; paste a headline and view:
* class probabilities (↑ → ↓)  
* SHAP bar explaining the top 10 features

---

## Limitations
* **Sarcasm & hyperbole** often mislead sentiment models.  
* **Macro shocks** (FOMC meetings) override crypto-specific news.  
* **Concept drift** – narratives change; periodic re-training is vital.

See the full discussion in the `/report` directory.

---

## License & citation
Apache 2.0 – free for academic and commercial use.  
If you build on this work, please cite:

```
@misc{asl2024btcnews,
  title  = {Forecasting Bitcoin Price from Crypto-News Sentiment},
  author = {Reza Asl},
  year   = {2024},
  url    = {https://github.com/your-handle/crypto-news-btc-forecast}
}
```

---

*Happy coding & profitable trading!*