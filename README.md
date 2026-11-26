# Explainable ML-Based Phishing URL Detection  (Group InsightX, UPES)

Team: Som Agrawal (590017323), Raunak Raghvendra (590014977), Archit Ojha (590012675), Yug Sharma (590011934), Aryan Sharma (590017816)

Research Methodology in CS - project code. Classifies URLs as **legitimate / phishing** using **lexical, DNS and WHOIS**
features, compares four ML classifiers (logistic regression, random forest, gradient boosting, linear SVM), reports
**success and failure percentages** and explains predictions with **XAI** (permutation importance, PDP, SHAP, LIME).

## Folder layout
```
src/features.py            lexical features (+ optional live DNS/WHOIS lookup)
src/generate_sample_data.py   SYNTHETIC demo data generator
src/make_pipeline_figure.py   draws the simple pipeline figure
src/train_evaluate.py      trains/compares models, metrics, confusion matrices, ROC, saves best model
src/explain.py             XAI: permutation importance, feature-family importance, PDP, local explanations, SHAP/LIME (optional)
src/predict.py             classify + explain a single URL
data/                      sample (synthetic) data + instructions for the real Kaggle file
results/                   metrics.csv, success_failure.csv, figures/, explanations (DEMO run on synthetic data); figures/architecture.png is your own diagram
models/best_model.joblib   saved best model (demo)
docs/                      viva Q&A and submission checklist
```

## Setup
```
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install shap lime xgboost                            # optional: SHAP, LIME, XGBoost
```

## Run (demo, ~10 seconds)
```
python src/generate_sample_data.py
cd src
python train_evaluate.py --data ../data/sample_phishing_synthetic.csv --out ../results --models-dir ../models
python explain.py --results ../results --model ../models/best_model.joblib
python predict.py "http://paypal.secure-login-x9.xyz/verify" --model ../models/best_model.joblib --meta ../results/meta.json --domain-age-days 5 --mx-count 0
```

## Run on the real Kaggle dataset
See `data/README.md`. Typical command (from `src/`):
```
python train_evaluate.py --data ../data/final_dataset.csv --url-col none --label-col label --positive-label 1 --groups-json ../data/groups.json --cv-folds 3
python explain.py --results ../results --model ../models/best_model.joblib --n 2000 --shap-n 300
```
Dataset: https://www.kaggle.com/datasets/elifzelik/phishing-url-features-dataset (file `final_dataset.csv`, 74 numeric features; the loader auto-detects a label column and infers the Lexical/DNS/WHOIS family of each feature from its name - check the printed counts against 53/9/12, or pass `--groups-json`).
Then copy the numbers in `results/metrics.csv` and `results/success_failure.csv` and the figures in `results/figures/` into your report, paper (Table III) and slides.

## Results (full run on final_dataset.csv, 115,984 test URLs)
| Model | Success % | Failure % | Phishing missed % | False alarms % | F1 |
|---|---|---|---|---|---|
| Random Forest (best) | 98.51 | 1.49 | 2.63 | 0.68 | 0.9819 |
| Gradient Boosting | 98.23 | 1.77 | 2.71 | 1.10 | 0.9786 |
| Logistic Regression | 94.38 | 5.62 | 8.04 | 3.90 | 0.9315 |
| SVM (Linear) | 94.28 | 5.72 | 8.25 | 3.92 | 0.9302 |

Permutation importance by family: Lexical 87.1%, WHOIS 9.2%, DNS 3.7% (2,000-URL sample; shares vary with sample size).
`results/test_split.csv` (47 MB) and the trained model are not included; re-run `train_evaluate.py` to recreate them.
`data/sample_phishing_synthetic.csv` is synthetic and was used only to test the code.

## Explainability methods
* Permutation importance (global), aggregated by feature family (Lexical / DNS / WHOIS)
* Partial dependence plots of the most important features
* Occlusion-based local explanation (effect of each feature on the log-odds of "phishing") - used by `predict.py`
* SHAP summary plot and LIME HTML explanations when `shap` / `lime` are installed

## Note on single-URL prediction
`predict.py` computes 21 lexical features from a URL (plus optional DNS/WHOIS values you pass in). A model trained on the Kaggle file's 74 ready-made features needs the dataset authors' feature extraction to score brand-new URLs; use `predict.py` with a model trained on URL-based features (the demo flow) or implement that extraction.
