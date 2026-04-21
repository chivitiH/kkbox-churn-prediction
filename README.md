# 🚀 KKBox Churn Prediction System

End-to-end machine learning pipeline to predict customer churn using real-world subscription and user behavior data.

---

## 🎯 Objective

Predict whether a user will churn (stop subscription) and enable retention strategies.

---

## 📊 Dataset

KKBox churn dataset:
- Transactions
- User logs (listening behavior)
- Members information

---

## ⚙️ Pipeline

1. Data loading & preprocessing
2. Feature engineering (recency, frequency, account age)
3. Model training (LightGBM)
4. Evaluation (AUC, F1, Recall)
5. Inference pipeline
6. Feature alignment (train vs inference)

---

## 📈 Results

- AUC: ~0.80
- Recall (churn): ~0.53
- F1-score: ~0.33

---

## 📦 Outputs

- artifacts/model.pkl
- artifacts/predictions.csv
- artifacts/feature_importance.csv
- artifacts/metrics.json

---

## 🚀 How to run

### Train
PYTHONPATH=. python src/pipeline/train_pipeline.py

### Inference
PYTHONPATH=. python src/pipeline/inference_pipeline.py

---

## 🧠 Key Learnings

- Handling large multi-table datasets
- Preventing data leakage
- Feature engineering for churn prediction
- Building production-ready ML pipelines
- Aligning training and inference features

---

## 💡 Business Use

Target high-risk users with retention campaigns (emails, discounts, offers).

