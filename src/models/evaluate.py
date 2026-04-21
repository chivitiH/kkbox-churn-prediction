import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score


def evaluate(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))

    print(f"🔥 AUC: {auc:.4f}")
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"⚖️ F1 Score: {f1:.4f}")
    print(f"📢 Recall (churn): {recall:.4f}")

    return {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "recall": recall
    }
