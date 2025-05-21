# utils/metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def compute_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

def print_metrics(metrics_dict):
    print("\nPerformance Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k.capitalize():<10}: {v:.4f}")

def aggregate_metrics_over_runs(metrics_list):
    keys = metrics_list[0].keys()
    mean_metrics = {
        k: np.mean([m[k] for m in metrics_list]) for k in keys
    }
    std_metrics = {
        k: np.std([m[k] for m in metrics_list]) for k in keys
    }
    return mean_metrics, std_metrics
