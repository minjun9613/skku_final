# train_eval/stability.py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def compute_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean, mean - h, mean + h

def plot_learning_curves(train_acc, val_acc, train_loss, val_loss, model_name=None):
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_learning_curves.png")  # 또는 PDF에 저장용
    plt.close()


def report_stability(val_acc_list):
    mean, lower, upper = compute_confidence_interval(val_acc_list)
    print(f"Validation Accuracy Stability (95% CI): {mean:.4f} ({lower:.4f} - {upper:.4f})")
    return mean, lower, upper
