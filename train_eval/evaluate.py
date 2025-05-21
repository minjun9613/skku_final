# train_eval/evaluate.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from ptflops import get_model_complexity_info

def evaluate_model(model, dataloader, class_names=None, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[0] if isinstance(model(inputs), tuple) else model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return cm


def compute_efficiency(model, input_size=(3, 224, 224), device='cuda', test_loader=None):
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(1, *input_size).to(device)

    # GFLOPs
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
    print(f"\nModel GFLOPs and Parameters:")
    print(f"GFLOPs: {macs}, Params: {params}")

    # FPS (Inference Speed)
    if test_loader:
        total_images = 0
        start_time = time.time()
        with torch.no_grad():
            for inputs, _, _, _ in test_loader:
                inputs = inputs.to(device)
                _ = model(inputs)[0] if isinstance(model(inputs), tuple) else model(inputs)
                total_images += inputs.size(0)
        end_time = time.time()
        fps = total_images / (end_time - start_time)
        print(f"FPS (Images/sec): {fps:.2f}")
        return macs, params, fps
    else:
        return macs, params, None
