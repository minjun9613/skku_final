# main.py
import torch
import yaml
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from all_models import get_model
from utils.dataset import get_dataloaders
from train_eval.train import train_model
from train_eval.evaluate import evaluate_model, compute_efficiency
from train_eval.stability import plot_learning_curves, report_stability
from train_eval.explainability_lime import run_lime_explanation
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from matplotlib.backends.backend_pdf import PdfPages

# 재현성 확보
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)

    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(config['log_dir'])
    logger.info("Loaded configuration.")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        config['data_dir'], batch_size=config['batch_size'])


    os.makedirs("results", exist_ok=True)
    all_results = []
    history_log = []

    for model_name in config['model_names']:
        logger.info(f"Training model: {model_name}")
        model = get_model(model_name).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        model, history = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            num_epochs=config['epochs'], device=device
        )

        save_path = os.path.join(config['save_dir'], f"{model_name}_model.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")

        # Save best epoch info
        best_epoch = int(pd.Series(history[3]).idxmax()) + 1
        with open(f"results/{model_name}_best_epoch.txt", "w") as f:
            f.write(f"Best epoch based on val acc: {best_epoch}\n")

        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels, _, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)[0] if isinstance(model(inputs), tuple) else model(inputs)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_confusion_matrix.png")
        plt.close()

        # Save metrics
        metrics = compute_metrics(y_true, y_pred)
        gflops, params, fps = compute_efficiency(model, input_size=tuple(config['input_size']), test_loader=test_loader)

        # 단위를 자동 처리하여 gflops를 float로 변환
        if isinstance(gflops, str):
            if "GMac" in gflops:
                gflops = float(gflops.replace(" GMac", ""))
            elif "MMac" in gflops:
                gflops = float(gflops.replace(" MMac", "")) / 1000  # Mega → Giga 변환
            else:
                gflops = float(gflops)

        metrics.update({
            'model_name': model_name,
            'gflops': gflops,
            'fps': fps
        })

        all_results.append(metrics)

        # Save learning history
        for epoch, (tr_acc, va_acc, tr_loss, va_loss) in enumerate(zip(*history), 1):
            history_log.append({
                'model': model_name,
                'epoch': epoch,
                'train_acc': tr_acc,
                'val_acc': va_acc,
                'train_loss': tr_loss,
                'val_loss': va_loss
            })

        plot_learning_curves(*history, model_name=model_name)
        report_stability(history[3])

        run_lime_explanation(
            model,
            test_loader,
            class_names=class_names,
            device=device,
            samples_per_class=1,
            save_dir=f'./explanations/{model_name}'
        )

    # Save metrics to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("results/metrics.csv", index=False)
    logger.info("Saved all metrics to results/metrics.csv")

    # Save full learning log
    df_log = pd.DataFrame(history_log)
    df_log.to_csv("results/training_history.csv", index=False)

    # Save LaTeX table
    latex_table = df[["model_name", "accuracy", "precision", "recall", "f1_score"]].round(4).to_latex(index=False)
    with open("results/latex_metrics_table.tex", "w") as f:
        f.write(latex_table)

    # Generate PDF Report
    with PdfPages("results/summary_report.pdf") as pdf:
        plt.figure(figsize=(11, 8))
        plt.axis('off')
        plt.title("Model Evaluation Summary Report", fontsize=24)
        plt.text(0.1, 0.5, "Generated Performance Summary\nIncluding Metrics, Graphs, and Learning Curves",
                 fontsize=16, verticalalignment='center')
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis('off')
        tbl = ax.table(cellText=df.round(4).values, colLabels=df.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.5)
        plt.title("Performance Metrics Table", fontsize=16)
        pdf.savefig()
        plt.close()

        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            plt.figure(figsize=(10, 6))
            sns.barplot(x="model_name", y=metric, data=df)
            plt.title(f"Model {metric.capitalize()} Comparison")
            plt.xticks(rotation=30)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="gflops", y="f1_score", hue="model_name", data=df, s=100)
        for i in range(len(df)):
            plt.text(df["gflops"][i], df["f1_score"][i], df["model_name"][i])
        plt.title("GFLOPs vs F1 Score Trade-off")
        plt.xlabel("GFLOPs")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        for model in df_log["model"].unique():
            df_model = df_log[df_log["model"] == model]
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(df_model["epoch"], df_model["train_acc"], label="Train Acc")
            plt.plot(df_model["epoch"], df_model["val_acc"], label="Val Acc")
            plt.title(f"{model} Accuracy Curve")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(df_model["epoch"], df_model["train_loss"], label="Train Loss")
            plt.plot(df_model["epoch"], df_model["val_loss"], label="Val Loss")
            plt.title(f"{model} Loss Curve")
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()