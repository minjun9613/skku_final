import os
import torch
import numpy as np
import yaml
from lime import lime_image
from utils.dataset import get_dataloaders
from all_models import get_model
from skimage.segmentation import mark_boundaries
import pandas as pd
from collections import defaultdict


def to_tensor(img_np, mean, std, device):
    img_np = img_np.astype(np.float32)
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    img_tensor = (img_tensor - mean_tensor) / std_tensor
    return img_tensor.unsqueeze(0).to(device)


def run_lime_perturbation(model, dataloader, class_names, device, samples_per_class=5,
                          result_path="./lime_score_drop.csv"):
    model.eval()
    model = model.to(device)
    explainer = lime_image.LimeImageExplainer()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    results = []
    saved_counts = defaultdict(int)
    total_required = {i: samples_per_class for i in range(len(class_names))}

    def batch_predict(images_np):
        batch = torch.tensor(images_np).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            logits = model(batch)
            logits = logits[0] if isinstance(logits, tuple) else logits
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    for images, labels, _, paths in dataloader:
        for img, label, path in zip(images, labels, paths):
            label_id = label.item()
            if saved_counts[label_id] >= total_required[label_id]:
                continue

            img_tensor = img.to(device).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                logits = output[0] if isinstance(output, tuple) else output
                pred_class = torch.argmax(logits, dim=1).item()

            if pred_class != label_id:
                continue

            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            explanation = explainer.explain_instance(
                img_np,
                batch_predict,
                labels=(pred_class,),
                top_labels=None,
                hide_color=0,
                num_samples=2000
            )

            _, mask = explanation.get_image_and_mask(
                pred_class,
                positive_only=True,
                num_features=100,
                hide_rest=False
            )

            # 마스킹
            masked_img_np = img_np.copy()
            masked_img_np[mask == 1] = 0

            input_orig = to_tensor(img_np, mean, std, device)
            input_masked = to_tensor(masked_img_np, mean, std, device)

            with torch.no_grad():
                logits_orig = model(input_orig)
                logits_masked = model(input_masked)
                if isinstance(logits_orig, tuple):
                    logits_orig = logits_orig[0]
                if isinstance(logits_masked, tuple):
                    logits_masked = logits_masked[0]

                score_orig = torch.softmax(logits_orig, dim=1)[0, pred_class].item()
                score_masked = torch.softmax(logits_masked, dim=1)[0, pred_class].item()

            score_drop = score_orig - score_masked
            results.append({
                "filename": os.path.basename(path),
                "class": class_names[label_id],
                "original_score": round(score_orig, 4),
                "masked_score": round(score_masked, 4),
                "score_drop": round(score_drop, 4)
            })

            saved_counts[label_id] += 1
            if all(saved_counts[i] >= total_required[i] for i in total_required):
                print("[LIME PERTURBATION] Done for all classes.")
                break

    df = pd.DataFrame(results)
    df.to_csv(result_path, index=False)
    print(f"[SAVED] Score drop results saved to {result_path}")


def main():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = get_dataloaders(
        config['data_dir'], batch_size=config['batch_size']
    )

    model_name = "hybrid"
    model_path = os.path.join(config['save_dir'], f"{model_name}_model.pt")

    if not os.path.exists(model_path):
        print(f"[ERROR] No saved model found at {model_path}")
        return

    print(f"[LOAD] {model_name} from {model_path}")
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    run_lime_perturbation(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
        samples_per_class=30,
        result_path=f"./lime_score_drop_{model_name}.csv"
    )


if __name__ == "__main__":
    main()
