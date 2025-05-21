import torch
import numpy as np
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
from collections import defaultdict
import matplotlib.pyplot as plt

def run_lime_explanation(model, dataloader, class_names, device='cuda',
                         samples_per_class=5, save_dir='./explanations',
                         correct_only=False):  # ✅ 추가된 인자

    model.eval()
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    explainer = lime_image.LimeImageExplainer()
    saved_counts = defaultdict(int)
    total_required = {i: samples_per_class for i in range(len(class_names))}

    def batch_predict(images_np):
        batch = torch.tensor(images_np).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            logits = model(batch)[0] if isinstance(model(batch), tuple) else model(batch)
        return logits.detach().cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

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

            # ✅ 정답 아닐 경우 스킵
            if correct_only and pred_class != label_id:
                continue

            # 이미지 복원
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

            temp, mask = explanation.get_image_and_mask(
                pred_class,
                positive_only=True,
                num_features=100,
                hide_rest=False
            )

            filename = os.path.basename(path)
            filename_no_ext = os.path.splitext(filename)[0]  # 확장자 제거
            output_path = os.path.join(save_dir, f"{class_names[label_id]}_{filename_no_ext}.png")

            plt.imshow(mark_boundaries(temp, mask))
            plt.axis('off')
            plt.title(f"LIME - {class_names[label_id]}")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

            saved_counts[label_id] += 1
            if all(saved_counts[i] >= total_required[i] for i in total_required):
                print("[LIME] Done for all classes.")
                return
