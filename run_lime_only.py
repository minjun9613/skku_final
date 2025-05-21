# lime_hybrid_only.py
import os
import torch
import yaml
from all_models import get_model
from utils.dataset import get_dataloaders
from train_eval.explainability_lime import run_lime_explanation

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

    print(f"[LIME] Running only for correct predictions of {model_name}")
    run_lime_explanation(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
        samples_per_class=30,  # 필요에 따라 조절
        save_dir=f"./explanations/{model_name}",
        correct_only=True
    )

if __name__ == "__main__":
    main()
