# utils/gflops_fps.py
import torch
import time
from ptflops import get_model_complexity_info

def compute_gflops(model, input_size=(3, 224, 224)):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model,
            input_size,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
    return macs, params

def compute_fps(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    total_images = 0
    start = time.time()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)[0] if isinstance(model(inputs), tuple) else model(inputs)
            total_images += inputs.size(0)
    end = time.time()
    elapsed = end - start
    fps = total_images / elapsed
    return fps

def print_efficiency(macs, params, fps=None):
    print(f"\nGFLOPs: {macs}")
    print(f"Params: {params}")
    if fps is not None:
        print(f"FPS: {fps:.2f}")
