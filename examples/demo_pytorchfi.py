"""
PytorchFI demo: 
- Define a small CNN
- Create a FaultInjection instance
- Inject a neuron fault and a weight fault
- Run original and corrupted models and print outputs
Run (from repo root):
    bash -lc "python examples/demo_pytorchfi.py"
"""

import os
import sys
import time
import math
from PIL import Image
from torchvision import transforms

# __file__ = .../pytorchfi/examples/demo_pytorchfi.py
# -> repo_root = .../pytorchfi
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchfi import core
from pytorchfi import neuron_error_models as nem
from pytorchfi import weight_error_models as wem
from pytorchfi import fault_detector as fd
from pytorchfi.neuron_error_models import NO_FAULTS

torch.manual_seed(0)
IN_SIZE = 26 # Change for diff. input size
N_RUNS = 10000 # Change if want more run
# Change algorithm, one true at a time (e.g. IS_TVLSI = FALSE, IS_TC = FALSE => "OURS" algorithm)
IS_TVLSI = False 
IS_TC = True
if IS_TVLSI:
    NAME = "TVLSI'21"
elif IS_TC:
    NAME = "TC'23"
else:
    NAME = "OURS"
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1: Conv 3x3 + Pool
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # Spatial size
        self.conv1_pointwise = nn.Conv2d(16, 16, kernel_size=1)   # 1x1 conv

        # Block 2: Conv 3x3 + Pool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_pointwise = nn.Conv2d(32, 32, kernel_size=1)

        # 32x32 -> 8x8
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))               # 16 x 32 x 32
        x = F.relu(self.conv1_pointwise(x))     # 16 x 32 x 32
        x = F.max_pool2d(x, 2)                  # 16 x 16 x 16

        # Block 2
        x = F.relu(self.conv2(x))               # 32 x 16 x 16
        x = F.relu(self.conv2_pointwise(x))     # 32 x 16 x 16
        x = F.max_pool2d(x, 2)                  # 32 x 8 x 8

        # Flatten
        x = x.view(x.size(0), -1)               # 32*8*8
        x = self.fc(x)
        return x

class SimpleCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

def get_detection_result(inj_dict, det_dict):
    inj_set= set(zip(inj_dict["layer"], inj_dict["batch"], inj_dict["channel"]))
    det_set = set(zip(det_dict["layer"], det_dict["batch"], det_dict["channel"]))
    
    detected = len(inj_set & det_set)
    fn       = len(inj_set - det_set)
    fp       = len(det_set - inj_set)
    
    return detected, fn, fp 

def injection_and_detect():
    # Prepare model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN2().to(device)
    model.eval()
    
    # Create a dummy input batch (batch_size=1)
    batch_size = 1
    input_shape = [3, IN_SIZE, IN_SIZE]
    # dummy = torch.rand((batch_size, *input_shape), device=device)
    dummy = get_img("lena.png")

    # Create FaultInjection instance with single_bit_flip_func for bit flip
    pfi = nem.single_bit_flip_func(
        model=model,
        batch_size=batch_size,
        input_shape=input_shape,
        layer_types=[nn.Conv2d],
        use_cuda=torch.cuda.is_available(),
        bits=8  # 8-bit quantization
    )
    
    layer_ranges = get_max_range(model, dummy)
    # Single bit flip injection
    # nem.random_neuron_single_bit_inj(pfi, layer_ranges)
    
    # Multiple bit flip injection
    nem.random_neuron_multiple_bit_inj(pfi, layer_ranges)
        
    # Setup Fault Detector
    detector = fd.FaultDetector(
        model=pfi.corrupted_model,
        layer_types=[nn.Conv2d],
        use_cuda=torch.cuda.is_available(),
        remove_bias=True,
        total_faults=pfi.total_faults_injected,
        is_tvlsi=IS_TVLSI,
        is_tc= IS_TC
    )
    detector.register_hooks()

    with torch.no_grad():
        out_neuron_fault = pfi.corrupted_model(dummy)
    
    inj_dict, det_dict = pfi.injection_dict, detector.detected_dict
    
    print(f"[RESULT] Injection summary: {inj_dict}")
    print(f"[RESULT] Detection summary: {det_dict}")
    
    detected, fn, fp = get_detection_result(inj_dict, det_dict)

    return detected, fn, fp

def get_max_range(model, inp, margin=1.1, eps=1e-6):
    model.eval()
    acts_max = []
    
    def hook_collect_max(module, inp, out):
        # out: Tensor (N, C, H, W)
        max_val = out.detach().abs().max().item()
        acts_max.append(max_val)
    
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_collect_max))
    
    with torch.no_grad():
        _ = model(inp)
    
    layer_ranges = [max(m, eps) * margin for m in acts_max] # 10% margin
    
    return layer_ranges

def wilson_score_interval(detected, n, z=1.96):
    """
    Wilson Score Interval for detection rate.
    Returns:
        tuple: confidence interval 95% (lower, upper).
    """
    det_rate = detected/n
    denominator = 1 + z**2 / n
    center_adjusted_probability = det_rate + z**2 / (2 * n)
    margin_of_error = z * math.sqrt((det_rate * (1 - det_rate)) / n + z**2 / (4 * n**2))

    lower_bound = (center_adjusted_probability - margin_of_error) / denominator
    upper_bound = (center_adjusted_probability + margin_of_error) / denominator

    return lower_bound, upper_bound

def get_img(path):
    # Abs path
    base_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(base_path, path) 

    img_path = os.path.abspath(img_path)
   
    image = Image.open(img_path)
    
    transform = transforms.Compose([transforms.Resize((IN_SIZE, IN_SIZE)), transforms.ToTensor()])
    dummy_input = transform(image).unsqueeze(0)
    
    return dummy_input

def main():
    total_detected = 0
    total_fn = 0
    total_fp = 0

    for i in range(N_RUNS):
        detected, fn, fp = injection_and_detect()
        total_detected += detected
        total_fn += fn
        total_fp += fp

        print(f"Run {i+1:3d}: detected={detected}, fn={fn}, fp={fp}")
    
    lower, upper = wilson_score_interval (total_detected, N_RUNS*NO_FAULTS) # Tem faults each run
    detection_rate =(total_detected/(N_RUNS*NO_FAULTS))*100
    total = total_detected + total_fn + total_fp
    det_rate = (total_detected/total)*100
    fn_rate =  (total_fn/total)*100
    fp_rate =  (total_fp/total)*100
    
    print("\n==== Summary after", N_RUNS, "runs", NAME, "====")
    print("Total detected:", total_detected)
    print("Total FN      :", total_fn)
    print("Total FP      :", total_fp)
    print("FD Rate       :", f"{detection_rate:.2f}%")
    print("FD Rate Confidence interval 95%:", f"({lower:.4f}, {upper:.4f})")
    print(f"DET, FP, FN   : {det_rate:.2f}%, {fn_rate:.2f}%, {fp_rate:.2f}%")
    
if __name__ == "__main__":
    start = time.perf_counter()     

    main()                          

    end = time.perf_counter()       
    
    # Compute time for running experiment
    elapsed = end - start
    
    print(f"\n[Time] Finished in {elapsed:.4f} seconds")
    
