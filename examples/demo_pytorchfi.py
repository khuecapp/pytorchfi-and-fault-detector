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


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
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
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 16x16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 8x8
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
    model = SimpleCNN().to(device)
    model.eval()

    # Create a dummy input batch (batch_size=1)
    batch_size = 1
    input_shape = [3, 32, 32]
    dummy = torch.rand((batch_size, *input_shape), device=device)

    # Create FaultInjection instance with single_bit_flip_func for bit flip
    pfi = nem.single_bit_flip_func(
        model=model,
        batch_size=batch_size,
        input_shape=input_shape,
        layer_types=[nn.Conv2d],
        use_cuda=torch.cuda.is_available(),
        bits=8  # 8-bit quantization
    )
    
    # Inject a single bit flip fault
    layer_ranges = [1.0] * pfi.get_total_layers()
    
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
        total_faults=pfi.total_faults_injected
    )
    detector.register_hooks()

    with torch.no_grad():
        out_neuron_fault = pfi.corrupted_model(dummy)
    
    inj_dict, det_dict = pfi.injection_dict, detector.detected_dict
    print(f"[RESULT] Injection summary: {inj_dict}")
    print(f"[RESULT] Detection summary: {det_dict}")
    
    detected, fn, fp = get_detection_result(inj_dict, det_dict)

    return detected, fn, fp

def main():
    N_RUNS = 100 # Change if want more run

    total_detected = 0
    total_fn = 0
    total_fp = 0

    for i in range(N_RUNS):
        detected, fn, fp = injection_and_detect()
        total_detected += detected
        total_fn += fn
        total_fp += fp

        print(f"Run {i+1:3d}: detected={detected}, fn={fn}, fp={fp}")

    print("\n==== Summary after", N_RUNS, "runs ====")
    print("Total detected:", total_detected)
    print("Total FN      :", total_fn)
    print("Total FP      :", total_fp)
    
if __name__ == "__main__":
    start = time.perf_counter()     

    main()                          

    end = time.perf_counter()       
    
    # Compute time for running experiment
    elapsed = end - start
    
    print(f"\n[Time] Finished in {elapsed:.4f} seconds")
