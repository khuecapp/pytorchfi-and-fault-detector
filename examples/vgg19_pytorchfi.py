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
import torchvision.models as models


#vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1) # pretrained model
vgg19 = models.vgg19(weights=None) # untrained model
print(vgg19)

def main():
    # Prepare model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg19.to(device)
    model.eval()

    # Create a dummy input batch (batch_size=1)
    batch_size = 1
    input_shape = [3, 224, 224] #alexnet input size
    dummy = torch.randn((batch_size, *input_shape), device=device)

    # # Run original model
    # with torch.no_grad():
    #     orig_out = model(dummy)

    # print("Original output (first 5 values):", orig_out[0, :5].cpu().numpy())

    # Create FaultInjection instance with single_bit_flip_func for bit flip
    pfi = nem.single_bit_flip_func(
        model,
        batch_size=batch_size,
        input_shape=input_shape,
        layer_types=[nn.Conv2d],
        use_cuda=torch.cuda.is_available(),
        bits=8  # 8-bit quantization
    )

    print("Injecting a single bit flip fault...\n")
    # 1) Inject a single bit flip fault
    # Create layer_ranges (max values for each layer)
    layer_ranges = [1.0] * pfi.get_total_layers()
    nem.random_neuron_single_bit_inj(pfi, layer_ranges)
    
    print(pfi.print_pytorchfi_layer_summary())
    print("\n")
    
    #Setup Fault Detector
    detector = fd.FaultDetector(
        model=pfi.corrupted_model,
        layer_types=[nn.Conv2d],
        use_cuda=torch.cuda.is_available(),
        remove_bias=True
    )

    print("\nDetecting faults with FaultDetector...")
    detector.register_hooks()

    with torch.no_grad():
        out_neuron_fault = pfi.corrupted_model(dummy)
    
    print("Output with bit flip fault (first 5 values):", out_neuron_fault[0, :5].cpu().numpy())
    print("\n")
    
    # Print detection results
    print("\n")
    print(detector.print_detection_detailed_summary())
    
if __name__ == "__main__":
    main()
