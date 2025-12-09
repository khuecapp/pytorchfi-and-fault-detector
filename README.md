🧪 Fault Injection & Detection Experiments using PyTorchFI

This repository contains example code and experiments for testing fault detection algorithms using PyTorchFI, a fault-injection framework for PyTorch-based neural networks.

The main purpose of this repo is to:

Inject neuron-level or weight-level faults into CNN models

Evaluate the detection accuracy of custom fault detectors

Provide demo scripts and utilities for quick experimentation

📌 Features

Simple CNN models for demonstration

Support for single-bit and multi-bit neuron fault injection

Automatic fault detection using feature-map hooks

Utilities for evaluating:

Detected faults (TP)

False negatives (FN)

False positives (FP)

Ready-to-run experiment script

🚀 Getting Started
1. Install dependencies
pip install torch torchvision
pip install pytorchfi

2. Run the experiment demo

From the repo root directory, execute:

bash -lc "python examples/demo_pytorchfi.py"


Or simply:

python examples/demo_pytorchfi.py


This script will:

Load the CNN model

Inject random neuron bit-flip faults

Run the fault detector

Print injection & detection summary

Compute detected / FN / FP statistics

📁 Repository Structure
.
├── examples/
│   ├── demo_pytorchfi.py     # Main experiment demo
│   └── ...
├── pytorchfi/                # PyTorchFI framework (if included as submodule)
├── README.md
└── requirements.txt

📝 Usage Notes

demo_pytorchfi.py contains two CNN models and shows how to:

Initialize PyTorchFI

Inject random multi-bit neuron faults

Register detection hooks

Compare injected vs. detected faults

Modify run_once() or loop calls inside main() to perform repeated experiments (e.g., 100 runs).