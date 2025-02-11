import torch
print(f"Current device: {torch.cuda.current_device()}")  # Shows current GPU device number
print(f"Device being used: {next(model.parameters()).device}")  # Shows device for a specific model
print(f"Is CUDA available? {torch.cuda.is_available()}")  # Checks if CUDA is available

import tensorflow as tf
print(f"Devices available: {tf.config.list_physical_devices()}")
print(f"Using GPU: {tf.test.is_gpu_available()}")

import psutil
import platform

print(f"CPU count: {psutil.cpu_count()}")
print(f"CPU info: {platform.processor()}")

# Requires nvidia-smi
import subprocess
try:
    print(subprocess.check_output(['nvidia-smi']).decode())
except:
    print("nvidia-smi not available")

    