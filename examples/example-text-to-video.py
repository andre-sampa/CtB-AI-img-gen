import string
import time
from pathlib import Path

import modal

app = modal.App()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "accelerate==1.1.1",
        "hf_transfer==0.1.8",
        "sentencepiece==0.2.0",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "git+https://github.com/huggingface/transformers@30335093276212ce74938bdfd85bfd5df31a668a",
        "git+https://github.com/huggingface/diffusers@99c0483b67427de467f11aa35d54678fd36a7ea2",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/models",
        }
    )
)


https://modal.com/docs/examples/mochi

