from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

import modal

MINUTES = 60  # seconds

app = modal.App("chat-with-pdf")


CACHE_DIR = "/hf-cache"

model_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        [
            "git+https://github.com/illuin-tech/colpali.git@782edcd50108d1842d154730ad3ce72476a2d17d",  # we pin the commit id
            "hf_transfer==0.1.8",
            "qwen-vl-utils==0.0.8",
            "torchvision==0.19.1",
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": CACHE_DIR})
)


# These dependencies are only installed remotely, so we can’t import them locally. Use the .imports context manager to import them only on Modal instead.

with model_image.imports():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
