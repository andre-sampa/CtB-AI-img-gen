# config.py
import os
from config.prompts import prompts  # Import prompts from prompts.py
from config.models import models

# Retrieve the Hugging Face token
api_token = os.getenv("HF_TOKEN")

# Debugging: Print prompt and model options
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])

