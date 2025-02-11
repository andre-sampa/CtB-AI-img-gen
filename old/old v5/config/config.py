# config.py
import os
from config.prompts import prompts  # Direct Import prompts from prompts.py
from config.models import models, models_modal # Direct Import models

# Retrieve the Hugging Face token
api_token = os.getenv("HF_TOKEN")

# Debugging: Print prompt and model options
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])


