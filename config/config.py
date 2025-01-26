# config.py
import os
from config.prompts import prompts  # Import prompts from prompts.py
from config.models import models

# Retrieve the Hugging Face token
api_token = os.getenv("HF_TOKEN")

# Debugging: Check if the Hugging Face token is available
if not api_token:
    print("ERROR1: Hugging Face token (HF_CTB_TOKEN) is missing. Please set it as an environment variable or in Colab secrets.")
else:
    print("Hugging Face token loaded successfully.")


# Debugging: Print prompt and model options
print("Prompt Options:", [p["alias"] for p in prompts])
print("Model Options:", [m["alias"] for m in models])

