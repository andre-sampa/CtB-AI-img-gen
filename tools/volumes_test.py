
import modal
import sys
import os
import random
from datetime import datetime
import random
import io

# Create a Modal app
app = modal.App("ctb-image-generator")

volume = modal.Volume.from_name("flux-model-vol")  # Reference your volume

@app.function(
    #gpu="t4",  # No GPU needed for debugging volume
    volumes={"/model": volume}  # Replace `volume` with your actual volume name if needed
)
def debug_volume():
    import os

    # Directory to check inside the container
    model_dir = "/model"

    try:
        if os.path.exists(model_dir):
            print(f"Volume successfully mounted at: {model_dir}")
            print("Files and directories in the mounted volume:")

            # List files in the mounted directory
            for root, dirs, files in os.walk(model_dir):
                print(f"Root: {root}")
                for dir_name in dirs:
                    print(f"  Directory: {dir_name}")
                for file_name in files:
                    print(f"  File: {file_name}")
        else:
            print(f"Volume not found at: {model_dir}")
            print("Ensure the volume is correctly mounted and mapped.")
    except Exception as e:
        print(f"Error while accessing volume: {e}")
