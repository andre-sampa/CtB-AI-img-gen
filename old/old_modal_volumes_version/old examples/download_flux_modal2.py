# DOWNLOADING FROM HERE KEEPS THE /MODELS/ DIRECTORY
    # WITH A SCRIPT IT GOES AWAY
    # def download_flux():
    #     from huggingface_hub import snapshot_download
    #     import transformers

    #     repo_id = "black-forest-labs/FLUX.1-schnell"
    #     local_dir = "/data/models/FLUX.1-schnell"

    #     # **FASTEST METHOD:** Use max_workers for parallel download
    #     snapshot_download(
    #         repo_id,
    #         local_dir=local_dir,
    #         revision="main",
    #         #ignore_patterns=["*.pt", "*.bin"],  # Skip large model weights
    #         max_workers=8  # Higher concurrency for parallel chunk downloads
    #     )

    #     transformers.utils.move_cache()
    #     print(f"FLUX model downloaded to {local_dir}")
    # download_flux()