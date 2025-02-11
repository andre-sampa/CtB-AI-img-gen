import time
from io import BytesIO
from pathlib import Path
import modal

flux_image = (
        cuda_dev_image.apt_install(
            "git",
            "libglib2.0-0",
            "libsm6",
            "libxrender1",
            "libxext6",
            "ffmpeg",
            "libgl1",
        )
        .pip_install(
            "invisible_watermark==0.2.0",
            "transformers==4.44.0",
            "huggingface_hub[hf_transfer]==0.26.2",
            "accelerate==0.33.0",
            "safetensors==0.4.4",
            "sentencepiece==0.2.0",
            "torch==2.5.0",
            f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
            "numpy<2",
        )
        #.env({"HF_TOKEN": "1", "HF_HUB_CACHE_DIR": "/cache"})
    )


    # flux_image = flux_image.env(
    #     {
    #         "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
    #         "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    #     }
    # )



    with flux_image.imports():
        import torch
        from diffusers import FluxPipeline

    MINUTES = 60  # seconds
    VARIANT = "schnell"  # or "dev", but note [dev] requires you to accept terms and conditions on HF
    NUM_INFERENCE_STEPS = 40  # use ~50 for [dev], smaller for [schnell]


app = modal.App("example-flux", image=flux_image)

@app.local_entrypoint()
def main ():
    cuda_version = "12.4.0"  # should be no greater than host CUDA version
    flavor = "devel"  # includes full CUDA toolkit
    operating_sys = "ubuntu22.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"

    cuda_dev_image = modal.Image.from_registry(
        f"nvidia/cuda:{tag}", add_python="3.11"
    ).entrypoint([])



    diffusers_commit_sha = "81cf3b2f155f1de322079af28f625349ee21ec6b"

    
    @app.cls(
        gpu="H100",  # fastest GPU on Modal
        container_idle_timeout=20 * MINUTES,
        timeout=60 * MINUTES,  # leave plenty of time for compilation
        volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
            "/cache": modal.Volume.from_name(
                "hf-hub-cache", create_if_missing=True
            ),
            "/root/.nv": modal.Volume.from_name("nv-cache", create_if_missing=True),
            "/root/.triton": modal.Volume.from_name(
                "triton-cache", create_if_missing=True
            ),
            "/root/.inductor-cache": modal.Volume.from_name(
                "inductor-cache", create_if_missing=True
            ),
        },
    )
    class Model:
        compile: int = (  # see section on torch.compile below for details
            modal.parameter(default=0)
        )

        @modal.enter()
        def enter(self):
            pipe = FluxPipeline.from_pretrained(
                f"black-forest-labs/FLUX.1-{VARIANT}", torch_dtype=torch.bfloat16
            ).to("cuda")  # move model to GPU
            self.pipe = optimize(pipe, compile=bool(self.compile))

        @modal.method()
        def inference(self, prompt: str) -> bytes:
            print("🎨 generating image...")
            out = self.pipe(
                prompt,
                output_type="pil",
                num_inference_steps=NUM_INFERENCE_STEPS,
            ).images[0]

            byte_stream = BytesIO()
            out.save(byte_stream, format="JPEG")
            return byte_stream.getvalue()

