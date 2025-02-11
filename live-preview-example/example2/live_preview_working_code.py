##### GOT IT FROM https://github.com/huggingface/diffusers/issues/3579


import torch
import torchvision
from PIL import Image
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "prompthero/openjourney-v4", torch_dtype=torch.float16, safety_checker=None)
pipe = pipe.to(device)
pipe.enable_attention_slicing()
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True
)

prompt = "A futuristic cityscape at sunset"

negative_prompt = "low quality"

# num_images_per_prompt=4,

def progress(step, timestep, latents):
    print(step, timestep, latents[0][0][0][0])

    with torch.no_grad():

        latents = 1 / 0.18215 * latents
        image = pipe.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = pipe.numpy_to_pil(image)

        # do something with the Images
        for i, img in enumerate(image):
            img.save(f"step_{step}_img{i}.png")


result = pipe(prompt=prompt,
              num_inference_steps=20,
              height=512, width=512,
              guidance_scale=7,
              negative_prompt=negative_prompt,
              callback=progress,
              callback_steps=5
              )

image = result.images[0]

image.save(f"outputs/cikar goster.png")
print(result.nsfw_content_detected)
