import gradio as gr
import numpy as np
#import spaces
from diffusers import  DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderTiny, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
#from live_preview_helpers import calculate_shift, retrieve_timesteps, flux_pipe_call_that_returns_an_iterable_of_images
import modal

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 640

examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 [dev]
12B param rectified flow transformer guidance-distilled from [FLUX.1 [pro]](https://blackforestlabs.ai/)  
[[non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)] [[blog](https://blackforestlabs.ai/announcing-black-forest-labs/)] [[model](https://huggingface.co/black-forest-labs/FLUX.1-dev)]
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(label="Result", show_label=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )
            
            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=3.5,
                )
  
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )
        
        f = modal.Function.from_name("live-preview-test", "infer")
        gr.Examples(
            examples = examples,
            fn = f.remote_gen,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )
    # def generate(prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    #     f = modal.Function.from_name("live-preview-test", "infer")
    #     # Import the remote function
    #     result, seed = f.remote(prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)
    #     return result, seed 
    
    
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = f.remote_gen,
        inputs = [prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch()