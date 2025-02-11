print("-----LOADING QUANTA-----")
ckpt_path = (
    "/data/FLUX.1-dev-gguf/flux1-dev-Q8_0.gguf"
)
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)


print("-----INITIALIZING PIPE-----")
pipe = FluxPipeline.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    transformer=transformer,
    #torch_dtype=torch.float16,
    #torch_dtype=torch.float32,
    #vae=taef1,
    local_files_only=True,
)
#torch.cuda.empty_cache()