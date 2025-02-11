
# MODAL GPU's 

# REMEMBER TO SET PIPE TO CUDA
pipe = pipe.to("cuda")


#-------------CPU-LIST-------------------
# NVIDIA H100 Tensor Core GPU class.
# The flagship data center GPU of the Hopper architecture. Enhanced support for FP8 precision and a Transformer Engine that provides up to 4X faster training over the prior generation for GPT-3 (175B) models.
#BFLOAT16 BENCHMARK - 38sec (~$0.20)
gpu="H100" 

# NVIDIA A100 Tensor Core GPU class.
# The flagship data center GPU of the Ampere architecture. Available in 40GB and 80GB GPU memory configurations.
#BFLOAT16 BENCHMARK - 37sec ($0.10)
gpu="A100" #40gb
#BFLOAT16 BENCHMARK - #42sec (~$0.10) 
gpu=modal.gpu.A100(size="80GB") # 80gb - 

# NVIDIA L40S GPU class.
# The L40S is a data center GPU for the Ada Lovelace architecture. It has 48 GB of on-chip GDDR6 RAM and enhanced support for FP8 precision.
#FLOAT32 BENCHMARk - ERROR
#BFLOAT16 BENCHMARK - 28sec ($0.03)
#FLOAT16 BENCHMARK - 50sec
gpu="L40S" # 48gb 

#A mid-tier data center GPU based on the Ampere architecture, providing 24 GB of memory. 10G GPUs deliver up to 3.3x better ML training performance, 3x better ML inference performance, and 3x better graphics performance, in comparison to NVIDIA T4 GPUs.
#BFLOAT16 BENCHMARK - ERROR cuda out of memory - 44 sec (~$0.02 / $0.04)
gpu="A10G" #24gb

# NVIDIA L4 Tensor Core GPU class.
# A mid-tier data center GPU based on the Ada Lovelace architecture, providing 24GB of GPU memory. Includes RTX (ray tracing) support.
#BFLOAT16 BENCHMARK - ERROR cuda out of memory - 40sec (~$0.02)
gpu="L4" # 48gb

#NVIDIA T4 Tensor Core GPU class.
#A low-cost data center GPU based on the Turing architecture, providing 16GB of GPU memory.
#BFLOAT16 BENCHMARK - ERROR cuda out of memory - 37sec (~$0.02)
gpu="T4" 

#CPU
#BFLOAT16 BENCHMARK - ERROR out of memory - 37sec (~$0.02)
#FLOAT16 BENCHMARK - ERROR out of memory - 600sec (~$0.10)
cpu=2