from .wrapped_pipeline import CustomPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch, os
import numpy as np
half_prec = os.environ.get("DIFFUSION_V3_PRECISION", "FULL") == "FULL"
MODEL_ID = "../Models/SDModel"
TI_PATH = "../Models/Embeddings"
DEVICE = "cuda"

cumalpha = np.linspace(0.999**0.5, 0.001**0.5, 1000)**2
divisor = cumalpha.copy()
divisor[1:] = divisor[:-1]
divisor[0] = 1
alphas = cumalpha/divisor
betas = 1-alphas
if(os.environ.get("LINEAR_ALPHA", "NO") == "NO"):
    
    scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        num_train_timesteps=1000
    )
else:
    print("DEBUG: linear alpha")
    scheduler=DDIMScheduler(
        trained_betas=betas,
        clip_sample=False,
        set_alpha_to_one=False,
        num_train_timesteps=1000
    )

sched = DDIMScheduler

print("loading model %s" % MODEL_ID)
if(half_prec):
    pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    revision="fp16",
    scheduler=DDIMScheduler(
        trained_betas=betas,
        clip_sample=False,
        set_alpha_to_one=True,
    ),
    ).to(DEVICE)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision="fp32",
    scheduler=DDIMScheduler(
        trained_betas=betas,
        clip_sample=False,
        set_alpha_to_one=True
    )
    ).to(DEVICE)
pipe.enable_attention_slicing()

pipe = CustomPipeline(pipe, ti_autoload_path=TI_PATH)
