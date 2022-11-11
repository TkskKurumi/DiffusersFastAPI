from .wrapped_pipeline import CustomPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch, os
half_prec = os.environ.get("DIFFUSION_V3_PRECISION", "FULL") == "FULL"
MODEL_ID = "../Models/SDModel"
TI_PATH = "../Models/Embeddings"
DEVICE = "cuda"

print("loading model %s" % MODEL_ID)
if(half_prec):
    pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    revision="fp16",
    scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    ),
    ).to(DEVICE)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    revision="fp32",
    scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    ),
    ).to(DEVICE)
pipe.enable_attention_slicing()

pipe = CustomPipeline(pipe, ti_autoload_path=TI_PATH)