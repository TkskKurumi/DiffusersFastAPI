from .wrapped_pipeline import CustomPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
TI_PATH = "../Models/Embeddings"
model_id = "andite/anything-v4.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        num_train_timesteps=1000
    ))
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")
pipe = CustomPipeline(pipe, ti_autoload_path=TI_PATH)