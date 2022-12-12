from .wrapped_pipeline import CustomPipeline
from diffusers import StableDiffusionPipeline
import torch
TI_PATH = "../Models/Embeddings"
model_id = "Linaqruf/anything-v3.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")
pipe = CustomPipeline(pipe, ti_autoload_path=TI_PATH)