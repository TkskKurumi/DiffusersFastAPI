from .weight_merging import merge_into
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch, os
from .wrapped_pipeline import CustomPipeline
from .load_vae import load_ldm_vae_ckpt
from os import path
TI_PATH = "../Models/Embeddings"

# DIFFUSION_MODEL=merge:weightA:idA:vaeA,weightB,idB:vaeB...
model_ids = os.environ.get("DIFFUSION_MODEL")
model_ids = ":".join(model_ids.split(":")[1:])
model_ids = [i.split(":") for i in model_ids.split(",")]
unets = []
vaes = []

sched = DPMSolverMultistepScheduler(
    beta_schedule="scaled_linear",
    solver_order=2,
    beta_start=0.00085,
    beta_end=0.012,
)

for i in model_ids:
    i = [j.strip() for j in i]
    if (len(i) == 2):
        w, model_id = i
        w_vae = w
        vae_pth = ""
    elif(len(i) == 4):
        w, model_id, w_vae, vae_pth = i
    else:
        raise Exception("Unrecoginzed model %s"%i)
    w, w_vae = float(w), float(w_vae)
    model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=sched)

    if(path.exists(vae_pth)):
        load_ldm_vae_ckpt(model.vae, vae_pth)
    unets.append((w, model.unet))
    vaes.append((w_vae, model.vae))

merge_into(model.unet, *unets)
merge_into(model.vae, *vaes)
pipe = model
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")
pipe = CustomPipeline(pipe, ti_autoload_path=TI_PATH)