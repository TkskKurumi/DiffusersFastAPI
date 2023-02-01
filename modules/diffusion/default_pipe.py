import os
if(os.environ.get("DIFFUSION_MODEL") == "ANYTHING_V3"):
    from .default_pipe_anything_V3 import pipe
elif(os.environ.get("DIFFUSION_MODEL") == "ANYTHING_V4"):
    from .default_pipe_anything_V4 import pipe
elif(os.environ.get("DIFFUSION_MODEL") == "LOCAL"):
    from .default_pipe_local import pipe
elif(os.environ.get("DIFFUSION_MODEL", "").startswith("merge")):
    from .default_pipe_merged import pipe
else:
    from .default_pipe_diffusers import pipe
VAE_PATH = os.environ.get("DIFFUSION_VAE", "")
if(os.path.exists(VAE_PATH)):
    from .load_vae import load_ldm_vae_ckpt
    load_ldm_vae_ckpt(pipe.sd_pipeline.vae, VAE_PATH)
if True:
    from accelerate import cpu_offload
    raw_pipe = pipe.sd_pipeline
    raw_pipe.enable_xformers_memory_efficient_attention()
    device = "cuda:0"
    for model in [raw_pipe.vae, raw_pipe.text_encoder, raw_pipe.safety_checker]:
        if(model is not None):
            cpu_offload(model, device)