import os
if(os.environ.get("DIFFUSION_MODEL") == "ANYTHING_V3"):
    from .default_pipe_anything_V3 import pipe
elif(os.environ.get("DIFFUSION_MODEL") == "ANYTHING_V4"):
    from .default_pipe_anything_V4 import pipe
else:
    from .default_pipe_old import pipe
if True:
    from accelerate import cpu_offload
    raw_pipe = pipe.sd_pipeline
    device = "cuda:0"
    for model in [raw_pipe.vae, raw_pipe.text_encoder, raw_pipe.safety_checker]:
        cpu_offload(model, device)