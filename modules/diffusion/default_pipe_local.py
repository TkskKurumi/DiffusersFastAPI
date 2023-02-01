from .wrapped_pipeline import CustomPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
import torch, os
half_prec = os.environ.get("DIFFUSION_V3_PRECISION", "HALF") == "HALF"
MODEL_ID = "../Models/SDModel"
TI_PATH = "../Models/Embeddings"
VAE_PATH = os.path.join(MODEL_ID, "replaced_vae.pt")
DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
print("loading model %s" % MODEL_ID)
if(False):
    sched = DPMSolverMultistepScheduler(
        beta_schedule="scaled_linear",
        solver_order=2,
        beta_start=0.00085,
        beta_end=0.012,
    )
else:
    sched = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
if(half_prec):
    print("fp16")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        revision="fp16",
        scheduler=sched
    ).to(DEVICE)
else:
    print("fp32")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        revision="fp32",
        scheduler=sched
    ).to(DEVICE)
# pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()

pipe = CustomPipeline(pipe, ti_autoload_path=TI_PATH)
# pipe.sd_pipeline.enable_sequential_cpu_offload()