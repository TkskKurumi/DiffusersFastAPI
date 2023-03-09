import os
from ...utils.debug_vram import debug_vram
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from accelerate import cpu_offload
from accelerate.hooks import remove_hook_from_module
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from ..wrapped_pipeline import CustomPipeline
from threading import Lock
from torch.nn import Module
from os import path
from ...utils.candy import locked, print_time
from ..load_vae import load_ldm_vae_ckpt
models = {}

@torch.no_grad()
def interpolate_into(module: Module, *args):

    state_dict = {}
    sd_orig = module.state_dict()
    orig_len = len(sd_orig)
    for w, m in args:
        state = m.state_dict()
        for k, v in state.items():
            state_dict[k] = state_dict.get(k, [])
            state_dict[k].append((w, v))
        del state
    a = list(state_dict)
    b = list(sd_orig)
    assert (a==b)
    for k, cands in state_dict.items():
        wsum = 0
        vsum = 0
        for w, v in cands:
            wsum+=w
            vsum+=v*w
        state_dict[k] = vsum/wsum
    module.load_state_dict(state_dict)
    del state_dict
sched = DPMSolverMultistepScheduler(
    beta_schedule="scaled_linear",
    solver_order=2,
    beta_start=0.00085,
    beta_end=0.012,
)
_INFER_MODEL: StableDiffusionPipeline = None
MODEL_LOADER_LOCK = Lock()

def _load_model(model_id):
    if(model_id.endswith("pt")):
        model = load_pipeline_from_original_stable_diffusion_ckpt(
            model_id, 
            prediction_type="epsilon",
            scheduler_type="ddim",
            original_config_file="./v1-inference.yaml"
        )
        model.scheduler = sched
    elif(model_id.endswith(".safetensors")):
        model = load_pipeline_from_original_stable_diffusion_ckpt(
            model_id, 
            prediction_type="epsilon",
            scheduler_type="ddim",
            from_safetensors=True,
            original_config_file="./v1-inference.yaml"
        )
        model.scheduler = sched
    else:
        model = StableDiffusionPipeline.from_pretrained(model_id, scheduler=sched, torch_dtype=torch.float16)
    return model

def load_model(name, model_id="", vae=""):
    with locked(MODEL_LOADER_LOCK):
        global _INFER_MODEL
        if(not model_id):
            model_id = name
        # save vram
        print("loading model %s"%name)
        # debug_vram("before load model %s"%name)
        model = _load_model(model_id).to("cpu")
        model.enable_xformers_memory_efficient_attention()
        if(_INFER_MODEL is None):
            print("loading model %s as base model"%name)
            _INFER_MODEL = _load_model(model_id).to("cuda").to(torch.float16)
            # _INFER_MODEL = StableDiffusionPipeline.from_pretrained(model_id, scheduler=sched, torch_dtype=torch.float16).to("cuda")
            _INFER_MODEL.enable_attention_slicing()
            _INFER_MODEL.enable_xformers_memory_efficient_attention()
            device = "cuda:0"
        if(path.exists(vae)):
            load_ldm_vae_ckpt(model.vae, vae)
        # debug_vram("after load model %s"%name)
        models[name] = model
        return model
        

MASTER_MODEL: CustomPipeline = None
TI_PATH = "../Models/Embeddings"
LR_PATH = "../Models/LoRA"
LOADED_VAES = None
LOADED_UNETS = None
def get_model(unets=None, vaes=None):
    with locked(MODEL_LOADER_LOCK), print_time("get model"):
        global MASTER_MODEL, LOADED_UNETS, LOADED_VAES
        if(MASTER_MODEL is None):
            if(unets is None):
                unets = [(1, i) for i in models]
            if(vaes is None):
                vaes = [(1, i) for i in models]
            if((not unets)or(not vaes)):
                print("no model loaded")
                return MASTER_MODEL
            MASTER_MODEL = CustomPipeline(_INFER_MODEL, TI_PATH, LR_PATH)
            # debug_vram("before interp")
            with locked(MASTER_MODEL.hijack_lock):
                print("interpolating models")
                LOADED_UNETS = unets
                LOADED_VAES = vaes
                unets = [(w, models[i].unet) for w, i in unets]
                vaes = [(w, models[i].vae) for w, i in vaes]
                interpolate_into(MASTER_MODEL.sd_pipeline.unet, *unets)
                interpolate_into(MASTER_MODEL.sd_pipeline.vae, *vaes)
            # debug_vram("after interp")
        elif(unets or vaes):
            assert isinstance(MASTER_MODEL, CustomPipeline)
            # debug_vram("before interp")
            with locked(MASTER_MODEL.hijack_lock):
                print("interpolating models")
                if(unets and unets!=LOADED_UNETS):
                    LOADED_UNETS = unets
                    unets = [(w, models[i].unet) for w, i in unets]
                    interpolate_into(MASTER_MODEL.sd_pipeline.unet, *unets)
                if(vaes and vaes!=LOADED_VAES):
                    LOADED_VAES = vaes
                    vaes = [(w, models[i].vae) for w, i in vaes]
                    
                    interpolate_into(MASTER_MODEL.sd_pipeline.vae, *vaes)
            debug_vram("after interp")
        p = MASTER_MODEL.sd_pipeline
        cnt = 0
        # for model in [p.text_encoder, p.safety_checker]:
        #     if(model is not None):
        #         if(str(model.device)=="cuda:0"):
        #             cnt += 1
        #             cpu_offload(model, "cuda:0")
        # print("offload %d models to cpu"%cnt)
        print(LOADED_UNETS, LOADED_VAES)
        return MASTER_MODEL
class PipeDummy:
    def __getattr__(self, name):
        return getattr(get_model(), name)


model_ids = os.environ.get("DIFFUSION_MODELS", "").split(",")
for i in model_ids:
    if(i):
        i = i.split(":")
        load_model(*i)


pipe = PipeDummy()

# default_unets = [(40, "PastelMix"), (20, "Counterfeit"), (20, "CF2.2"), (10, "Eimis"), (5, "Basil"), (5, "AOM")]
# default_vaes = [(1, "AOM"), (9, "PastelMix")]
# default_unets = [(w, k) for w, k in default_unets if k in models] or None
# default_vaes = [(w, k) for w, k in default_vaes if k in models] or None
# get_model(default_unets, default_vaes)
# if(__name__=="__main__"):
#     MASTER_MODEL.sd_pipeline.save_pretrained("../Models/KurumiMix")