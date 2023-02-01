from torch.nn import Module
from diffusers import StableDiffusionPipeline
import torch
def merge_into(module: Module, *args):
    state_dict = {}
    sd_orig = module.state_dict()
    orig_len = len(sd_orig)
    for w, m in args:
        state = m.state_dict()
        for k, v in state.items():
            state_dict[k] = state_dict.get(k, [])
            state_dict[k].append((w, v))
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
# A = StableDiffusionPipeline.from_pretrained("Linaqruf/anything-v3.0", torch_dtype=torch.float16)
# B = StableDiffusionPipeline.from_pretrained("andite/pastel-mix", torch_dtype=torch.float16)
# merge_into(A.unet, (1, A.unet), (1, B.unet))