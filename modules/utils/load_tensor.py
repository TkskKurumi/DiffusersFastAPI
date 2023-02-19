from safetensors import safe_open
import torch

def load(pth, device="cpu"):
    if(pth.endswith("pt")):
        return torch.load(pth)
    else:
        ret = {}
        with safe_open(pth, framework="pt", device=device) as f:
            for k in f.keys():
                ret[k] = f.get_tensor(k)
        return ret