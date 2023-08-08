from modules.diffusion.model_mgr import model_mgr
from modules.utils.load_tensor import load as load_tensor
from tqdm import trange, tqdm
import numpy as np
from torch import Tensor
from safetensors.torch import save_file
import torch
def lora_compression(matrix: np.ndarray, rank, iter=3, dtype=np.float16, verbose=False):
    shape = matrix.shape
    if(len(shape)==4):
        n, m = shape[:2]
        matrix = matrix.reshape((n, m))
        is_conv = True
    else:
        n, m = matrix.shape
        is_conv = False
    A = np.zeros((n, rank), dtype=dtype)
    B = np.zeros((rank, m), dtype=dtype)

    def solve_B(residual, _A):
        A2 = (_A*_A).sum()
        WA = (residual*_A).sum(axis=0, keepdims=True)
        ret = WA/A2
        ret[np.isnan(ret)] = 0
        return ret
    def solve_A(residual, _B):
        B2 = (_B*_B).sum()
        WB = (residual*_B).sum(axis=1, keepdims=True)
        ret = WB/B2
        ret[np.isnan(ret)] = 0
        return ret
    residual = matrix
    if(verbose):
        iterator = trange(rank)
    else:
        iterator = range(rank)
    for i in iterator:
        if(verbose):
            res_norm2 = np.sqrt((residual*residual).mean())
            iterator.set_postfix(shape=shape, residual = res_norm2)
            print("rank %d, residual: %.3f"%(i, res_norm2))
        # iterator.set_postfix(residual=res_norm2, shape=shape)
        _A = np.random.normal(size=(n, 1))
        # _A = residual.mean(axis=0, keepdims=True)
        for j in range(0, iter):
            if(j%2):
                _A = solve_A(residual, _B)
            else:
                _B = solve_B(residual, _A)
        A[:, i:i+1] = _A
        B[i:i+1, :] = _B
        delta = np.matmul(_A, _B)
        residual = residual - delta
    res_norm2 = np.sqrt((residual*residual).mean())
    if (is_conv):
        A = A.reshape((n, rank, 1, 1))
        B = B.reshape((rank, m, 1, 1))
    return A, B, res_norm2


def extract_from_dict(module_name, rank, base_sd, tar_sd, dtype=np.float16):
    key = module_name+".weight"
    delta = tar_sd[key]-base_sd[key]
    delta_np = delta.cpu().numpy()
    lora_up, lora_down, residual = lora_compression(delta_np, rank, dtype=dtype)
    return lora_up, lora_down, residual

def extract_from_models(base_model, tar_model, rank=None, rank_alpha=0.05, dtype=np.float16):
    ret = {}
    def is_suitable_for_lora(submodule):
        classname = submodule.__class__.__name__
        if (classname == "Linear"):
            return True
        elif (classname == "Conv2d" and submodule.kernel_size==(1, 1)):
            return True
        return False
    def extract_sub_model(prefix, bmodel, tmodel):
        base_sd = bmodel.state_dict()
        tar_sd = tmodel.state_dict()
        ls = [(name, module) for name, module in bmodel.named_modules() if is_suitable_for_lora(module)]
        iterator = tqdm(ls)
        for name, module in iterator:
            if (is_suitable_for_lora(module)):
                if(rank is None):
                    w, h = base_sd[name+".weight"].shape[:2]
                    _rank = w*h/(w+h)*rank_alpha
                    _rank = int(max(_rank, 1))
                else:
                    _rank = rank
                lora_up, lora_down, res = extract_from_dict(name, _rank, base_sd, tar_sd, dtype=dtype)
                key = prefix+"_"+name.replace(".", "_")
                ret[key+'.lora_up.weight'] = torch.from_numpy(lora_up)
                ret[key+'.lora_down.weight'] = torch.from_numpy(lora_down)
                ret[key+".alpha"] = torch.as_tensor(_rank)
                iterator.set_postfix(module=name, rank=_rank, residual=res)
    extract_sub_model("lora_unet", base_model.unet, tar_model.unet)
    extract_sub_model("lora_te", base_model.text_encoder, tar_model.text_encoder)
    return ret

if (__name__=="__main__"):
    if(False):
        with open("lora0.log", "w") as f:
            lora = load_tensor(r"E:\Diffusion-FastAPI\Models\LoRA\榨乳v1.4.safetensors")
            for key, val in lora.items():
                print(key, val.shape, file=f)
        with open("lora1.log", "w") as f:
            lora = load_tensor(r"temp.safetensors")
            for key, val in lora.items():
                print(key, val.shape, file=f)
        
    else:
        sdbase = r"D:\StableDiffusion\BaseModels\RMHF\RMHF-Anime-V4\rmhf.safetensors"
        model_base = model_mgr.load_model("base", sdbase)
        rank_alpha = 0.25
        for model, model_name in [
            (r"D:\StableDiffusion\BaseModels\Realistic-2.5D\fantexiV09beta_fantexiV09beta.ckpt", "fantexi"),
            (r"D:\StableDiffusion\BaseModels\Anime\Cute\cuteyukimixAdorable_neochapter2.safetensors", "cute_yuki"),
            # (r"D:\StableDiffusion\BaseModels\Anime\Cute\cocotifacute_v20.safetensors", "cococute"),
            # (r"D:\StableDiffusion\BaseModels\Anime\Cute\nextgenmix_r28Bakedvae.safetensors", "next_gen"),
            # (r"D:\StableDiffusion\BaseModels\Anime\blazingDrive_V02.safetensors", "BlazingDrive"),
            
            
        ]:
            model_tar = model_mgr.load_model(model_name, model)
            
            lora = extract_from_models(model_base, model_tar, rank=None, rank_alpha=rank_alpha, dtype=np.float32)

            save_file(lora, "%s_ada_%d%%_fp32.safetensors"%(model_name, rank_alpha*100))