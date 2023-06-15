from torch import nn
import torch
from typing import List
from ...utils.load_tensor import load as load_tensor
from ...utils.misc import DEVICE as DEFAULT_DEVICE

class LoRALayer(nn.Module):
    def __init__(self, orig_module, channels, alpha=1):
        super().__init__()

        if (orig_module.__class__.__name__ == "Conv2d"):
            in_channels = orig_module.in_channels
            out_channels = orig_module.out_channels
            self.lora_down = nn.Conv2d(
                in_channels, channels, (1, 1), bias=False)
            self.lora_up = nn.Conv2d(
                channels, out_channels, (1, 1), bias=False)
        elif (orig_module.__class__.__name__ == "Linear"):
            in_ch = orig_module.in_features
            out_ch = orig_module.out_features
            self.lora_down = nn.Linear(in_ch, channels, bias=False)
            self.lora_up = nn.Linear(channels, out_ch, bias=False)
        else:
            raise TypeError(type(orig_module))
        self.scale = alpha/channels

    def forward(self, x):
        return self.lora_up(self.lora_down(x))*self.scale


class WrapLoRAModule(nn.Module):
    def __init__(self, orig_module, layers: nn.ModuleList | List[LoRALayer], lora_scales=None, torch_dtype=torch.float16):
        super().__init__()
        self.orig_module = orig_module
        self.orig_forward = orig_module.forward
        if (isinstance(layers, list)):
            self.layers = nn.ModuleList(layers)
        elif (isinstance(layers, nn.ModuleList)):
            self.layers = layers
        else:
            raise TypeError(type(layers))
        if (lora_scales is None):
            self.lora_scales = [1 for i in layers]
        else:
            self.lora_scales = lora_scales

    def forward(self, x):
        ret = self.orig_forward(x)
        for idx, i in enumerate(self.layers):
            ret = ret + i(x)*self.lora_scales[idx]
        return ret

    def _enter(self):
        self.orig_forward = self.orig_module.forward
        self.orig_module.forward = self.forward


    def _exit(self):
        self.orig_module.forward = self.orig_forward

    # def __enter__(self):
    #     self.orig_module.forward = self.forward

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.orig_module.forward = self.orig_forward


LORA_UNET_PREFIX = "lora_unet"
LORA_TE_PREFIX = "lora_te"
class Empty:
    def __enter__(self):
        pass
    def __exit__(self, _, __, ___):
        return

class WrapLoRA:
    def __init__(self, orig_unet: nn.Module, lora_sd, lora_scales, torch_dtype=torch.float16, device=DEFAULT_DEVICE, prefix=LORA_UNET_PREFIX):
        self.orig_unet = orig_unet
        self.lora_modules = []
        for name, module in orig_unet.named_modules():
            if (module.__class__.__name__ not in ["Linear", "Conv2d"]):
                continue
            lora_weight_name = prefix+"_"+name.replace(".", "_")
            # print(name, module.__class__, lora_weight_name)
            lora_layers = []
            scales = []
            for idx, i in enumerate(lora_sd):
                has_down = lora_weight_name+".lora_down.weight" in i
                has_up = lora_weight_name+".lora_up.weight" in i
                has_alpha = lora_weight_name+".alpha" in i
                if (has_down and has_up):
                    down_weight = i[lora_weight_name+".lora_down.weight"]
                    up_weight = i[lora_weight_name+".lora_up.weight"]
                    chdown = down_weight.shape[0]
                    chup = up_weight.shape[1]
                    assert chdown == chup
                    alpha = i[lora_weight_name+".alpha"] if has_alpha else 1
                    layer = LoRALayer(module, chup, alpha=alpha)
                    if (layer.lora_down.weight.shape != down_weight.shape):
                        print("??? lora down weight shape mismatch for layer", name)
                        continue
                    if (layer.lora_up.weight.shape != up_weight.shape):
                        print("??? lora up   weight shape mismatch for layer", name)
                        continue
                    layer.lora_down.weight = nn.Parameter(
                        down_weight.to(torch_dtype).to(device))
                    layer.lora_up.weight = nn.Parameter(
                        up_weight.to(torch_dtype).to(device))

                    lora_layers.append(layer)
                    scales.append(lora_scales[idx])
            if (lora_layers):
                lora_module = WrapLoRAModule(module, lora_layers, scales)
                self.lora_modules.append(lora_module)
        if (not self.lora_modules):
            print("no lora loaded")
        else:
            print("injected %d layers" % len(self.lora_modules))

    def __enter__(self):
        for i in self.lora_modules:
            i._enter()

    def __exit__(self, _, __, ___):
        for i in self.lora_modules[::-1]:
            i._exit()


class WrapLoRAUNet(WrapLoRA):
    def __init__(self, *args, **kwargs):
        WrapLoRA.__init__(self, *args, **kwargs, prefix=LORA_UNET_PREFIX)


class WrapLoRATextEncoder(WrapLoRA):
    def __init__(self, *args, **kwargs):
        WrapLoRA.__init__(self, *args, **kwargs, prefix=LORA_TE_PREFIX)


if (__name__ == "__main__"):
    import os
    os.environ["DIFFUSION_MODEL"] = ""
    from . import model_mgr
    model = model_mgr.load_model("AOM", r"E:\Diffusion-FastAPI\Models\AOM",
                                 r"E:\Diffusion-FastAPI\Models\AOM\orangemix.vae.pt")
    # model = model_mgr.load_model("Cillout", "E:\Diffusion-FastAPI\Models\LdmModels\chilloutmix_NiPrunedFp16.safetensors")
    # model = model_mgr.load_model("WD1.4", r"E:\Diffusion-FastAPI\Models\LdmModels\wd-1-4-anime_e2.ckpt")
    model = model_mgr.get_model()
    # lora_sd = load_tensor(r"E:\Diffusion-FastAPI\Models\LoRA\koreanDollLikeness_v10.safetensors")
    lora_sd_slingshot = load_tensor(
        r"E:\Diffusion-FastAPI\Models\LoRA\SlingShot.safetensors")
    lora_sd_ontray = load_tensor(
        r"E:\Diffusion-FastAPI\Models\LoRA\BreatsOnTray.safetensors")
    states = [lora_sd_slingshot, lora_sd_ontray]
    scales = [1, 0]
    torch.manual_seed(998244353)
    noise = torch.randn((4, 640//8, 512//8)).cpu().numpy()
    with WrapLoRAUNet(model.sd_pipeline.unet, states, scales):
        with WrapLoRATextEncoder(model.sd_pipeline.text_encoder, states, scales):
            repro = model.txt2img("extremely delicate and beautiful girl, slingshot swimsuit/*worst quality, EasyNegative*/",
                                  width=512, height=640, steps=40, cfg=12, use_noise=noise)
            repro.result.save("test_lora_1.png")
    again = repro.reproduce()
    again.result.save("test_lora_0.png")
else:
    pass
