# $A="NAI:../Models/SDModel"
# $B="AOM:../Models/AOM:../Models/AOM/orangemix.vae.pt"
import os
os.environ["DIFFUSION_MODELS"]=",".join([
    "CF:..\Models\LdmModels\Counterfeit-V2.5_fp16.safetensors",
    "AOM:../Models/AOM:../Models/AOM/orangemix.vae.pt"
])
from modules.diffusion import model_mgr 
from torch import nn
import torch
import numpy as np
from functools import partial
unet: nn.Module = model_mgr.get_model().sd_pipeline.unet
unet_keys = unet.state_dict()
print(len(list(unet_keys)), "params")
# print(list(unet_keys))
tmp = []
def _prt(ls, *args, sep=" ", end="\n"):
    args = [str(i) for i in args]
    ls.append(sep.join(args)+end)
prt=partial(_prt, tmp)
prt("digraph {")
prt("    node [shape=record]")
edges = set()
for i in unet_keys:
    splits = i.split(".")
    for jdx, j in enumerate(splits):
        if(jdx):
            A = "_".join(splits[:jdx])
            B = "_".join(splits[:jdx+1])
            if((A, B) not in edges):
                edges.add((A, B))
                prt("   ", A, "->", B)
prt("}")
with open("weights.dot", "w") as f:
    print("".join(tmp), file=f)

layer2params = {}

for i in unet_keys:
    splits = i.split(".")
    if("to_k" in i or "to_q" in i or "to_v" in i or "to_out" in i):
        meow = 2
    else:
        meow = 1
    weight = i
    layer = ".".join(splits[:-meow])
    layer2params[layer] = layer2params.get(layer, [])
    layer2params[layer].append(weight)
for k, v in layer2params.items():
    print(k)


base_model = 'AOM'
tar_model = "CF"
base_weights = model_mgr.models[base_model].unet.state_dict()
tar_weights = model_mgr.models[tar_model].unet.state_dict()
assert model_mgr.models[base_model].unet is not model_mgr.MASTER_MODEL.sd_pipeline.unet
assert model_mgr.models[tar_model].unet is not model_mgr.MASTER_MODEL.sd_pipeline.unet
import re
selected_layers_by = [".*norm"]
tar_directory = "AOM2CF_norm"
selected_layers = set()
for i in base_weights:
    for j in selected_layers_by:
        if(re.match(j, i)):
            selected_layers.add(i)
            break
print(selected_layers)
def interpolate_model(rate):
    state_dict = {}
    anything_changed = False
    
    for key in base_weights:
        if(key in selected_layers):
            if(not anything_changed):
                diff = tar_weights[key]-base_weights[key]
                diff_any = np.abs(diff.cpu().numpy())>1e-6
                if(diff_any.any()):anything_changed = True

            weight = base_weights[key]*(1-rate)+tar_weights[key]*rate
            state_dict[key] = weight
        else:
            state_dict[key] = base_weights[key]
    assert(anything_changed), "models are the same in selected layers"
    model_mgr.MASTER_MODEL.sd_pipeline.unet.load_state_dict(state_dict)

torch.manual_seed(114514)
noise = torch.randn((4, 512//8, 512//8)).cpu().numpy()
prompt = "1girl, pink hair, cat ears, best quality"
neg = "bad anatomy, worst quality"
steps = 20

os.makedirs(tar_directory, exist_ok=True)
for i in range(steps+1):
    r = i/steps
    interpolate_model(r)
    im = model_mgr.MASTER_MODEL.txt2img(prompt, width=512, height=512, use_noise=noise, neg_prompt=neg)
    im.result.save(os.path.join(tar_directory, "%03d.png"%i))

wid, hei = im.result.size
target = 5<<20
rate = (target/(wid*hei*steps*0.12))**0.5
rate = min(1, rate)
while(True):
    w, h = wid*rate, hei*rate
    input_images = os.path.join(tar_directory, "*.png")
    output_path = os.path.join(tar_directory, "output.gif")
    ret = os.system("gifski %s --width %d --height %d --fps 8 -o %s"%(input_images, w, h, output_path))
    if(ret!=0):
        print("using gifski to make gif but seem it's not installed")
    cur = os.path.getsize(output_path)
    if(cur<target):
        break
    else:
        rate *= 0.97
        rate *= (target/cur)**0.5
print(output_path)
print(tar_directory)