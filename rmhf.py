import random
import time, shutil
from termcolor import colored
from math import sqrt
from PIL import Image
from modules.diffusion.model_mgr import load_model
from modules.diffusion.model_mgr import model_mgr
from modules.diffusion.wrapped_pipeline import CustomPipeline
from modules.utils.diffuers_conversion import save_ldm
from modules.utils.candy import print_time
from modules.utils.make_gif import make_gif
import argparse
import json
import numpy as np
import torch
from functools import partial
from typing import Callable
import os
from tqdm import tqdm
from modules.utils.pil_layout import RichText, Column, Row
from modules.utils.pil_layout.resize import limit as limit_image_size


def softmax(vec, axis=1):
    exp = np.exp(vec)
    expsum = np.sum(exp, axis=axis, keepdims=True)
    ret = exp/expsum
    sm = np.abs(np.sum(ret, axis=axis)-1)
    assert np.all(sm<1e-8)
    return ret


def clip_std(vec, axis=1, clip=0.55, eps=1e-7):
    std = np.std(vec, axis=axis, keepdims=True)+1e-7
    target_std = np.minimum(std, clip)
    vec = vec/std*target_std
    mean = np.mean(vec, axis=axis, keepdims=True)
    vec = vec-mean+1/vec.shape[axis]
    return vec


def clip_overflow(vec, axis=1, of=0.1):
    # example: of = 0.05
    #   [-0.1, 1.1]        -> [-0.05, 1.05]
    #   [-0.1, -0.1, 1.2]  -> [-0.025, -0.025, 1.05]
    #   [-0.1, 0.55, 0.55] -> [-0.05, 0.525, 0.525]
    # ensuring sum = 1, min>-of, max<1+of
    # to ensure the sum of it is 1, just keep its mean = 1/n
    # first normalize to mean=0
    # then, if some of elements is bigger than 1+of, scale it in range; lower than -of, the same
    # calc the scaling, scale and make it mean = 1/n
    mean = np.mean(vec, axis=axis, keepdims=True)
    vec = vec-mean

    mean = 1/vec.shape[axis]
    mx = np.max(vec, axis=axis, keepdims=True)
    scale_mx = np.maximum(mx, 1+of-mean)/(1+of-mean)
    mn = np.min(vec, axis=axis, keepdims=True)
    scale_mn = np.minimum(mn, -of-mean)/(-of-mean)
    scale = np.maximum(scale_mx, scale_mn)
    return vec/scale+mean


class Interpolater:
    def __init__(self, name, dst_model, states, model_names):
        self.dst_model = dst_model
        self.states = states
        self.keys = list(states[0].keys())
        self.ch = len(self.keys)
        self.n = len(states)
        print("interpolating %d sets of %s weight, each set has %d weigths" %
              (self.n, name, self.ch))
        self.name = name
        self.model_names = model_names
        self.vec = self.rnd_vec()
        self._commited_vec = None
        self.is_slow = False
    def rnd_vec(self):
        return np.random.normal(0, 1, (self.ch, self.n))

    def zeros(self):
        return np.zeros((self.ch, self.n))
    def dump_statistics(self, analyze_vec, prt=print):
        level_vec = dict()
        for kdx, k in enumerate(self.keys):
            vec = analyze_vec[kdx]
            splk = k.split(".")
            for idx in range(len(splk)+1):
                nm = ".".join(splk[:idx])
                if(idx != len(splk)):
                    nm+=".*"
                level_vec[nm] = level_vec.get(nm, [])
                level_vec[nm].append(vec)
        names = sorted(list(level_vec), key=lambda x:len(level_vec[x]), reverse=True)
        for name in names:
            vecs = level_vec[name]
            avg = np.array(vecs).mean(axis=0)
            prt("    %s: %s (avg of %d weights)"%(name, avg, len(vecs)))
        
    def commit_vec(self, vec, normalizer: Callable = lambda x: x):
        vec = normalizer(vec)
        if (self._commited_vec is not None):
            dry = np.all(self._commited_vec==vec)
        else:
            dry = False
        self._commited_vec = vec
        # orig_vec = normalizer(self.vec)
        def fdelta(x):
            if(x>0):
                return "+%.2f%%"%(x)
            elif(x==0):
                return " %.2f%%"%(x)
            else:
                return "%.2f%%"%(x)
        def fpercent(x):
            if(x>0.7):
                color = "green"
            elif(x>0.3):
                color = "yellow"
            else:
                color = "red"
            return colored("%.2f%%"%(100*x), color)
        if(not dry):
            with print_time("interpolate model %s" % self.name):
                result_state = {}
                iterator = enumerate(self.keys)
                if(self.is_slow):
                    iterator = tqdm(list(iterator), desc=self.name)
                t = time.time()
                for idx, key in iterator:
                    w = vec[idx]
                    # print(key, w)
                    tensor = 0
                    for jdx, j in enumerate(self.states):
                        tensor = tensor + j[key]*w[jdx]
                    result_state[key] = tensor
                self.dst_model.load_state_dict(result_state)
                self.is_slow = (time.time()-t)>0.5
        print("%s:" % self.name, end="")
        for i in range(self.n):
            contrib = vec[:, i]
            mean = contrib.mean()
            print(" %.2f%%" % (mean*100), end="")
        print()

def load_json(fn):
    with open(fn, "r") as f:
        j = json.load(f)
    return j
def get_prompt_v2(j):
    positives = []
    negatives = []
    if(isinstance(j, dict)):
        typ = j["type"]
        frm = j["from"]
        if(isinstance(frm, str)):
            frm = [_.strip() for _ in frm.split(",")]
        if(typ=="choice"):
            result = [random.choice(frm)]
        elif(typ=="sample"):
            cnt = j["count"]
            result = random.sample(frm, min(len(frm), cnt))
        else:
            assert typ=="sample_freq", typ
            freq = j["freq"]
            result = []
            for _ in frm:
                if(random.random()<freq):
                    result.append(_)
            random.shuffle(result)
        result = [get_prompt_v2(_) for _ in result]
        if(not j.get("positive", True)):
            result = [(n, p) for p, n in result]
        for p, n in result:
            if(p):
                positives.append(p)
            if(n):
                negatives.append(n)
    elif(isinstance(j, str)):
        positives = [j]
    else:
        assert isinstance(j, list)
        for _ in j:
            pos, neg = get_prompt_v2(_)
            if(pos):
                positives.append(pos)
            if(neg):
                negatives.append(neg)
    return ", ".join(positives), ", ".join(negatives)



def input_yn(prompt, chars="ny"):
    while(True):
        inp = input(prompt)
        if(inp):
            ch = inp[0].lower()
            if(ch in chars.lower()):
                return chars.lower().find(ch)
def add_title(img, text, fill=(0, 0, 0, 255)):
    w, h = img.size
    font_size = int(sqrt(w*h)/10)
    RT = RichText([text], font_size=font_size, width=int(w*0.9), fill=fill)
    COL = Column([img, RT], bg=(255,)*4)
    return COL.render()
VAE_EXTRA = 4
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str,
                        required=True, help="specify model paths in this file")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--prompt-config", type=str, default="")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--clip-overflow", type=float, default=0)
    parser.add_argument("--infer-steps", type=int, default=32)
    parser.add_argument("--normalizer", type=str,
                        choices=["softmax", "clip_std", "clip_overflow"], default="softmax")
    parser.add_argument("--lr", type=float, default=2)
    parser.add_argument("--lowram", action="store_true", help="Loading models from disk consumes much runtime; rmhf stores model weights in RAM, but each may cost about 7GB RAM; enable this to store them in fp16 format")
    args = parser.parse_args()
    with open(args.prompt_config) as f:
        prompt_dict = json.load(f)
    prompt, neg = get_prompt_v2(prompt_dict)
    print(prompt)
    print(neg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.model_config, "r") as f:
        j = json.load(f)
    models = []
    vaes = []
    unets = []
    tes = []
    model_names = []
    model_init = []
    for idx, i in enumerate(j):
        if(not i.get("enabled", True)):
            continue
        pth = i["pth"]
        vae = i.get("vae", "")
        model_name = i.get("name", os.path.basename(pth)[:8])
        model_init.append(i.get("init_bias", 0))
        model = load_model(model_name, pth, vae)
        if(args.lowram):
            model.to(torch.float16)
        models.append(model)
        model_names.append(model_name)
        vaes.append(model.vae.state_dict())
        unets.append(model.unet.state_dict())
        tes.append(model.text_encoder.state_dict())
    normalizer = {
        "softmax": softmax,
        "clip_std": clip_std,
        "clip_overflow": partial(clip_overflow, of=args.clip_overflow)
    }[args.normalizer]
    _model = model_mgr._INFER_MODEL
    interp_vae = Interpolater("vae", _model.vae, vaes, model_names)

    interp_unet = Interpolater("unet", _model.unet, unets, model_names)
    interp_te = Interpolater(
        "text_encoder", _model.text_encoder, tes, model_names)
    interps = [interp_vae, interp_unet, interp_te]
    for i in interps:
        i.vec = i.vec + model_init
    _model = model_mgr._INFER_MODEL
    model = CustomPipeline(_model)
    steps = args.steps
    momentum = [i.zeros()*0 for i in interps]
    
    generate_sample_repro = model.txt2img(
        prompt, steps=args.infer_steps, width=640, height=768, neg_prompt=neg)
    generate_sample = lambda *args, **kwargs: generate_sample_repro.reproduce(
        *args, **kwargs).result
    
    
    nmodels = interp_vae.n
    model_samples = []
    for i in range(nmodels):
        print("generate base sample for %s"%model_names[i])
        for jdx, j in enumerate(interps):
            vec = j.zeros()
            vec[:, i] = 1
            j.commit_vec(vec)
        im = generate_sample()
        title = "Model=%s"%model_names[i]
        im = add_title(im, title)
        # im = Column([im, RichText([title], font_size=30)], bg=(255,)*4).render()
        model_samples.append(im)
        im.save("sample_%s.png"%(model_names[i]))
    step_selections = []
    for step in range(steps):
        try:
            direction = [i.rnd_vec() for i in interps]
            step_rate = args.alpha**(step/(steps-1))
            for idx, i in enumerate(direction):
                bymodel_bias = np.random.normal(0, 1, (1, interps[idx].n))
                r = step_rate
                if(interps[idx].name=="vae"):
                    # VQVAE is quite non-linear.
                    # In practice I found that when interpolate, images becomes gray, undersaturated.
                    # Thus, should make sure one model has dominate contribution.
                    r*=VAE_EXTRA*sqrt(nmodels)
                vec = i + bymodel_bias*r + momentum[idx]
                vec /= (2+r*r)**0.5
                direction[idx] = vec
                

            prompt, neg = get_prompt_v2(prompt_dict)
            delta_norm = []
            for idx, i in enumerate(interps):
                delta = direction[idx]*step_rate*args.lr
                newvec = i.vec - delta
                delta_norm.append((delta**2).mean())
                i.commit_vec(i.vec, normalizer)
            delta_norm = np.array(delta_norm).mean() ** 0.5
            momentum_norm = np.array([(i**2).mean()
                                     for i in momentum]).mean()**0.5
            repro = model.txt2img(
                prompt, steps=args.infer_steps, width=640, height=768, neg_prompt=neg)
            image0 = repro.result

            for idx, i in enumerate(interps):
                newvec = i.vec + direction[idx]*step_rate*args.lr
                i.commit_vec(i.vec + newvec, normalizer)

            image1 = repro.reproduce().result

            image0.save("./0.png")
            image1.save("./1.png")

            arr = np.array(image0).astype(np.float32) - np.array(image1)
            arr = (arr**2).sum(axis=2)
            arr = arr**0.5
            arr = (arr-arr.min())/(arr.max()-arr.min())*255
            image_d = Image.fromarray(arr.astype(np.uint8))
            Row([image0, image_d, image1]).render().save("./diff.png")
            print("Moving Speed", delta_norm, "momentum", momentum_norm)
            which = input_yn(
                "./0.png <-> ./1.png / ./diff.png which is better? (0/1)", "01")
            forward = which*2-1
            for idx, i in enumerate(interps):
                fd = forward*direction[idx]
                cross = (fd*momentum[idx]).sum() / ((fd**2).sum()
                                                    ** 0.5) / ((momentum[idx]**2 + 1e-7).sum()**0.5)
                print("foward direction is %.2f%% same with momentum" %
                      (cross*100))
                newvec = i.vec + fd*step_rate*args.lr
                momentum[idx] = fd * (1-args.beta) + momentum[idx]*args.beta
                i.vec = newvec
                norm = normalizer(i.vec)
                model_contrib = np.mean(norm, axis=0)
                print("avg. weight of %s:" % i.name, model_contrib)
                i.commit_vec(i.vec, normalizer)
            
            selection = []
            sel0 = forward<0
            if(sel0):
                text0, fill0 = "Selected", (0, 255, 0, 255)
                text1, fill1 = ":(", (255, 0, 0, 255)
            else:
                text0, fill0 = ":(", (255, 0, 0, 255)
                text1, fill1 = "Selected", (0, 255, 0, 255)
            image0 = add_title(image0, text0, fill=fill0)
            # image0 = Column([image0, RichText([text0], fill=fill0, font_size=48)])
            # image1 = Column([image1, RichText([text1], fill=fill1, font_size=48)])
            image1 = add_title(image1, text1, fill=fill1)
            stepim = add_title(generate_sample(), "Step %03d merged"%step)
            # stepim = Column([generate_sample(), RichText(["Step %03d Merged"%step], font_size=30)], bg=(255,)*4).render()
            step_selections.append([image0, stepim, image1])
            contents = []
            for i in step_selections:
                contents.append(Row(i))
            im = Column(contents, bg=(255,)*4).render()
            limit_image_size(im, area=1.5e7).convert("RGB").save("steps.jpg", quality=97)
            contents = []
            for i in model_samples:
                contents.append(Row([i, stepim]))
            im = Column(contents, bg=(255,)*4).render()
            limit_image_size(im, area=1.5e7).convert("RGB").save("compare_models.jpg", quality=97)
        except KeyboardInterrupt:
            break
    print("saving result..")
    for idx, i in enumerate(interps):
        i.commit_vec(i.vec, normalizer)
    save_ldm("./rmhf.safetensors", _model)
    print("model saved. generating detailed comparison between models")
    generate_sample().save("./sample_merged.png")
    
    with open("./rmhf.log", "w") as f:
        prt = partial(print, file=f)
        for interp in interps:
            prt(interp.name+":")
            prt("    models: %s"%(model_names))
            norm = normalizer(interp.vec)
            interp.dump_statistics(norm, prt)
            # for idx, k in enumerate(interp.keys):
            #     prt("    %s: %s"%(k, norm[idx, :]))
    # rows = [[]]
    rows = []
    nprompts = nmodels
    repros = []
    if(os.path.exists("./rmhf_samples")):
        shutil.rmtree("./rmhf_samples")
    os.makedirs("./rmhf_samples")
    for jdx in range(nprompts):
        p, n = get_prompt_v2(prompt_dict)
        repro = model.txt2img(p, steps=args.infer_steps, width=640, height=768, neg_prompt=n)
        pth = os.path.join("./rmhf_samples", "%03d.jpg"%jdx)
        repro.result.save(pth)
        with open(pth+".caption", "w") as f:
            print("Positive:", p, file = f)
            print("Negative:", n, file = f)
        repros.append(repro)
    # rows[0] = Row(rows[0])
    for idx in range(nmodels):
        for interp in interps:
            vec = interp.zeros()
            vec[:, idx] = 1
            interp.commit_vec(vec)
        row = []
        for jdx in range(nprompts):
            repro = repros[jdx]
            im = repro.reproduce().result
            im = add_title(im, "model=%s"%model_names[idx])
            row.append(im)
            row.append(add_title(repro.result, "merged model"))
        rows.append(Row(row))
    im = Column(rows).render()
    im = limit_image_size(im, area=1e8).convert("RGB")
    im.save("./compare_models.jpg")


    
main()
