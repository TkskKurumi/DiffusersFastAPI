import random
from modules.diffusion.model_mgr import load_model
from modules.diffusion.model_mgr import model_mgr
from modules.diffusion.wrapped_pipeline import CustomPipeline
from modules.utils.diffuers_conversion import save_ldm
from modules.utils.candy import print_time
import argparse
import json
import numpy as np
import torch
from functools import partial
from typing import Callable


def softmax(vec, axis=1):
    exp = np.exp(vec)
    expsum = np.sum(exp, axis=axis, keepdims=True)
    return exp/expsum


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
    def __init__(self, name, dst_model, *states):
        self.dst_model = dst_model
        self.states = states
        self.keys = states[0].keys()
        self.ch = len(self.keys)
        self.n = len(states)
        self.name = name
        self.vec = self.rnd_vec()

    def rnd_vec(self):
        return np.random.normal(0, 1, (self.ch, self.n))

    def zeros(self):
        return np.zeros((self.ch, self.n))

    def commit_vec(self, vec, normalizer: Callable):
        vec = normalizer(vec)
        with print_time("interpolate model %s" % self.name):
            result_state = {}
            for idx, key in enumerate(self.keys):
                w = vec[idx]
                # print(key, w)
                tensor = 0
                for jdx, j in enumerate(self.states):
                    tensor = tensor + j[key]*w[jdx]
                result_state[key] = tensor
            self.dst_model.load_state_dict(result_state)


def get_prompt(prompt_file):
    with open(prompt_file, "r") as f:
        j = json.load(f)
    j = random.choice(j)

    def _(k):
        if (k in j):
            return j[k]
        else:
            ps = j[k+"_sample"]
            n = ps["num"]
            sep = ps["sep"]
            sample_from = ps["from"].split(sep)
            if(len(sample_from) > n):
                sample_from = random.sample(sample_from, n)
            return sep.join(sample_from)

    prompt = _("prompt")
    neg = _("negative")
    return prompt, neg


def input_yn(prompt, chars="ny"):
    while(True):
        inp = input(prompt)
        if(inp):
            ch = inp[0].lower()
            if(ch in chars.lower()):
                return chars.lower().find(ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str,
                        required=True, help="specify model paths in this file")
    parser.add_argument("--seed", type=int, default=998244353)
    parser.add_argument("--prompt-config", type=str, default="")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--clip-overflow", type=float, default=0)
    parser.add_argument("--infer-steps", type=int, default=25)
    parser.add_argument("--normalizer", type=str,
                        choices=["softmax", "clip_std", "clip_overflow"], default="clip_overflow")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.model_config, "r") as f:
        j = json.load(f)
    models = []
    vaes = []
    unets = []
    tes = []
    for idx, i in enumerate(j):
        pth = i["pth"]
        vae = i.get("vae", "")
        model = load_model("model_%d" % idx, pth, vae)
        models.append(model)
        vaes.append(model.vae.state_dict())
        unets.append(model.unet.state_dict())
        tes.append(model.text_encoder.state_dict())
    normalizer = {
        "softmax": softmax,
        "clip_std": clip_std,
        "clip_overflow": partial(clip_overflow, of=args.clip_overflow)
    }[args.normalizer]
    _model = model_mgr._INFER_MODEL
    interp_vae = Interpolater("vae", _model.vae, *vaes)
    interp_unet = Interpolater("unet", _model.unet, *unets)
    interp_te = Interpolater("text_encoder", _model.text_encoder, *tes)
    interps = [interp_vae, interp_unet, interp_te]

    _model = model_mgr._INFER_MODEL
    model = CustomPipeline(_model)
    steps = args.steps
    momentum = [i.zeros()*0 for i in interps]
    for step in range(steps):
        try:
            direction = [i.rnd_vec() for i in interps]
            step_rate = args.alpha**(step/(steps-1))
            for idx, i in enumerate(direction):
                bymodel_bias = np.random.normal(0, 1, (1, interps[idx].n))
                r = step_rate**4
                vec = i + bymodel_bias*r + momentum[idx]
                vec /= (2+r*r)**0.5
                direction[idx] = vec

            prompt, neg = get_prompt(args.prompt_config)
            mean2 = []
            for idx, i in enumerate(interps):
                delta = direction[idx]*step_rate
                newvec = i.vec - delta
                l2mean = (delta**2).mean()
                mean2.append(l2mean)
                i.commit_vec(i.vec, normalizer)
            mean2 = np.array(mean2).mean() ** 0.5
            print("Moving Speed", mean2)
            repro = model.txt2img(
                prompt, steps=args.infer_steps, width=640, height=768, neg_prompt=neg)
            image0 = repro.result

            for idx, i in enumerate(interps):
                newvec = i.vec + direction[idx]*step_rate
                i.commit_vec(i.vec + newvec, normalizer)

            image1 = repro.reproduce().result

            image0.save("./0.png")
            image1.save("./1.png")
            which = input_yn(
                "./0.png <-> ./1.png which is better? (0/1)", "01")
            forward = which*2-1
            for idx, i in enumerate(interps):
                newvec = i.vec + forward*direction[idx]*step_rate
                momentum[idx] = forward*direction[idx]*(1-args.alpha) + momentum[idx]*args.alpha
                i.vec = newvec
                norm = normalizer(i.vec)
                model_contrib = np.mean(norm, axis=0)
                print("avg. weight of %s:" % i.name, model_contrib)
        except KeyboardInterrupt:
            break
    print("saving result..")
    for idx, i in enumerate(interps):
        i.commit_vec(i.vec, normalizer)
    save_ldm("./rmhf.safetensors", _model)


main()
