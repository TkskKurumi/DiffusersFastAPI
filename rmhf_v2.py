import argparse
import math, os
from termcolor import colored
from math import sqrt
from dataclasses import dataclass
import numpy as np
from PIL import Image
from os import path
from modules.diffusion.model_mgr import model_mgr
from modules.diffusion.model_mgr import load_model
from modules.diffusion.wrapped_pipeline import CustomPipeline
from modules.utils.diffuers_conversion import save_ldm
from modules.utils.pil_layout import RichText, Column, Row
from modules.utils.pil_layout.resize import limit as limit_image_size
from modules.utils.candy import print_time
from rmhf2 import permute
from tqdm import tqdm, trange
import json, random, shutil
from typing import List
def softmax(w: np.ndarray, axis=-1):
    exp = np.exp(w)
    expsum = exp.sum(axis=axis, keepdims=True)
    ret = exp/expsum
    # sm = ret.sum(axis=axis)
    # sm1 = np.abs(sm-1)
    # assert np.all(sm1<1e-8)
    return ret

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

class Interpolate:
    def __init__(self, name, dst_model, states, names):
        self.dst_model = dst_model
        self.name = name
        self.states = states
        self.names = names
        self.n_model = len(names)
        self.n_ch = len(states[0])
        self.keys = list(states[0].keys())
        self._current = None
        self._current_norm = None
    def display(self):
        print(self.name, *self.names, end="\n")
        print(" "*len(self.name), end="")
        for idx, name in enumerate(self.names):
            contribution = self._current_norm[:, idx].mean()
            n = len(name)
            # %n.2f%%
            format = "%" + str(n) + ".2f" + "%%"
            print(format%(100*contribution), end="")
        print()
        
    def zeros(self):
        return np.zeros((self.n_ch, self.n_model))
    def randn(self):
        return np.random.normal(size=(self.n_ch, self.n_model))
    
    def apply(self, w, do_softmax=True):
        dry = False
        if(do_softmax):
            self._current = w
            w = softmax(w)
            if(self._current_norm is not None):
                diff = np.abs(w-self._current_norm)
                if(np.all(diff<1e-6)):
                    dry = True
            self._current_norm = w
        else:
            self._current = np.log(w)
            if(self._current_norm is not None):
                diff = np.abs(w-self._current_norm)
                if(np.all(diff<1e-6)):
                    dry = True
            self._current_norm = w
        if(not dry):
            with print_time("interpolate model %s" % self.name):
                ret = {}
                ls = list(enumerate(self.keys))
                for key_idx, key in tqdm(ls):
                    tensor = 0
                    for model_idx, sd in enumerate(self.states):
                        model_tensor = sd[key]
                        model_w = w[key_idx, model_idx]
                        tensor += model_tensor*model_w
                    ret[key] = tensor
                self.dst_model.load_state_dict(ret)
        self.display()

class Optimize:
    def __init__(self, i: Interpolate, alpha=0.85, beta = None, eps=1e-7, initial=None):
        self.i = i
        self.i.apply(i.zeros())
        if(initial is None):
            self.w        = i.zeros()
        else:
            self.w        = i.zeros()+initial
        print("init", self.i.name, self.w.mean(axis=0))
        self.momentum = i.zeros()
        self.m2       = i.zeros()
        self.alpha = alpha
        if(beta is None):
            self.beta = alpha**2
        else:
            self.beta  = beta
        self.eps   = eps
    def step(self, w, lr):
        self.momentum = self.momentum*self.alpha + w*(1-self.alpha)                   # moving mean
        self.m2       = self.m2*self.beta + self.momentum*self.momentum*(1-self.beta) # w is not the accurate gradient, use momentum as gradient
        deltaw = self.momentum/(np.sqrt(self.m2)+self.eps)*lr
        print(self.i.name, "moving", np.abs(deltaw).mean())
        self.w = self.w+deltaw
    def randn(self):
        return self.i.randn()
    def dummy_step(self, w, lr):
        momentum = self.momentum*self.alpha + w*(1-self.alpha)
        m2       = self.m2*self.beta + momentum*momentum*(1-self.beta)
        deltaw = momentum/(np.sqrt(m2)+self.eps)*lr
        return self.w+deltaw


class ModelConfig:
    def __init__(self, pth, name=None, enabled=True, vae="", init=1):
        self.pth = pth
        if(name is None):
            self.name = path.splitext(path.basename(pth))[0]
        else:
            self.name = name
        self.enabled = enabled
        self.vae = vae
        self.init = init
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
GREEN = (0, 255, 255, 255)
def add_title(img, text, fill=BLACK):
    w, h = img.size
    font_size = int(sqrt(w*h)/10)
    RT = RichText([text], font_size=font_size, width=int(w*0.9), fill=fill)
    COL = Column([img, RT], bg=WHITE)
    return COL.render()

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str,
                        required=True, help="specify model paths in this file")
    parser.add_argument("--prompt-config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2)
    parser.add_argument("--lr_final", type=float, default=None)
    parser.add_argument("--by_model", type=float, default=1)
    parser.add_argument("--infer-steps", type=int, default=40)
    args = parser.parse_args()
    infer_steps = args.infer_steps

    with open(args.prompt_config) as f:
        prompt_dict = json.load(f)
    
    

    if(args.prompt_config.endswith(".permute")):
        sample_prompt = lambda:(permute.sample_text(prompt_dict), "")
    else:
        sample_prompt = lambda:get_prompt_v2(prompt_dict)
    
    prompt, negative = sample_prompt()

    with open(args.model_config, "r") as f:
        model_dict = json.load(f)
    steps = args.steps
    lr_init = args.lr
    lr_final = args.lr_final
    if(lr_final is None):
        lr_final = min(lr_init*0.8, 0.45)
    def lr_sched(s):
        global lr_init, lr_final
        s = s/steps
        ret  = lr_init  ** (1-s) 
        ret *= lr_final ** s
        return ret



    models: List[ModelConfig] = []
    model_names = []
    unets  = []
    tes    = []
    for idx, i in enumerate(model_dict):
        model_cfg = ModelConfig(**i)
        if(not model_cfg.enabled):
            continue
        model = load_model(model_cfg.name, model_cfg.pth, model_cfg.vae)
        models.append(model_cfg)
        model_names.append(model_cfg.name)
        unets.append(model.unet.state_dict())
        tes.append(model.text_encoder.state_dict())

    
    _model = model_mgr._INFER_MODEL
    model = CustomPipeline(_model)
    init    = [math.log(i.init) for i in models]
    interps = [
        Interpolate("unet", _model.unet, unets, model_names),
        Interpolate("text_encoder", _model.text_encoder, tes, model_names),
    ]
    opts = [
        Optimize(i, initial=init) for i in interps
    ]

    validate = model.txt2img(prompt, neg_prompt = negative, steps=infer_steps)

    step_images = []
    compare_models = []

    n_models = len(models)
    for i_model in range(n_models):
        for opt in opts:
            w = opt.i.zeros()
            w[:, i_model] += 1
            opt.i.apply(w, False)
        compare_models.append(validate.reproduce().result)

    R = Row([add_title(i, model_names[idx]) for idx, i in enumerate(compare_models)])
    R.render().convert("RGB").save("compare_models.jpg")

    for step in range(args.steps):
        try:
            lr = lr_sched(step)
            print("Step: %03d, LR: %.8f"%(step, lr))
            
            while(True):

                bym = args.by_model
                ws = []
                for opt in opts:
                    noise = opt.randn()
                    noise_bym = np.random.normal(size=(1, opt.i.n_model))
                    noise = (noise+noise_bym*bym)/sqrt(1+bym*bym)
                    if((noise*opt.momentum).sum()>0):
                        ws.append(noise)
                    else:
                        ws.append(-noise)
                
                prompt, negative = sample_prompt()

                for idx, opt in enumerate(opts):
                    w = opt.dummy_step(-ws[idx], lr)
                    opt.i.apply(w)
                
                repro = model.txt2img(prompt, neg_prompt = negative, steps=infer_steps)

                for idx, opt in enumerate(opts):
                    w = opt.dummy_step(ws[idx], lr)
                    opt.i.apply(w)
                
                img0 = repro.result
                img1 = repro.reproduce().result
                
                diff = np.array(img0).astype(np.float32) - np.array(img1).astype(np.float32)
                diff = np.sqrt((diff**2).sum(axis=-1))
                diff = (diff-diff.min())/(diff.max()-diff.min()+1e-7)*255

                imgd = Image.fromarray(diff.astype(np.uint8))

                R = Row([
                    add_title(img0, "0"),
                    add_title(imgd, "<- diff ->"),
                    add_title(img1, "1")
                ])
                R.render().save("./diff.png")
                
                regen = False
                while(True):
                    i = input("check ./diff.png (0/1): ")
                    if(i == "0"):
                        forward = -1
                        break
                    elif(i == "1"):
                        forward = 1
                        break
                    elif(i == "skip"):
                        forward = 0
                        break
                    elif(i == "alt"):
                        regen=True
                        break
                    elif(i=="up"):
                        regen=True
                        lr_init*=1.05
                        lr_final*=1.05
                        break
                if(not regen):
                    break
                        


            for idx, w in enumerate(ws):
                opts[idx].step(w*forward, lr)
                opts[idx].i.apply(opts[idx].w)
            imgv = validate.reproduce().result

            if(forward == 1):
                v = "validation, step %02d/%02d"%(step+1, steps)
                R = Row([
                    add_title(img0, ":("),
                    add_title(imgd, "diff"),
                    add_title(img1, "chosen", fill=GREEN),
                    add_title(imgv, v)
                ])
            else:
                v = "validation, step %02d/%02d"%(step+1, steps)
                R = Row([
                    add_title(img0, "chosen", fill=GREEN),
                    add_title(imgd, "diff"),
                    add_title(img1, ":("),
                    add_title(imgv, v)
                ])
            step_images.append(R.render())
            Column(step_images).render().convert("RGB").save("./steps.jpg")

            imgv_titled = add_title(imgv, 'merged')
            R = Row([
                Column([add_title(i, model_names[idx]), imgv_titled])
                for idx, i in enumerate(compare_models)
            ])
            R.render().convert("RGB").save("compare_models.jpg")
        except KeyboardInterrupt:
            break
    print("saving result")
    save_ldm("./rmhf.safetensors", _model)
    print("saved")
    if(path.exists("rmhf_samples")):
        shutil.rmtree("rmhf_samples")
    os.makedirs("rmhf_samples")
    for i in range(10):
        prompt, negative = sample_prompt()
        repro = model.txt2img(prompt, neg_prompt = negative, steps=infer_steps)
        img = repro.result
        img.save(path.join("rmhf_samples", "%02d.png"%i))
