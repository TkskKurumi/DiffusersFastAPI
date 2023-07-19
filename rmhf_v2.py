import argparse
from functools import partial
import math, os
from termcolor import colored
from math import sqrt
from dataclasses import dataclass
import numpy as np
from PIL import Image
from os import path
from modules.diffusion.model_mgr import model_mgr
from modules.diffusion.model_mgr import load_model
from modules.diffusion.wrapped_pipeline import CustomPipeline, WeightedPrompt
from modules.utils.diffuers_conversion import save_ldm
from modules.utils.pil_layout import RichText, Column, Row
from modules.utils.pil_layout.resize import limit as limit_image_size
from modules.utils.candy import print_time
from modules.utils.misc import normalize_resolution
from modules.utils.debug_vram import debug_vram
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
        EPS = 1e-5
        dry = False
        debug_vram("rmhf_v2.py:102")
        prev_w = self._current_norm
        prev_sd = self.dst_model.state_dict()
        debug_vram("rmhf_v2.py:105")
        if(do_softmax):
            self._current = w
            w = softmax(w)
            if(self._current_norm is not None):
                diff = np.abs(w-self._current_norm)
                if(np.all(diff<EPS)):
                    dry = True
            self._current_norm = w
        else:
            self._current = np.log(w)
            if(self._current_norm is not None):
                diff = np.abs(w-self._current_norm)
                if(np.all(diff<EPS)):
                    dry = True
            self._current_norm = w
        assert not (np.any(np.isnan(self._current_norm)))
        if(not dry):
            with print_time("interpolate model %s" % self.name):
                ret = {}
                ls = list(enumerate(self.keys))
                for key_idx, key in tqdm(ls):
                    _dry = False
                    if(prev_w is not None):
                        ws = w[key_idx]
                        prev_ws = prev_w[key_idx]
                        diff = np.abs(ws-prev_ws)
                        if(np.all(diff<EPS)):
                            tensor = prev_sd[key]
                            _dry = True
                    if(not _dry):
                        tensor = 0
                        for model_idx, sd in enumerate(self.states):
                            model_tensor = sd[key]
                            model_w = w[key_idx, model_idx]
                            if(abs(model_w)>1e-6):
                                tensor += model_tensor*model_w
                    ret[key] = tensor
                self.dst_model.load_state_dict(ret)
        self.display()
    def dump_detail(self, prt):
        rows = []
        rows.append((self.name, *self.names))
        rows.append(["-"]*len(rows[0]))
        for key_idx, key in enumerate(self.keys):
            ws = self._current_norm[key_idx]
            row = [self.name+"."+key]
            for w in ws:
                row.append("%.2f%%"%(w*100))
            rows.append(row)
        n_columns = len(rows[0])
        column_w = []
        for i in range(n_columns):
            w = 0
            for jdx, j in enumerate(rows):
                w = max(w, len(j[i]))
            column_w.append(w)
        for row_i, row in enumerate(rows):
            prt("| ", end="")
            for col_i, elem in enumerate(row):
                if(col_i):
                    prt(" ", end="")
                compensate = column_w[col_i]-len(elem)
                prt(elem+" "*compensate, end=" |")
            prt("")
        



def check_castable(arr0, arr1):
    shape0 = arr0.shape
    shape1 = arr1.shape
    if(len(shape0) > len(shape1)):
        return check_castable(arr1, arr0)
    print(shape0, shape1)
    for idx, i in enumerate(shape0):
        jdx = idx-len(shape0)
        j = shape1[jdx]
        if(i!=1 and j!=1 and i!=j):
            return False
    return True
        
    

class Optimize:
    def __init__(self, i: Interpolate, initial=None):
        self.i = i
        if(initial is None):
            self.w = i.zeros()
        else:
            z = i.zeros()
            if(not check_castable(z, np.array(initial))):
                print("WARNING: not using initial weight for", i.name, "since shape doesn't match", z.shape, np.array(initial).shape)
                self.w = z
            else:
                initw = str(initial)
                if(len(initw)>255):
                    initw = initw[:252]+"..."
                print("init", i.name, initw)
                self.w = i.zeros()+initial
        print("init", self.i.name, self.w.mean(axis=0))
        self.i.apply(self.w)
    def step(self, w, lr):
        raise NotImplementedError()
    def dummy_step(self, w, lr):
        raise NotImplementedError()
    def randn(self):
        return self.i.randn()
    @property
    def momentum(self):
        raise NotImplementedError()

class OptimizeAdamLike(Optimize):
    def __init__(self, i: Interpolate, initial=None, beta1=0.9, beta2=0.8, decay=0.02, eps=1e-6):
        super().__init__(i, initial)
        self.init_w = self.w
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.decay = decay
        self.m1 = i.zeros()
        self.m2 = i.zeros()
    def dummy_step(self, w, lr):
        m1 = self.m1*self.beta1 + w*(1-self.beta1)
        g2 = m1*m1 # w is noise, use moving avg for m2
        m2 = self.m2*self.beta2 + g2*(1-self.beta2)
        forward = m1/(m2+self.eps) - self.decay*(self.w-self.init_w)
        w = self.w+lr*forward
        return w, (m1, m2, forward, w)
    def step(self, w, lr):
        w, states = self.dummy_step(w, lr)
        self.m1, self.m2, forward, w = states
        print(self.i.name, "moving", np.abs(lr*forward).mean())
        self.w = w
        return w
    @property
    def momentum(self):
        return self.m1

class OptimizeLionLike(Optimize):
    def __init__(self, i: Interpolate, initial=None, beta1=0.95, beta2=0.8, decay=0.1):
        super().__init__(i, initial)
        self.init_w = self.w
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.m1 = i.zeros()
    def dummy_step(self, w, lr):
        c = self.m1*self.beta2 + w*(1-self.beta2)
        m1 = self.m1*self.beta1 + w*(1-self.beta1)
        
        forward = np.sign(c)-self.decay*(self.w-self.init_w)

        w = self.w+lr*forward
        return w, (c, m1, forward, w)
    def step(self, w, lr):
        w, states = self.dummy_step(w, lr)
        c, self.m1, forward, w = states
        print(self.i.name, "moving", np.abs(lr*forward).mean())
        self.w = w
        return w
    @property
    def momentum(self):
        return self.m1



class ModelConfig:
    def __init__(self, pth, name=None, enabled=True, vae="", init=1, src=None, **kwargs):
        self.pth = pth
        if(name is None):
            self.name = path.splitext(path.basename(pth))[0]
        else:
            self.name = name
        self.enabled = enabled
        self.vae = vae
        self.init = init
        self.src = src
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
GREEN = (0, 255, 255, 255)
def add_title(img, text, fill=BLACK):
    w, h = img.size
    font_size = int(sqrt(w*h)/10)
    RT = RichText([text], font_size=font_size, width=int(w*0.9), fill=fill)
    COL = Column([img, RT], bg=WHITE)
    return COL.render()
def _sample_params(args, sample_prompt, ar=None):
    pos, neg = sample_prompt()
    resolution = 512*512*2
    # ar_min = math.log(9/16)
    # ar_max = math.log(16/9)
    # ar = math.exp(ar_min + random.random()*(ar_max-ar_min))
    if(ar is None):
        ar = random.choice([9/16, 5/7, 3/4, 5/8, 1, 4/3])
    width, height = normalize_resolution(ar, 1, resolution)
    return (lambda **kwargs: kwargs)(prompt=pos, steps=args.infer_steps, width=width, height=height, neg_prompt=neg)

def do_compare_only(n: int, n_models: int, opts: List[Optimize], sample_params, model: CustomPipeline, model_names):
    
    repros = [None for i in range(n)]
    rows = []

    for i_model in range(n_models):
        for opt in opts:
            w = opt.i.zeros()
            w[:, i_model] += 1
            opt.i.apply(w, False)
        model_images = []
        for idx, repro in enumerate(repros):
            if(repro is None):
                ar = random.choice([5/7, 9/16])
                repro = model.txt2img(**sample_params(ar=ar))
                img = repro.result
                repros[idx] = repro
            else:
                img = repro.reproduce().result
            model_images.append(add_title(img, text=model_names[i_model]))
        rows.append(Row(model_images))
    col = Column(rows)
    col.render().save("compare_models.png")
    

    if(path.exists("rmhf_samples")):
        shutil.rmtree("rmhf_samples")
    os.makedirs("rmhf_samples")
    for i in range(256):
        params = sample_params()
        pro = params["prompt"]
        neg = params["neg_prompt"]
        repro = model.txt2img(**params)
        img = repro.result
        img.save(path.join("rmhf_samples", "%02d.png"%i))
        with open(path.join("rmhf_samples", "%02d.caption"%i), "w") as f:
            wp = WeightedPrompt(pro + "/*" + neg + "*/")
            pro, neg = [], []
            for k, v in wp.as_dict().items():
                if(k > 0):
                    pro.append(v)
                else:
                    neg.append(v)
            print("prompt:", ", ".join(pro), file=f)
            print("negative:", ", ".join(neg), file=f)



if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str,
                        required=True, help="specify model paths in this file")
    parser.add_argument("--prompt-config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=4)
    parser.add_argument("--lr_final", type=float, default=None)
    parser.add_argument("--by_model", type=float, default=1)
    parser.add_argument("--infer-steps", type=int, default=40)
    parser.add_argument("--compare-and-exit", type=int, default=0)
    args = parser.parse_args()
    infer_steps = args.infer_steps

    with open(args.prompt_config) as f:
        prompt_dict = json.load(f)
    
    

    if(args.prompt_config.endswith(".permute")):
        sample_prompt = lambda:(permute.sample_text(prompt_dict), "")
    else:
        sample_prompt = lambda:get_prompt_v2(prompt_dict)
    
    sample_params = partial(_sample_params, args, sample_prompt)

    with open(args.model_config, "r") as f:
        model_dict = json.load(f)
    steps = args.steps
    lr_init = args.lr
    lr_final = args.lr_final
    if(lr_final is None):
        lr_final = min(lr_init*0.5, 2)
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

    
    _model   = model_mgr._INFER_MODEL
    model    = CustomPipeline(_model)
    init_cfg = [math.log(i.init) for i in models]
    interps  = [
        Interpolate("unet", _model.unet, unets, model_names),
        Interpolate("text_encoder", _model.text_encoder, tes, model_names),
    ]
    init  = {}
    saved = {}
    if(path.exists("weights.npz")):
        saved = np.load("weights.npz")
    for i in interps:
        init[i.name] = init_cfg
        if(i.name in saved):
            w = saved[i.name]
            if(check_castable(w, i.zeros())):
                bym = args.by_model
                init[i.name] = (w+np.array(init_cfg)*bym)/sqrt(1+bym*bym)
                continue

    opts = [
        OptimizeLionLike(i, initial=init.get(i.name, None)) for i in interps
    ]

    n_models = len(models)

    if(args.compare_and_exit):
        do_compare_only(args.compare_and_exit, n_models, opts, sample_params, model, model_names)

    validate = model.txt2img(**sample_params(ar=5/7))

    step_images = []
    compare_models = []

    
    for i_model in range(n_models):
        for opt in opts:
            w = opt.i.zeros()
            w[:, i_model] += 1
            opt.i.apply(w, False)
        compare_models.append(validate.reproduce().result)

    R = Row([add_title(i, model_names[idx]) for idx, i in enumerate(compare_models)])
    R.render().convert("RGB").save("compare_models.jpg")

    for step in range(args.steps):
        lr_boost = 1
        try:
            while(True):
                lr = lr_sched(step)*lr_boost
                print("Step: %03d, LR: %.8f"%(step, lr))

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
                    w, states = opt.dummy_step(-ws[idx], lr)
                    opt.i.apply(w)
                
                repro = model.txt2img(**sample_params())

                for idx, opt in enumerate(opts):
                    w, states = opt.dummy_step(ws[idx], lr)
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
                        lr_boost*=1.1
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
            
            os.makedirs("rmhf2/steps", exist_ok=True)
            R.render().save("rmhf2/steps/%03d.png"%(step+1))



            imgv_titled = add_title(imgv, 'merged')
            compare_model_row = []
            for idx, mod in enumerate(models):
                contrib_unet = opts[0].i._current_norm[:, idx].mean()
                compare_model_row.append(
                    Column([
                        add_title(compare_models[idx], "%s - %.1f%%"%(mod.name, contrib_unet*100)),
                        imgv_titled]
                    )
                )
            R = Row(compare_model_row)
            R.render().convert("RGB").save("compare_models.jpg")
        except KeyboardInterrupt:
            break
    print("saving result")
    
    weights = {opt.i.name: opt.i._current for opt in opts}
    np.savez("weights.npz", **weights)

    save_ldm("./rmhf.safetensors", _model)
    with open("./rmhf.md", "w") as f:
        prt = partial(print, file=f)
        prt("This is a merged model of following models. Huge thanks for these amazing model creators.")
        for cfg in models:
            if(cfg.src):
                prt("+ [%s](%s)"%(cfg.name, cfg.src))
            else:
                prt("+", cfg.name)
        prt()
        for opt in opts:
            opt.i.dump_detail(prt)
    print("saved")
    if(path.exists("rmhf_samples")):
        shutil.rmtree("rmhf_samples")
    os.makedirs("rmhf_samples")
    for i in range(50):
        params = sample_params()
        pro = params["prompt"]
        neg = params["neg_prompt"]
        repro = model.txt2img(**params)
        img = repro.result
        img.save(path.join("rmhf_samples", "%02d.png"%i))
        with open(path.join("rmhf_samples", "%02d.caption"%i), "w") as f:
            wp = WeightedPrompt(pro + "/*" + neg + "*/")
            pro, neg = [], []
            for k, v in wp.as_dict().items():
                if(k > 0):
                    pro.append(v)
                else:
                    neg.append(v)
            print("prompt:", ", ".join(pro), file=f)
            print("negative:", ", ".join(neg), file=f)

