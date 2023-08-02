from __future__ import annotations
from datetime import datetime
from functools import partial
from os import path
import numpy as np
import os
from ...diffusion.model_mgr import load_model
from ...diffusion.model_mgr import model_mgr
from ...diffusion.wrapped_pipeline import CustomPipeline
from .optimizer import OptimizeLionLike as Optimize
from .interp import Interpolate
from typing import Any, List
from math import log, sqrt, exp
from ...utils.diffuers_conversion import save_ldm

INFER_MODEL = None
MODEL: CustomPipeline = None
MODELS: List[ModelConfig] = []
INTERPS: List[Interpolate] = []
OPTS: List[Optimize] = []
READY = False
class ModelConfig:
    def __init__(self, pth, name=None, enabled=True, vae="", init=None, src=None, **kwargs):
        self.pth = pth
        if(name is None):
            self.name = path.splitext(path.basename(pth))[0]
        else:
            self.name = name
        self.enabled = enabled
        self.vae = vae
        self.init = init
        self.src = src
        self.loaded = False
        self.unet = None
        self.text_encoder = None
        self.unet_w = None
        self.text_encoder_w = None
        self.idx = None
    def load(self):
        if(self.loaded):
            return
        model = load_model(self.name, self.pth, self.vae)
        self.unet = model.unet
        self.text_encoder = model.text_encoder
        self.loaded = True
        self.model = model
    def __getattr__(self, __name: str) -> Any:
        return getattr(self.model, __name)
def one_model_hot(idx):
    for opt in OPTS:
        intp = opt.i
        w = intp.zeros()-10
        w[:, idx] = 10
        intp.apply(w)


class Comparison:
    def __init__(self, model: CustomPipeline, opts: List[Optimize], lr, sample_params, ex_w = None, ex_shape=None, ex_ratio=0.5, repro=None):
        self.model = model
        self.opts = opts
        self.ws = []
        for opt in opts:
            w = opt.i.randn()
            if(ex_w is not None):
                w = (w+ex_w*ex_ratio)/sqrt(1+ex_ratio**2)
            elif(ex_shape is not None):
                ex_w = np.random.normal(0, 1, ex_shape)
                w = (w+ex_w*ex_ratio)/sqrt(1+ex_ratio**2)
            momentum = opt.momentum
            if((w*momentum).sum()<0):
                w = -w
            self.ws.append(w)
        if(callable(sample_params)):
            self.generation_params = sample_params()
        else:
            self.generation_params = sample_params
        self.repro = repro
        self._image0 = None
        self._image1 = None
        self.lr = lr
    @property
    def image0(self):
        if(self._image0 is not None):
            return self._image0
        for idx, opt in enumerate(self.opts):
            w, states = opt.dummy_step(-self.ws[idx], self.lr)
            opt.i.apply(w)
        if(self.repro is None):
            repro = self.model.txt2img(**self.generation_params)
            self._image0 = repro.result
            self.repro = repro
        else:
            repro = self.repro.reproduce()
            self._image0 = repro.result
            self.repro = repro
        return self._image0
    @property
    def image1(self):
        if(self._image1 is not None):
            return self._image1
        for idx, opt in enumerate(self.opts):
            w, states = opt.dummy_step(self.ws[idx], self.lr)
            opt.i.apply(w)
        if(self.repro is None):
            repro = self.model.txt2img(**self.generation_params)
            self._image1 = repro.result
            self.repro = repro
        else:
            repro = self.repro.reproduce()
            self._image1 = repro.result
            self.repro = repro
        return self._image1
    
    def commit(self, forward):
        for idx, w in enumerate(self.ws):
            self.opts[idx].step(w*forward, self.lr)
            # self.opts[idx].i.apply(self.opts[idx].w)

def get_weight_for_save():
    ret = {}
    for intp in INTERPS:
        name = intp.name
        ws = intp._current_norm
        for idx, model_name in enumerate(intp.names):
            w = ws[:, idx]
            ret[model_name+"."+name] = w*len(MODELS)
    return ret

def _prepare_init_weights(preload, unet_shape, text_encoder_shape):
    global MODELS
    unet_init = []
    text_encoder_init = []
    for model in MODELS:
        def foo(subname, shape):
            if(model.name+subname in preload):
                return preload[model.name+subname]
            elif(model.init is not None):
                return model.init * np.ones(shape)
            return None
        un = foo(".unet", unet_shape)
        te = foo(".text_encoder", text_encoder_shape)
        unet_init.append(un)
        text_encoder_init.append(te)
    def fill_none(ls, shape):
        notnone = []
        for i in ls:
            if(i is not None):
                notnone.append(np.mean(i))

        if(notnone):
            avg = np.mean(notnone)
        else:
            avg = 1/len(ls)

        for idx, i in enumerate(ls):
            if(i is None):
                ls[idx] = avg*np.ones()
        return ls

    unet_init = fill_none(unet_init, unet_shape)
    text_encoder_init = fill_none(text_encoder_init, text_encoder_shape)

    unet_init = np.stack(unet_init, axis=-1)
    text_encoder_init = np.stack(text_encoder_init, axis=-1)

    unet_init = np.log(unet_init/unet_init.sum(axis=1, keepdims=True))
    text_encoder_init = np.log(text_encoder_init/text_encoder_init.sum(axis=1, keepdims=True))

    return unet_init, text_encoder_init

def load_w(pth="./rmhf.npz"):
    if(path.exists(pth)):
        return dict(np.load(pth))
    else:
        return {}

def save_w(pth="./rmhf.npz"):
    prev = load_w(pth)
    prev.update(get_weight_for_save())
    np.savez(pth, **prev)

def prepare_learning():
    global MODELS, INFER_MODEL, OPTS, INTERPS
    assert (INFER_MODEL is not None), "No model loaded"
    for model in MODELS:
        model.load()
    
    unet_keys = list(model.unet.state_dict().keys())
    text_encoder_keys = list(model.text_encoder.state_dict().keys())

    preload = load_w()

    unet_init, text_encoder_init = _prepare_init_weights(preload, (len(unet_keys), ), (len(text_encoder_keys), ))


    names = [model.name for model in MODELS]
    unet_states = [model.unet.state_dict() for model in MODELS]
    te_states = [model.text_encoder.state_dict() for model in MODELS]

    unet_interp = Interpolate("unet", INFER_MODEL.unet, unet_states, names)
    te_interp = Interpolate("text_encoder", INFER_MODEL.text_encoder, te_states, names)

    unet_opt = Optimize(unet_interp, unet_init)
    te_opt = Optimize(te_interp, text_encoder_init)

    INTERPS = [unet_interp, te_interp]
    OPTS = [unet_opt, te_opt]
    READY = True
    return
    
            

def load_models(cfgs):
    global MODELS, INFER_MODEL, MODEL, READY
    READY = False
    MODELS = []
    for cfg in cfgs:
        if(isinstance(cfg, dict)):
            cfg = ModelConfig(**cfg)
        assert isinstance(cfg, ModelConfig)
        if(cfg.enabled):
            cfg.idx = len(MODELS)
            MODELS.append(cfg)
            cfg.load()
    INFER_MODEL = model_mgr._INFER_MODEL
    MODEL = CustomPipeline(INFER_MODEL)

def save(pth=None):
    assert INFER_MODEL
    if(pth is None):
        pth = "rmhf-" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(pth)

    save_ldm(path.join(pth, "rmhf.safetensors"), INFER_MODEL)
    with open(path.join(pth, "readme.md"), "w") as f:
        prt = partial(print, file=f)
        prt("This is a merged model of following models. Huge thanks for the creators for these amazing models.")
        for model in MODELS:
            if(model.src):
                prt("+ [%s](%s)"%(model.name, model.src))
            else:
                prt("+", model.name)
        prt()
        for opt in OPTS:
            opt.i.dump_detail(prt)
    save_w()
    return pth