from ..config import AppCFG
from ..back import learning
from functools import wraps, partial
import json, traceback, random
import gradio as gr
import numpy as np
from .gradio_helper import ThenChain
from ..back.permute import sample_text
from ...utils import normalize_resolution
from types import NoneType


class TabMain:
    def __init__(self):
        self._has_widgets = False
        self.cmp : NoneType|learning.Comparison = None
        self._tab_manual = None
    
    def set_child(self, name, obj):
        setattr(self, "_"+name, obj)
        obj._tab_main = self

    def init_widgets(self):
        if(self._has_widgets):
            return
        with gr.Tab("main") as tab_main:
            self.tab = tab_main
            self.label_status = gr.Label("Hello")
            with gr.Blocks() as block_load:
                with gr.Row():
                    self.txt_model_cfg = gr.Textbox(label="model - config", value=AppCFG.default("model_config", "./model_config.json"))
                    self.txt_prompt_permute = gr.Textbox(label="prompt - permute", value=AppCFG.default("prompte_permute", "./prompt.permute"))
                self.button_load = gr.Button("Load")
                self.button_save = gr.Button("Save")
            with gr.Blocks() as block_img_cmp:
                with gr.Row() as col:
                    with gr.Column() as row0:
                        self.lbl0 = gr.Label()
                        self.image0 = gr.Image()
                        self.button0 = gr.Button("This is better")
                    with gr.Column() as col1:
                        self.lbl1 = gr.Label()
                        self.image1 = gr.Image()
                        self.button1 = gr.Button("This is better")
            with gr.Blocks() as block_learning:
                with gr.Row() as row:
                    self.slide_lr = gr.Slider(minimum=0, maximum=10, label="learning rate", value=5)
                    self.slide_lr_decay = gr.Slider(minimum=0, maximum=1, label="learning rate devay", value=AppCFG.default("lrd", 1))
                with gr.Row() as row:
                    self.btn_compare = gr.Button("Generate compare")

        self._has_widgets = True
    def random_sample_params(self, **kwargs):
        prompt = sample_text(self.pp)
        ar = random.choice([9/16, 5/7, 3/4, 1, 7/5])
        w, h = normalize_resolution(ar, 1, 512*512*2)
        args = (lambda **x:x)(prompt=prompt, steps=40, width=w, height=h)
        args.update(kwargs)
        return args

    def register_then_compare(self, tch: ThenChain):
        def init_cmp(learning_rate):
            cmp = learning.Comparison(learning.MODEL, learning.OPTS, learning_rate, self.random_sample_params, ex_shape=(1, len(learning.MODELS)))
            self.cmp = cmp

        def img0():
            image0 = self.cmp.image0

            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)

            return image0, model_w
        def img1():
            image1 = self.cmp.image1

            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)

            return image1, model_w
        
        tch.showlabel(self.label_status, "init comp")\
           .then(init_cmp, self.slide_lr, None)\
           .showlabel(self.label_status, "generate image-0")\
           .then(img0, None, [self.image0, self.lbl0])\
           .showlabel(self.label_status, "generate image-1")\
           .then(img1, None, [self.image1, self.lbl1])\
           .showlabel(self.label_status, "Samples generated OK")
        return tch

    def register_comparison(self):
        self.init_widgets()
        tch = ThenChain()

        def init_cmp(learning_rate):
            cmp = learning.Comparison(learning.MODEL, learning.OPTS, learning_rate, self.random_sample_params, ex_shape=(1, len(learning.MODELS)))
            self.cmp = cmp

        def img0():
            image0 = self.cmp.image0
            
            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)

            return image0, model_w
        def img1():
            image1 = self.cmp.image1

            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)

            return image1, model_w
        tch.start(self.btn_compare.click, lambda:None, None, None)\
           .showlabel(self.label_status, "init comp")\
           .then(init_cmp, self.slide_lr, None)\
           .showlabel(self.label_status, "generate image-0")\
           .then(img0, None, [self.image0, self.lbl0])\
           .showlabel(self.label_status, "generate image-1")\
           .then(img1, None, [self.image1, self.lbl1])\
           .showlabel(self.label_status, "Samples generated OK")
        
        def fo(forward):
            assert self.cmp is not None, "No compare is started"
            self.cmp.commit(forward)
            self.cmp = None

        def decay_lr(lr, lr_d):
            return lr*lr_d

        tch_click0 = ThenChain()
        tch_click0.start(self.button0.click, lambda:None, None, None)\
                  .showlabel(self.label_status, "Image0 selected, updating weights")\
                  .then(partial(fo, -1), None, None)\
                  .then(decay_lr, [self.slide_lr, self.slide_lr_decay], self.slide_lr)
        
        tch_click1 = ThenChain()
        tch_click1.start(self.button1.click, lambda:None, None, None)\
                  .showlabel(self.label_status, "Image1 selected, updating weights")\
                  .then(partial(fo, 1), None, None)\
                  .then(decay_lr, [self.slide_lr, self.slide_lr_decay], self.slide_lr)
        
        self.register_then_compare(tch_click0)
        self.register_then_compare(tch_click1)

        self._tab_manual.register_then_update_contrib(tch_click0)
        self._tab_manual.register_then_update_contrib(tch_click1)

        return tch

    def register_load(self):
        self.init_widgets()
        tch = ThenChain()

        def setcfg(cfg_file, pp_file):
            AppCFG.model_config = cfg_file
            AppCFG.prompt_permute = pp_file
            return
        
        def load_models(cfg_file, pp_file):
            print("loading models", cfg_file, pp_file)
            with open(cfg_file, "r") as f:
                cfgs = json.load(f)
            with open(pp_file, "r") as f:
                self.pp = json.load(f)
            learning.load_models(cfgs)

        def prepare_learning():
            learning.prepare_learning()

        tch.start(self.button_load.click, setcfg,  [self.txt_model_cfg, self.txt_prompt_permute], None)\
           .showlabel(self.label_status, "loading models")\
           .then(load_models, [self.txt_model_cfg, self.txt_prompt_permute], None)\
           .showlabel(self.label_status, "initialize weights")\
           .then(prepare_learning, None, None)\
           .showlabel(self.label_status, "Models loaded OK")
        
        self._tab_manual.register_then_update_contrib(tch)
        self._tab_manual.register_then_show_radios(tch)

        return tch
    def register_save(self):
        self.init_widgets()
        tch = ThenChain()

        def foo():
            for opt in learning.OPTS:
                opt.i.apply(opt.w)
            pth = learning.save()
            ret = "saved to "+pth
            print(ret)
            return ret
        
        tch.start(self.button_save.click, foo, None, self.label_status)

        return tch
    

           
           

        


