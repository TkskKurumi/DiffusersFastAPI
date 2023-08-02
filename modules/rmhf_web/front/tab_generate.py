import gradio as gr
import os
from ..back import learning
from .gradio_helper import ThenChain, ImageRow
from ...utils import normalize_resolution
import numpy as np
from ..back.generation import get_n_models, get_model_index, apply_model, StableDraw
from os import path            
from .. import constants
from ...diffusion.wrapped_pipeline import WeightedPrompt

        



        

        

class TabGen:
    def __init__(self):
        self._has_widgets = False
        self.tab_main = None
        self.repros = {}
        self.generated_image = {}
        self._m_model_name = "<merged model>"
        self.drawer = StableDraw()
    def init_widgets(self):
        if(self._has_widgets):
            return
        with gr.Tab("generate samples"):
            with gr.Blocks() as block_inputs:
                with gr.Row():
                    with gr.Column(scale=0.15):
                        self.btn_refresh_loaded_models = gr.Button("⟳")
                        self.select_model = gr.Dropdown(label="model", interactive=True, type="value")
                        self.label_contrib = gr.Label()
                    with gr.Column(scale=0.15):
                        self.txt_prompt = gr.Textbox(label = "prompt")
                        self.slide_aspect = gr.Slider(minimum=0.5, maximum=2, label="Aspect Ratio", interactive=True)
                        self.btn_rnd = gr.Button("Random")
                    with gr.Column(scale=0.7):
                        
                        self.btn_gen = gr.Button("Generate.")
                        self.image = gr.Image()
            with gr.Blocks() as block_cmp:
                with gr.Column():
                    self.btn_cmp = gr.Button("Compare models")
                self.cmp_row = ImageRow()
                self.cmp_row.init_widgets()
            with gr.Blocks() as block_gen_batch:
                self.batch_btn_generate = gr.Button("Generate batch of images.")
                self.batch_count = gr.Slider(minimum=1, maximum=constants.MAX_ROW_IMAGES, step=1, label="Count")

        self._has_widgets = True
    
    def register_generate_batch(self):
        self.init_widgets()
        tch = ThenChain()

        def _get_basedir(model_name):
            idx = get_model_index(model_name)
            if(idx<get_n_models()):
                model = learning.MODELS[idx]
                basedir = path.dirname(model.pth)
                basedir = path.join(basedir, model.name)
            else:
                basedir = "./"
            os.makedirs(basedir, exist_ok = True)
            return basedir
        def _get_savefile(model_name):
            basedir = _get_basedir(model_name)
            idx = 0
            def foo(ext=""):
                nonlocal idx
                return path.join(basedir, "%03d"%idx+ext)
            while(path.exists(foo(".jpg")) or path.exists(foo(".caption"))):
                idx += 1
            return foo()
        def set_draw_prelabel():
            def inner(idx):
                return "generating %d"%idx
            self.cmp_row.draw_prelabel = inner
        def set_draw(max_count, model_name, prompt, ar):
            def inner(idx):
                if(idx>=max_count):
                    return None
                elif(idx!=0):
                    params = self.tab_main.random_sample_params()
                else:
                    w, h = normalize_resolution(ar, 1, 512*512*2)
                    params = (lambda **kwargs:kwargs)(prompt=prompt, width=w, height=h)
                img = self.drawer(model_name, **params)
                
                pth = _get_savefile(model_name)

                pro, neg = [], []
                for k, v in WeightedPrompt(params["prompt"]).as_dict().items():
                    if (k>0):
                        pro.append(v)
                    else:
                        neg.append(v)
                img.save(pth+".jpg")
                with open(pth+'.caption', "w") as f:
                    print("prompt:", ", ".join(pro), file=f)
                    print("negative:", ", ".join(neg), file=f)
                
                return True, img, pth+".jpg"
            self.cmp_row.draw = inner
        
        tch.start(self.batch_btn_generate.click, set_draw_prelabel, None, None)\
           .then(set_draw, [self.batch_count, self.select_model, self.txt_prompt, self.slide_aspect], None)
        
        self.cmp_row.register_then(tch)

        return tch
           


    def register_generate_compare(self):
        def get_nmodels():
            return len(learning.MODELS)
        
        def set_draw_prelabel():
            def inner(idx):
                if(idx<get_nmodels()):
                    return "generating for %s"%learning.MODELS[idx].name
                elif(idx==get_nmodels()):
                    return "generating for %s"%self._m_model_name
                else:
                    return None
            self.cmp_row.draw_prelabel = inner
        def set_draw(prompt, ar):
            def inner(idx):
                nonlocal prompt, ar
                if(idx<=get_n_models()):
                    if(idx<get_n_models()):
                        model_name = learning.MODELS[idx].name
                    else:
                        model_name = self._m_model_name
                    w, h = normalize_resolution(ar, 1, 512*512*2)
                    return True, self.drawer(idx, prompt=prompt, width=w, height=h), model_name
                else:
                    return None
            self.cmp_row.draw = inner
        tch = ThenChain()
        tch.start(self.btn_cmp.click, set_draw_prelabel, None, None)\
           .then(set_draw, [self.txt_prompt, self.slide_aspect], None)
        self.cmp_row.register_then(tch)
        return tch

        
        


    def register_generate(self):
        self.init_widgets()
        def update_weights(model_name):
            model_name = model_name
            model_idx = len(learning.MODELS)+1
            for idx, model in enumerate(learning.MODELS):
                if(model.name == model_name):
                    model_idx = idx
            if(model_idx<len(learning.MODELS)):
                learning.one_model_hot(model_idx)
            else:
                for opt in learning.OPTS:
                    opt.i.apply(opt.w)
        def generate_image(model_name, prompt, ar):
            # update_weights(model_name)
            w, h = normalize_resolution(ar, 1, 512*512*2)
            img = self.drawer(model_name, prompt=prompt, width=w, height=h)
            # key = (model_name, prompt, w, h)
            # if (key in self.generated_image):
            #     return self.generated_image[key]
            
            # update_weights(model_name)
            # key = (prompt, w, h)
            # if (key in self.repros):
            #     repro = self.repros[key]
            #     img = repro.reproduce().result
            # else:
            #     repro = learning.MODEL.txt2img(prompt, width=w, height=h, steps=30)
            #     self.repros[key] = repro
            #     img = repro.result
            # if (model_name!=self._m_model_name):
            #     key = (model_name, prompt, w, h)
            #     self.generated_image[key] = img
            return img
        
        tch = ThenChain()
        tch.start(self.btn_gen.click, generate_image, [self.select_model, self.txt_prompt, self.slide_aspect], self.image)
        
        return tch

    def register_rand(self):
        self.init_widgets()
        def random_sample_params(prompt, ar):
            if(self.tab_main is not None):
                args = self.tab_main.random_sample_params()
                w, h = args["width"], args["height"]
                return args["prompt"], w/h
            else:
                return prompt, ar
        tch = ThenChain()
        tch.start(self.btn_rnd.click, random_sample_params, [self.txt_prompt, self.slide_aspect], [self.txt_prompt, self.slide_aspect])
        return tch

    def register(self):
        self.init_widgets()
        self.register_rand()
        self.register_generate()

        def foo():
            models = gr.Dropdown.update(choices=[model.name for model in learning.MODELS] + [self._m_model_name])
            
            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)
            return models, model_w
        tch = ThenChain()
        tch.start(self.btn_refresh_loaded_models.click, foo, None, [self.select_model, self.label_contrib])

        self.register_generate_compare()

        self.register_generate_batch()
    def register_then_update_model_choice(self, tch: ThenChain):
        self.init_widgets()
        def foo():
            return gr.Dropdown.update(choices=[model.name for model in learning.MODELS] + [self._m_model_name])
        tch.then(foo, None, self.select_model)
        return tch
