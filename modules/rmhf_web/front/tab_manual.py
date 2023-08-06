from ..back import learning
import gradio as gr
from .gradio_helper import ThenChain
from .. import constants
import numpy as np
from ..back.generation import get_n_models, get_model_index, apply_model, StableDraw
from ..back.interp import softmax
from functools import partial

class TabManualAdjust:
    def __init__(self):
        self._has_widgets = False
        self._tab_main = None
    def init_widgets(self):
        if(self._has_widgets):
            return
        self.radios = []
        with gr.Tab("manual"):
            self.label = gr.Label(label="Model Contrib")
            with gr.Blocks():
                for i in range(constants.MAX_MODELS):
                    radio = gr.Radio(["-1", "+1"], type="value", visible=False)
                    self.radios.append(radio)
                with gr.Row():
                    self.btn_do_rev = gr.Button("Run-Reverse")
                    self.btn_do = gr.Button("Run")
                    self.btn_refresh = gr.Button("⟳")
            with gr.Blocks():
                self.slide_over = gr.Slider(value=1, minimum=1, maximum=5)
                self.btn_do_over = gr.Button("Over 1 contrib")
        self._has_widgets = True

    

    def register_then_show_radios(self, tch: ThenChain):
        self.init_widgets()

        def foo():
            ret = []
            for idx, r in enumerate(self.radios):
                if(idx<get_n_models()):
                    label = learning.MODELS[idx].name
                    ret.append(gr.Radio.update(label=label, visible=True, interactive=True))
                else:
                    ret.append(gr.Radio.update(visible=False))
            return ret
        tch.then(foo, None, self.radios)

    def register_then_compare(self, tch: ThenChain):
        main = self._tab_main
        def init_cmp(learning_rate, *radios):
            w = []
            for idx, r in enumerate(radios):
                if(idx<get_n_models()):
                    w.append(float(r))
            w = np.array(w)
            w = (w-w.mean())/w.std()

            cmp = learning.Comparison(
                learning.MODEL,
                learning.OPTS,
                learning_rate,
                main.random_sample_params,
                ex_w = w, ex_ratio=0.5
            )
            main.cmp = cmp

        def img0():
            image0 = main.cmp.image0

            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)

            return image0, model_w
        def img1():
            image1 = main.cmp.image1

            opt_unet = learning.OPTS[0]
            w = opt_unet.i._current_norm
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)
            
            return image1, model_w
        
        tch.showlabel(self.label, "init-cmp")\
           .then(init_cmp, [main.slide_lr]+self.radios, None)\
           .showlabel(self.label, "image0")\
           .then(img0, None, [main.image0, main.lbl0])\
           .showlabel(self.label, "image1")\
           .then(img1, None, [main.image1, main.lbl1])
        
        return tch
    def register_then_update_contrib(self, tch: ThenChain):
        self.init_widgets()

        def foo():
            opt_unet = learning.OPTS[0]
            # w = opt_unet.i._current_norm
            w = softmax(opt_unet.w)
            w_models = np.mean(w, axis=0)
            model_w = {model.name:w_models[idx] for idx, model in enumerate(learning.MODELS)}
            model_w = gr.Label.update(value=model_w)
            return model_w
        
        tch.then(foo, None, self.label)

    def register_refre(self):
        self.init_widgets()
        tch = ThenChain()
        tch.start(self.btn_refresh.click, lambda:None, None, None)
        self.register_then_show_radios(tch)
        self.register_then_update_contrib(tch)
        return tch

    def register_run(self):
        self.init_widgets()
        tch = ThenChain()
        
        def foo(forward, learning_rate, *radios):
            w = []
            for idx, r in enumerate(radios):
                if(idx<get_n_models()):
                    w.append(float(r))
            w = np.array(w)
            w = (w-w.mean())/w.std()
            print(w)
            for opt in learning.OPTS:
                # opt.step(w, learning_rate)
                opt.w += w*learning_rate*forward
            
        
        tch.start(self.btn_do.click, partial(foo, 1), [self._tab_main.slide_lr]+self.radios, None)
        self.register_then_update_contrib(tch)
        
        tch = ThenChain()
        tch.start(self.btn_do_rev.click, partial(foo, -1), [self._tab_main.slide_lr]+self.radios, None)
        self.register_then_update_contrib(tch)

        def apply_overdrive(ratio):
            for i in learning.INTERPS:
                w = i._current_norm
                w = (w-1/i.n_model)*ratio + 1/i.n_model
                i.apply(w, False)

        tch = ThenChain()
        tch.start(self.btn_do_over.click, apply_overdrive, [self.slide_over], None)
        tch.showlabel(self.label, "OK")

    def register(self):
        self.register_run()
        self.register_refre()