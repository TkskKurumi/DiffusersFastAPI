import gradio as gr
import traceback
from ..constants import MAX_ROW_IMAGES
from functools import partial
from ..util import logging
class ThenChain:
    def __init__(self):
        self.exception = None
        self.tail = None
    def setok(self):
        self.exception = None
    def start(self, foo, fun, inputs, outputs, *args, **kwargs):
        assert self.tail is None
        self.tail = foo(self.setok, None, None)
        self.then(fun, inputs, outputs, *args, **kwargs)
        return self
    def isok(self):
        return self.exception is None
    def then(self, fun, inputs, outputs, *args, **kwargs):
        def inner(*args, **kwargs):
            if (not self.isok()):
                return
            try:
                ret = fun(*args, **kwargs)
            except Exception as e:
                traceback.print_exc()
                self.exception = e
                return
            return ret
        self.tail = self.tail.then(inner, inputs, outputs, *args, **kwargs)
        return self
    def showlabel(self, label_widget, msg=None):
        def inner():
            if(not self.isok()):
                return repr(self.exception)
            if(msg is not None):
                return msg
            return logging.LAST
        self.tail = self.tail.then(inner, None, [label_widget])
        return self
class ImageRow:
    def __init__(self, row0col1 = False):
        self._has_widgets = False
        self._row0col1 = row0col1
    def init_widgets(self):
        if(self._has_widgets):
            return
        if(self._row0col1):
            row_or_col = gr.Column
        else:
            row_or_col = gr.Row
        with row_or_col() as row:
            self.imgs = []
            self.label = gr.Label(visible=False)
            for i in range(MAX_ROW_IMAGES):
                with gr.Column() as col:
                    image = gr.Image(visible=False)
                    self.imgs.append(image)
    def register_then(self, tch: ThenChain):
        # draw_prelabel
        # draw
        def _draw(idx):
            print(_draw, idx)
            if(hasattr(self, "draw")):
                dr = self.draw
                result = dr(idx)
                if(result is not None):
                    ok, img, label = result
                    if(not ok):
                        return gr.Image.update(visible=False)
                    else:
                        if (not label):
                            label = str(idx)
                        return gr.Image.update(visible=True, value=img, label=label)
                else:
                    return gr.Image.update(visible=False)
            else:
                return gr.Image.update(visible=False)
        def _draw_pre(idx):
            print(_draw_pre, idx)
            if(hasattr(self, "draw_prelabel")):
                label = self.draw_prelabel(idx)
                return gr.Label.update(visible=True, value=label)
            return None
        
        for idx, i in enumerate(self.imgs):
            tch.then(partial(_draw_pre, idx), None, self.label)
            tch.then(partial(_draw, idx), None, i)
        return tch
