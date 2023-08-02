import gradio as gr
import time
from functools import partial
with gr.Blocks() as demo:
    label = gr.Label()
    btn = gr.Button()
    def fok(t):
        return t
    def fsleep():
        time.sleep(1)
    btn.click(partial(fok, "loading"), None, label).then(fsleep).then(partial(fok, "OK"), None, label)
    

demo.launch()