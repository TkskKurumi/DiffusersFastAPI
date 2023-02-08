import gradio as gr

from . import tab_inference
from . import functions
with gr.Blocks() as demo:
    tab_inference.create()
    functions.register()
    