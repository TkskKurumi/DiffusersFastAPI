import gradio as gr

SINGLETON = None

IMG0 = None
BTN0 = None
IMG1 = None
BTN1 = None

def init_widgets():
    global SINGLETON, IMG0, BTN0, IMG1, BTN1
    if(SINGLETON):
        return
    with gr.Row() as row:
        with gr.Column() as col0:
            image0 = gr.Image()
            select0 = gr.Button("This is better")
        with gr.Column() as col1:
            image1 = gr.Image()
            select1 = gr.Button("This is better")
    SINGLETON = True
    IMG0, BTN0, IMG1, BTN1 = image0, select0, image1, select1
    return
