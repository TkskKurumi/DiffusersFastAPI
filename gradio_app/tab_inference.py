import gradio as gr
_created = False

def create():
    global _created, img_output, txt_prompts, txt_negprompts, txt_resolution
    global sld_aspect_ratio
    global btn_submit
    if(_created):
        return
    _created = True
    with gr.Tab("txt2img"):
        with gr.Box():
            txt_prompts = gr.Textbox(label="Prompt", interactive=True)
            txt_negprompts = gr.Textbox(label="Negative Prompt", interactive=True)
            sld_aspect_ratio = gr.Slider(label="Aspect Ratio", minimum=0.5, maximum=2, step=0.05, interactive=True)
            txt_resolution = gr.Textbox(label="Resolution", interactive=False)
            btn_submit = gr.Button(label="Submit")
        img_output = gr.Image(interactive=False)
    return btn_submit, img_output