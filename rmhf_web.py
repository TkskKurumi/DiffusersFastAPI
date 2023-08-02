from modules.rmhf_web.front import ImageRow, TabMain, TabGen, TabManualAdjust
import gradio as gr
with gr.Blocks() as demo:
    tab_main = TabMain()
    tab_gen = TabGen()
    tab_manual = TabManualAdjust()

    tab_main.set_child("tab_manual", tab_manual)
    
    tab_main.init_widgets()
    tab_gen.init_widgets()
    tab_gen.tab_main = tab_main

    load_click = tab_main.register_load()
    tab_main.register_comparison()
    tab_main.register_save()

    tab_gen.register_then_update_model_choice(load_click)
    tab_gen.register()

    tab_manual.register()
    
demo.launch(share=False)