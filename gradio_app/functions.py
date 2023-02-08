from . import tab_inference
# from modules.diffusion import model_mgr
from modules.utils.misc import DEFAULT_RESOLUTION, normalize_resolution
_done = False


def txt2img(prompt, neg_prompt):
    pass


def adjust_aspect_ratio(a):
    w, h = normalize_resolution(a, 1)
    ret = "%d x %d" % (w, h)
    return ret


def register():
    global _done
    if (_done):
        return
    tab_inference.create()

    tab_inference.sld_aspect_ratio.change(
        adjust_aspect_ratio, inputs=tab_inference.sld_aspect_ratio, outputs=tab_inference.txt_resolution)

    # tab_inference.btn_submit.click()
