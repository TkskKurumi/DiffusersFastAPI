from ..client import DiffuserFastAPITicket as Ticket
import numpy as np
from PIL import Image
import os
import shutil
outpath = "temp"
if(os.path.exists(outpath)):
    shutil.rmtree(outpath)
os.makedirs(outpath)

def scale_min_max(arr: np.ndarray, lo=0, hi=1):
    mn = arr.min()
    mx = arr.max()
    return (arr-mn)/(mx-mn)*(hi-lo)+lo
def gen_noise_image(w=512, h=512):
    shape=(h//8, w//8, 4)
    arr = np.random.normal(0, 1, shape)
    arr = scale_min_max(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img
if(__name__=="__main__"):
    prompt_A = "an extremely delicated and beautiful girl with red hair, wearing black dress/*lowres, bad quality, error, blurry, bad anatomy, mutation*/"
    prompt_B = "an extremely delicated and beautiful girl with blue hair, wearing black dress/*lowres, bad quality, error, blurry, bad anatomy, mutation*/"
    prompt_C = "an extremely delicated and beautiful girl with silver hair, wearing white dress/*lowres, bad quality, error, blurry, bad anatomy, mutation*/"
    noise = gen_noise_image(1024, 1024)
    t1 = Ticket("txt2img_interp")
    t1.param(prompt=prompt_A, prompt1=prompt_B, nframes=15)
    t1.upload_image(noise, "noise_image")
    t2 = Ticket("txt2img_interp")
    t2.param(prompt=prompt_B, prompt1=prompt_C, nframes=15)
    t2.upload_image(noise, "noise_image")
    
    images = t1.get_image_seq()
    images.extend(t2.get_image_seq()[1:])

    for idx, i in enumerate(images):
        pth = os.path.join(outpath, "%02d.png"%idx)
        i.save(pth)
    