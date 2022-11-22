from ..client import DiffuserFastAPITicket as Ticket
import numpy as np
from PIL import Image
import os
import shutil
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor

outpath = "temp_nahida"
if(os.path.exists(outpath)):
    shutil.rmtree(outpath)
os.makedirs(outpath)

negative = "/*bad anatomy, bad perspective, error, artifacts, low quality, multiple girls, error, mutation, extra legs, bad proportion, mutation, 2girls, bad fingers, extra digits, extra legs, extra limbs, poorly drawn face, text, username, blurry, sketch*/"

prompts = [
    (0, "beautiful girl, silver hair, upper body, flat chest, little girl, loli, petite /*close-up*/"+negative),
    (10, "beautiful girl, silver hair, upper body, big breats, mature female, pregnant, pregnancy /*close-up*/"+negative)
]

def get_prompt(n):
    for idx, _ in enumerate(prompts):
        i, p = _
        i1, p1 = prompts[idx+1]
        if(i<=n and n<=i1):
            ratio = (n-i)/(i1-i)
            return p, p1, ratio
    raise IndexError(n)


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
    noise = gen_noise_image(450, 800)
    nframes = 100
    nframes_tot = max(prompts)[0]

    tasks = []
    tpool = ThreadPoolExecutor(2)
    def f(idx):
        t = Ticket("txt2img_interp")
        prompt, prompt1, ratio = get_prompt(idx/nframes*nframes_tot)
        t.upload_image(noise, "noise_image")
        print(ratio, prompt)
        print(ratio, prompt1)
        t.param(prompt=prompt, prompt1=prompt1, nframes=[ratio])
        return idx, t.get_image()
    for idx in range(nframes):
        task = tpool.submit(f, idx)
        tasks.append((idx, task))
        
    for i in trange(nframes):
        idx, task = tasks[i]
        idx, im = task.result()
        im.save(os.path.join(outpath, "%03d.png"%idx))