from ..client import DiffuserFastAPITicket as Ticket
import numpy as np
from PIL import Image
import os
import shutil
from tqdm import trange, tqdm
from concurrent.futures import ThreadPoolExecutor

outpath = "temp_interp_loop"
if(os.path.exists(outpath)):
    shutil.rmtree(outpath)
os.makedirs(outpath)

negative = "/*bad anatomy, bad perspective, error, artifacts, low quality, multiple girls, error, mutation, extra legs, bad proportion, mutation, 2girls, bad fingers, extra digits, extra legs, extra limbs, poorly drawn face, text, username, blurry, sketch, extra legs, bad anatomy, error, blurry, field of depth*/"
positive = "best quality, high quality, highres, masterpiece, adorable face, an extremely delicated and beautiful girl"
character = "pink hair, cat ears, heterochromia, red eyes, blue eyes"
common = character+positive+negative
prompts = [
    [0, common+"female child, little girl, loli, petite, flat chest, holding lollipop, holding food, from above"],
    [1, common+"              little girl, loli, petite, small breasts, serafuku, sailor collar, yellow hat"],
    [2, common+"                           loli, petite, small breasts, black skirt, school uniform, pleated skirt"],
    [3, common+"medium breasts, black skirt, miniskirt, business suit, formal, office lady, lapels, suit jacket"],
    [4, common+"big breasts, open clothes, breasts out, drunk, shy, blush, evil smile, holding drink, alcohol, /*bad anatomy, error, upside-down, bad perspective*/"],
    [5, common+"big breasts, open clothes, nipples, sex, cum, bukkake, cum on body, cum in pussy, cumdrip, navel, pussy, wide spread leg, hetero sex, half closed eyes, cum in pussy, crying, indoors, on bed"],
    [6, common+"medium breasts, blush, shy, after sex, cum on body, from side, looking to the side, embarrased, messy clothes, adjusting clothes"],
    [7, common+"medium breasts, smile, leaning forward, pov hands, reaching out, white dress, frilled dress"],
    [8, common+"medium breasts, wedding dress, veil, frills, holding bouquet, flowers, white dress, strapless dress, veil, headdress, wedding"],
]
prompts = [i+[np.random.normal(0, 1, (100, 100, 4)), ] for i in prompts]
if(False):
    prompts.append(list(prompts[0]))
    prompts[-1][0] = prompts[-2][0]*2-prompts[-3][0]
print([i[0] for i in prompts])
def get_prompt(n):
    for idx, _ in enumerate(prompts):
        i, p, noi = _
        i1, p1, noi1 = prompts[idx+1]
        if(i<=n and n<=i1):
            ratio = (n-i)/(i1-i)
            print(i, n, i1, ratio)
            noise = noi1*ratio+noi*(1-ratio)
            print("noise", noise.mean(), noise.std())
            im = scale_min_max(noise, 0, 255)
            im = Image.fromarray(im.astype(np.uint8))
            
            return p, p1, ratio, im
    raise IndexError(n)


def scale_min_max(arr: np.ndarray, lo=0, hi=1):
    mn = arr.min()
    mx = arr.max()
    return (arr-mn)/(mx-mn)*(hi-lo)+lo
def gen_noise_image(w=512, h=512, arr=None):
    shape=(h//8, w//8, 4)
    if(arr is None):
        arr = np.random.normal(0, 1, shape)
    arr = scale_min_max(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img

if(__name__=="__main__"):
    # noise = gen_noise_image(450, 800)
    nframes = 150
    nframes_tot = max(prompts)[0]

    tasks = []
    submit_pool = ThreadPoolExecutor(4)
    infer_pool = ThreadPoolExecutor(2)
    def submit(idx):
        t = Ticket("txt2img_interp")
        prompt, prompt1, ratio, noise = get_prompt(idx/(nframes-1)*nframes_tot)
        t.upload_image(noise, "noise_image")
        # print(ratio, prompt)
        # print(ratio, prompt1)
        t.param(prompt=prompt, prompt1=prompt1, nframes=[ratio], guidance=12)
        return infer_pool.submit(t.get_image)
    for idx in range(nframes):
        task = submit_pool.submit(submit, idx)
        tasks.append((idx, task))
    for idx, task in tqdm(tasks):
        img = task.result().result()        
        pth = os.path.join(outpath, "%03d.png"%idx)
        img.save(pth)
        print(pth)