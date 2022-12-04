# noises and frames are storaged
# useful if want change only few frames compared to previous version PromptSchedule.py
import sys
from tqdm import tqdm
import os
from functools import partial
from os import path
from client import DiffuserFastAPITicket as Ticket
from gen_noise import gen_noise_image, combine_noise
from functools import wraps
from PIL import Image
import inspect
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore
cwd = os.getcwd()
tmp = set(sys.path)
tmp.add(cwd)
sys.path = list(tmp)

from modules.utils.myhash import hashi  # nopep8
from modules.utils.candy import locked  # nopep8


def hashed(x, length=8):
    i = hashi(x, length*4)
    return hex(i)[2:].zfill(length)


working_dir = path.join(os.getcwd(), "temp_interp")


def CacheImageResult(on_create=None, on_cached=None, on_hash=None):
    LCK = Lock()
    locks = {}
    def deco(func):
        footprint = inspect.getsource(func)
        @wraps(func)
        def inner(*args, **kwargs):
            if(on_hash):
                footprints = [footprint, on_hash(*args, **kwargs)]
            else:
                footprints = [footprint, args, kwargs]
            key = hashed(footprints, 20)
            svpth = path.join(working_dir, "cache_img", key+".png")
            os.makedirs(path.dirname(svpth), exist_ok=True)
            with locked(LCK):
                if(key in locks):
                    lck = locks[key]
                else:
                    lck = Lock()
                    locks[key] = lck
            with locked(lck):
                if (path.exists(svpth)):
                    if(callable(on_cached)):
                        on_cached(svpth, *args, **kwargs)
                    return Image.open(svpth)
                else:
                    if(callable(on_create)):
                        on_create(svpth, *args, **kwargs)
                    img = func(*args, **kwargs)
                    img.save(svpth)
                    return img
        return inner
    return deco

friendly_prompt = {}

def not_satisfy(cache1create0, svpth, prompt):
    pr = friendly_prompt.get(prompt, prompt)
    print('noise for "%s" -> %s'%(pr, svpth))
    # print("noise" if cache1create0 else "creating", end=" ")
    # if(prompt in friendly_hint_prompt):
    #     print(' "%s" -> %s'%(friendly_hint_prompt[prompt], svpth))
    # else:    
    #     print('image for prompt "%s" -> %s'%(prompt, svpth))
    # print("delete it if not satisfied with generation result")
@CacheImageResult(partial(not_satisfy, 1), partial(not_satisfy, 0))
def get_prompt_noise(prompt):
    return gen_noise_image(1024, 1024)

UPLOAD_CONNECTION = Semaphore(5)
INFER_CONNECTION = Semaphore(2)
pool = ThreadPoolExecutor(10)

friendly_frame = {}

def on_create(svpth, prompt0, prompt1, ratio, noise):
    k = (prompt0, prompt1, ratio)
    
    if(k in friendly_frame):
        pr = friendly_frame[k]
        print(pr, '->', svpth)
    else:
        print(prompt0, prompt1, ratio, "->", svpth)

@CacheImageResult(on_create, on_create)
def raw_generate(prompt0, prompt1, ratio, noise):
    with locked(UPLOAD_CONNECTION):
        t = Ticket("txt2img_interp")
        t.upload_image(noise, "noise_image")
        t.param(prompt=prompt0, prompt1=prompt1, nframes=[ratio])
    with locked(INFER_CONNECTION):
        im = t.get_image()
    return im
def generate(prompt0, prompt1, ratio):
    # if noise is changed manually by user, should generate different result
    # so that cache raw_generate but not this func
    noise0 = get_prompt_noise(prompt0)
    noise1 = get_prompt_noise(prompt1)
    noise = combine_noise(noise0, noise1, 1-ratio)
    img = raw_generate(prompt0, prompt1, ratio, noise)
    return img

if (__name__ == "__main__"):
    negative = "/*bad anatomy, bad perspective, error, artifacts, low quality, multiple girls, error, mutation, extra legs, bad proportion, mutation, 2girls, bad fingers, extra digits, extra legs, extra limbs, poorly drawn face, text, username, blurry, sketch, extra legs, bad anatomy, error, blurry, field of depth*/"
    positive = "best quality, high quality, highres, masterpiece, adorable face, an extremely delicated and beautiful girl"
    character = "pink hair, cat ears, heterochromia, red eyes, blue eyes"
    common = character+positive+negative
    keyframes = [
        (0, common+"female child, little girl, loli, petite, holding lollipop, little girl, female child, full body, minigirl, indoors"),
        (1, common+"              little girl, loli, petite, yellow hat, yellow headwear, backpack"),
        (2, common+"                           loli, petite, serafuku, sailor collar, classroom"),
        (3, common+"                           loli, petite, serafuku, sailor collar, street, food, cake, holding ice cream, holding food"),
        (4, common+"                           loli, petite, one-piece swimsuit, school swimsuit, pool, wet, name tag, swimsuit, blue swimsuit, highleg, skintight"),
        (5, common+"                           loli, petite, serafuku, sailor collar, rooftop, cityscape, city light, backlighting"),
        (6, common+"                           loli, petite, gym shorts, buruma, sports shorts, gym uniform, name tag, outdoors, sweating"),
        (7, common+"                           loli, petite, serafuku, sailor collar, jacket, lapels, high school"),
        (8, common+"                           loli, petite, serafuku, swimsuit, bikini, frilled bikini, thong bikini, thighs, thigh straps, bikini skirt, beach, ocean"),
        (9, common+"                           loli, petite, school uniform, black jacket, black hat"),
        (10, common+"petite, office lady, black jacket, lapels, business suit, glasses"),
    ]
    for idx, p in keyframes:
        friendly_prompt[p] = "keyframe%03d"%idx
    loop = True
    last_frame = False
    nframes = 150
    if(loop):
        k = keyframes[-1][0]*2-keyframes[-2][0]
        p = keyframes[0][1]
        keyframes.append((k, p))
    print(keyframes)
    
    def get_frame_prompt(n):
        for idx, _ in enumerate(keyframes):
            i, p = keyframes[idx]
            i1, p1 = keyframes[idx+1]
            if(i<=n and n<=i1):
                r = (n-i)/(i1-i)
                return p, p1, r
    
    nkeys = max(keyframes)[0]
    svdir = path.join(working_dir, hashed([keyframes, nframes], 8))
    os.makedirs(svdir, exist_ok=True)
    wid, hei = None, None
    def get_frame(n):
        global wid, hei
        p0, p1, r = get_frame_prompt(n/nframes*nkeys)
        friendly_frame[(p0, p1, r)] = "frame%03d"%n
        img = generate(p0, p1, r)
        svpth = path.join(svdir, "%03d.png"%n)
        img.save(svpth)
        wid, hei = img.size
        return svpth
    tasks = []
    for i in range(nframes):
        tasks.append(pool.submit(get_frame, i))
    if(last_frame):
        tasks.append(pool.submit(get_frame, nframes))
    for t in tqdm(tasks):
        print(t.result())
    
    target = 5<<20
    rate = (target/(wid*hei*nframes*0.12))**0.5
    rate = min(1, rate)
    while(True):
        w, h = wid*rate, hei*rate
        input_images = path.join(svdir, "*.png")
        output_path = path.join(svdir, "output.gif")
        ret = os.system("gifski %s --width %d --height %d --fps 8 -o %s"%(input_images, w, h, output_path))
        if(ret!=0):
            print("using gifski to make gif but seem it's not installed")
        cur = os.path.getsize(output_path)
        if(cur<target):
            break
        else:
            rate *= 0.97
            rate *= (target/cur)**0.5
    print(output_path)
    print(svdir)
