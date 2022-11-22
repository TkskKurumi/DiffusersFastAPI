from .client import DiffuserFastAPITicket
import numpy as np
from PIL import Image
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
def scale_normal(arr: np.ndarray):
    return (arr-arr.mean())/arr.std()

if(__name__=="__main__"):
    noise = gen_noise_image(1024, 1024)
    pr = "seaside, beach side. An extremely delicated and beautiful girl is wearing bikini."
    
    t1 = DiffuserFastAPITicket("txt2img")
    t1.param(prompt=pr, aspect=1, neg_prompt="bad anatomy, lowres, blurry, low quality")
    t1.upload_image(noise, "noise_image")

    t2 = DiffuserFastAPITicket("txt2img")
    t2.param(prompt=pr, aspect=1, neg_prompt="bad anatomy, lowres, blurry, low quality")
    t2.upload_image(noise, "noise_image")

    t3 = DiffuserFastAPITicket("txt2img")
    t3.param(prompt=pr, aspect=0.78, neg_prompt="bad anatomy, lowres, blurry, low quality")
    t3.upload_image(noise, "noise_image")

    t1.get_image().save("tmp0.png")
    t2.get_image().save("tmp1.png")
    t3.get_image().save("tmp2.png")
