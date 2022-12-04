import numpy as np
from PIL import Image
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
def combine_noise(A, B, rateA):
    arr = np.array(A)
    arr = (arr-arr.mean())/arr.std()
    brr = np.array(B)
    brr = (brr-brr.mean())/brr.std()
    noise = arr*rateA+brr*(1-rateA)
    noise = scale_min_max(noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noise)