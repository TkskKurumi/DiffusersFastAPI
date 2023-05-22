from PIL import Image
from math import sqrt


def limit(im: Image.Image, width=None, height=None, area=None, resample=Image.Resampling.LANCZOS):
    ratio = 1
    w, h = im.size
    if(width is not None):
        ratio = min(ratio, width/w)
    if(height is not None):
        ratio = min(ratio, height/h)
    if(area is not None):
        ratio = min(ratio, sqrt(area/(w*h)))
    neww = int(w*ratio)
    newh = int(h*ratio)
    if(ratio != 1):
        return im.resize((neww, newh), resample)
    return im


def resize(im, width=None, height=None, area=None, resample=Image.Resampling.LANCZOS):
    ratio = 1
    w, h = im.size
    neww, newh = im.size
    if(width is not None):
        ratio = width/w
    elif(height is not None):
        ratio = height/h
    elif(area is not None):
        ratio = sqrt(area/(w*h))
    neww = int(w*ratio)
    newh = int(h*ratio)
    if(ratio != 1):
        return im.resize((neww, newh), resample)
    return im