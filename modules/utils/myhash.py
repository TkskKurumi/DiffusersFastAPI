import numpy as np
from PIL import Image


def hashi(x, length=40):
    if (isinstance(x, Image.Image)):
        x = x.resize((10, 10), Image.Resampling.LANCZOS)
        arr = np.array(x)
        return hashi(arr, length=length)
    elif (isinstance(x, np.ndarray)):
        return hashi(list(x), length=length)
    elif (isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple)):
        ret = 0
        mask = (1 << length)-1
        offset = 7
        for i in x:
            ret = (ret << offset) ^ hashi(i, length=length)
            ret = (ret & mask) ^ (ret >> length)
        return ret
    elif(np.issubdtype(type(x), np.integer)):
        mask = (1 << length)-1
        return (x >> length) ^ (x & mask)
    elif(np.issubdtype(type(x), np.floating)):
        x = int(x*1024)
        return hashi(x, length=40)
    elif (isinstance(x, int)):
        mask = (1 << length)-1
        return (x >> length) ^ (x & mask)
    elif (isinstance(x, float)):
        return hashi(int(x*10), length)
    elif (isinstance(x, dict)):
        keys = sorted(list(x))
        return hashi([(key, x[key]) for key in keys], length=length)
    elif (isinstance(x, str)):
        return hashi([ord(i) for i in x], length=length)
    else:
        raise TypeError(type(x))
