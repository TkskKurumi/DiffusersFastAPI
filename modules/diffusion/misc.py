import numpy as np
def normalize_weights(x, std=7.5, sum=1):
    ret = (x-x.mean())
    if(std is not None):
        ret=ret/(ret.std()+1e-10)*std
    ret += sum/np.prod(ret.shape)
    return ret
if(__name__=="__main__"):
    arr = np.array([-1, 1, 1.1, 1.2, 1.3])
    print(normalize_weights(arr))
