import numpy as np
from .interp import Interpolate
def check_castable(arr0, arr1):
    shape0 = arr0.shape
    shape1 = arr1.shape
    if(len(shape0) > len(shape1)):
        return check_castable(arr1, arr0)
    print(shape0, shape1)
    for idx, i in enumerate(shape0):
        jdx = idx-len(shape0)
        j = shape1[jdx]
        if(i!=1 and j!=1 and i!=j):
            return False
    return True


class Optimize:
    def __init__(self, i: Interpolate, initial=None):
        self.i = i
        if(initial is None):
            self.w = i.zeros()
        else:
            z = i.zeros()
            if(not check_castable(z, np.array(initial))):
                print("WARNING: not using initial weight for", i.name, "since shape doesn't match", z.shape, np.array(initial).shape)
                self.w = z
            else:
                initw = str(initial).replace("\n", "\\n ")
                if(len(initw)>255):
                    initw = initw[:252]+"..."
                print("init", i.name, initw)
                self.w = i.zeros()+initial
        print("init", self.i.name, self.w.mean(axis=0))
        self.i.apply(self.w)
    def step(self, w, lr):
        raise NotImplementedError()
    def dummy_step(self, w, lr):
        raise NotImplementedError()
    def randn(self):
        return self.i.randn()
    @property
    def momentum(self):
        raise NotImplementedError()
    

class OptimizeLionLike(Optimize):
    def __init__(self, i: Interpolate, initial=None, beta1=0.95, beta2=0.5, decay=0.01):
        super().__init__(i, initial)
        self.init_w = self.w
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.m1 = i.zeros()
    def dummy_step(self, w, lr):
        c = self.m1*self.beta2 + w*(1-self.beta2)
        m1 = self.m1*self.beta1 + w*(1-self.beta1)
        
        forward = np.sign(c)-self.decay*(self.w-self.init_w)

        w = self.w+lr*forward
        return w, (c, m1, forward, w)
    def step(self, w, lr):
        w, states = self.dummy_step(w, lr)
        c, self.m1, forward, w = states
        print(self.i.name, "moving", np.abs(lr*forward).mean())
        self.w = w
        return w
    @property
    def momentum(self):
        return self.m1