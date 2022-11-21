from torch import cuda as cuda
import numpy as np
import torch, inspect
from os import path
class memstatus:
    def __init__(self, tot=None, rsv=None, alo=None):
        if(tot is None):
            self.tot = cuda.get_device_properties(0).total_memory
        else:
            self.tot = tot
        if(rsv is None):
            self.rsv = cuda.memory_reserved(0)
        else:
            self.rsv = rsv
        if(alo is None):
            self.alo = cuda.memory_allocated(0)
        else:
            self.alo = alo
    def format(self, n, diff=False):
        _1gb = 1<<30
        ngb = n/_1gb
        ret = "%.3fGB"%ngb
        if(diff and n>0):
            ret = "+"+ret
        return ret
    def __sub__(self, other):
        return memstatus(self.tot-other.tot, self.rsv-other.rsv, self.alo-other.alo)
    def disply_diff(self, orig):
        diff = self-orig
        fmt = self.format
        print("Allocated Memory:", fmt(orig.alo), "->", fmt(self.alo), "(", fmt(diff.alo, True), ")")
        print("Reserved  Memory:", fmt(orig.rsv), "->", fmt(self.rsv), "(", fmt(diff.rsv, True), ")")
    def format_diff(self, orig):
        ret = []
        diff = self-orig
        fmt = self.format
        ret.append("Allocated Memory: %s -> %s, %s"%(fmt(orig.alo), fmt(self.alo), fmt(diff.alo, True)))
        ret.append("Reserved  Memory: %s -> %s, %s"%(fmt(orig.rsv), fmt(self.rsv), fmt(diff.rsv, True)))
        ret.append("Total     Memory: %s"%fmt(self.tot))
        return ret
prev = memstatus()
def simplify_filename(fn):
    ls = path.split(fn)
    while(path.split(ls[0])[0]!=ls[0]):
        di, base = path.split(ls[0])
        ls = (di, base) + ls[1:]
    return path.join(*ls[-3:])
def debug_vram(title = None):
    global prev
    cur = memstatus()
    diffs = cur.format_diff(prev)
    if(title is None):
        fback = inspect.currentframe().f_back
        filename, lno, funcname, lines, idx = inspect.getframeinfo(fback)
        filename = simplify_filename(filename)
        title = "%s:%d"%(filename, lno)
    print("DEBUG: VRAM", title)
    for i in diffs:
        print("    "+i)
    prev = cur

if(__name__=="__main__"):
    debug_vram()
    t = torch.tensor(np.random.normal(0, 1, (1, 4, 512, 512))).cuda()
    debug_vram()