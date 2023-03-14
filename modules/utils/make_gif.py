import os
from os import path
from .myhash import hashi
from .misc import normalize_resolution
import tempfile

def hashed(obj, length=10):
    i = hashi(obj, length*4)
    return hex(i)[2:]
def make_gif(images, exec="gifski", fps=8, max_filesize = 5<<20, outfilename = None):
    w, h = images[0].size
    area = w*h
    area = min(area, filesize/len(images)/0.6)
    tempdir = tempfile.gettempdir()
    files = []
    for im in images:
        filename = path.join(tempdir, hashed(im)+".png")
        im.save(filename)
        files.append(filename)
    if(outfilename is None):
        outfilename = hashed(images)+".gif"
    while(True):
        scripts = [exec, "--fps", fps, "--no-sort"]
        scripts.extend(files)
        neww, newh = normalize_resolution(w, h, area, 1)
        scripts.extend(["--width", neww, "--height", newh])
        scripts.extend(["-o", outfilename])
        scripts = " ".join([str(i) for i in scripts])
        print(scripts)
        syscode = os.system(scripts)
        if(syscode!=0):
            raise OSError("failed to execute gifski %d"%syscode)
        filesize = path.getsize(outfilename)
        if(filesize<max_filesize):
            return outfilename
        else:
            print("filesize/(area*nframe)", filesize/area/len(images))
            area *= 0.97
            area *= (max_filesize/filesize)**0.5