from super_image import MdsrModel, ImageLoader, DrlnModel, MsrnModel
from PIL import Image
from torch import no_grad
import numpy as np
import torch
model = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=3).cuda()
# model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=2).to("cuda")
# model = MsrnModel.from_pretrained('eugenesiow/msrn', scale=4).to("cuda")
def upscale(img: Image.Image):
    with no_grad():
        im = ImageLoader.load_image(img).cuda()
        pred = model(im)
        arr = ImageLoader._process_image_to_save(pred)
        arr = arr[:, :, ::-1]
        arr = np.minimum(arr, 255)
        arr = np.maximum(arr, 0)
        del pred
        del im
        return Image.fromarray(arr.astype(np.uint8))

