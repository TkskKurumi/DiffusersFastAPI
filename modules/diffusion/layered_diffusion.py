from .model_mgr import model_mgr
from .model_mgr import load_model
from .wrapped_pipeline import CustomPipeline, normalize_resolution, normalize_weights, DEFAULT_DEVICE, image_as_tensor
from .parse_weights import WeightedPrompt
from PIL import Image, ImageFilter
from types import NoneType
from typing import Union, List, Callable
import torch, os, random
from os import path
from math import sqrt
import numpy as np
from ..rmhf_web.back.permute import sample_text
import qrcode
DEBUG = False

def make_qr_with_mask(data, box_size=10, border=1, fdist=lambda x:1-x, l1=True):
    qrmake = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border
    )
    qrmake.add_data(data)
    qr = qrmake.make_image().convert("L")
    w, h = qr.size
    def dist_to_center(x, y, l1=l1):
        x1 = x%box_size
        y1 = y%box_size
        ctr = (box_size-1)/2
        xdist = abs(x1-ctr)/ctr
        ydist = abs(y1-ctr)/ctr
        if (l1):
            return max(xdist, ydist)
        else:
            l2 = (xdist**2 + ydist**2)**0.5
            return min(l2, 1)
    mask_arr = np.zeros((h, w), dtype=np.float16)
    for y in range(h):
        for x in range(w):
            dist = dist_to_center(x, y)
            beta = fdist(dist)
            mask_arr[y, x] = beta*255
    return qr, Image.fromarray(mask_arr.astype(np.uint8))
    

class Layer:
    beta: Union[NoneType, float] = None
    beta_mask: Union[NoneType, Image.Image] = None
    image: Union[NoneType, Image.Image] = None
    prompt: Union[NoneType, str] = None
    xpos: float = 0.5
    ypos: float = 0.5
    mixture: Union[NoneType, str, Callable] = None
    preserve_mean: bool = False
    def __init__(self, preserve_mean: bool = False, mixture: Union[NoneType, str] = None, beta: Union[NoneType, float]=None, beta_mask: Union[NoneType, Image.Image] = None, image: Union[NoneType, Image.Image] = None, prompt: Union[NoneType, str] = None, xpos: float = 0.5, ypos: float = 0.5):
        self.beta = beta
        if ((mixture is not None) and (image is not None) and (beta is not None) and (beta_mask is None)):
            if (callable(mixture)):
                beta_mask = mixture(image, beta)
            elif(mixture == "dark"):
                arr = np.array(image.convert("L")).astype(np.float16)
                arr = (255-arr)*beta
                beta_mask = Image.fromarray(arr.astype(np.uint8))
            else:
                arr = np.array(image.convert("L")).astype(np.float16)
                arr = arr*beta
                beta_mask = Image.fromarray(arr.astype(np.uint8))
            self.beta_mask = beta_mask
        else:
            self.beta_mask = beta_mask
        self.image = image
        self.prompt = prompt
        self.xpos = xpos
        self.ypos = ypos
        self.preserve_mean = preserve_mean
        self._cond_cfg = None
        self._image_latents = None
        self._mask_tensor = None
    
    def asdict(self):
        foo = lambda **kwargs:kwargs
        return foo(
            beta = self.beta,
            beta_mask = self.beta_mask,
            image = self.image,
            prompt = self.prompt,
            xpos = self.xpos,
            ypos = self.ypos,
            mixture = self.mixture,
            preserve_mean = self.preserve_mean
        )
    def expand(self, width, height):
        def foo(im, bg):
            w, h = im.size
            left, top = int((width-w)*self.xpos), int((height-h)*self.ypos)
            ret = Image.new(im.mode, (width, height), bg)
            ret.paste(im, box=(left, top))
            return ret
        beta_mask = self.beta_mask
        if (beta_mask is not None):
            beta_mask = foo(beta_mask, 0)
        image = self.image
        if (self.image is not None):
            if (beta_mask is None):
                beta_mask = Image.new("L", self.image.size, int(self.beta*255))
                beta_mask = foo(beta_mask, 0)
            image = foo(image.convert("RGB"), (0, )*3)
            # if (DEBUG):
            #     image.show()
            #     beta_mask.show()
        d = self.asdict()
        d["image"] = image
        d["beta_mask"] = beta_mask
        return Layer(**d)
    def get_cond(self, cfg, pipeline: CustomPipeline):
        cached = (self._cond_cfg is not None) and (self._cond_cfg == cfg)
        if (cached):
            return self._cond
        else:
            encoder = pipeline.text_encoder
            weights, ids = pipeline.get_ids(self.prompt)
            lweights = len(weights)
            weights = normalize_weights(weights, cfg/sqrt(lweights))
            pipeline.debug_ids(weights, ids)
            conds = []
            for idx, id in enumerate(ids):
                cond = encoder(torch.tensor(id).to(device=DEFAULT_DEVICE))[0]
                conds.append((weights[idx], cond))
            self._cond = conds
            self._cond_cfg = cfg
            return conds
    def get_image_latents(self, pipeline: CustomPipeline):
        if (self._image_latents is not None):
            return self._image_latents
        tensor = image_as_tensor(self.image).to(device=DEFAULT_DEVICE)
        latent_dist = pipeline.sd_pipeline.vae.encode(tensor).latent_dist
        latent = latent_dist.sample(generator=None)
        latent = 0.18215*latent
        self._image_latents = latent
        return self._image_latents
    def get_mask_tensor(self, w, h, device=None):
        if(self._mask_tensor is not None):
            return self._mask_tensor
        if (self.beta_mask is None):
            self.beta_mask = Image.new("L", (w, h), int(self.beta*255))
        im = self.beta_mask.resize((w//8, h//8), Image.LANCZOS)
        if (DEBUG):
            im.show()
        arr = np.array(im).astype(np.float16)/255
        arr = np.tile(arr, (4, 1, 1))
        arr = arr[None].transpose(0, 1, 2, 3)  # what does this step do?
        tensor = torch.from_numpy(arr)
        if (device is not None):
            tensor = tensor.to(device=device)
        self._mask_tensor = tensor
        return tensor

    def step(self, latents, step, next_step, cfg, pipeline: CustomPipeline, width, height):
        if (self.prompt):
            conds = self.get_cond(cfg, pipeline)
            noise_preds = []
            for w, cond in conds:
                noise_pred = pipeline.unet(latents, step, encoder_hidden_states=cond).sample
                noise_preds.append((w, noise_pred))
            noise_pred = None
            for w, n in noise_preds:
                if (noise_pred is None):
                    noise_pred = w*n
                else:
                    noise_pred = noise_pred + w*n
            return pipeline.sched.step(noise_pred, step, latents).prev_sample
        elif (self.image):
            return self.get_image_latents(pipeline)
        else:
            raise ValueError()
@torch.no_grad()
def layered_diffusion(pipeline: CustomPipeline, layers: List[Layer], steps=30, width=512, height=768, cfg=8):
    p = []
    for l in layers:
        if (l.prompt is not None):
            p.append(l.prompt)
    loras = WeightedPrompt(", ".join(p)).loras
    loras = [(w, name) for name, w in loras.items()]
    lora_u, lora_t = pipeline.wrap_lora(*loras)

    with torch.autocast("cuda"), pipeline.inference_lock, lora_u, lora_t:
        text_encoder = pipeline.text_encoder
        unet = pipeline.unet
        sched = pipeline.sched
        vae = pipeline.sd_pipeline.vae

        latents_shape = (1, unet.in_channels, height//8, width//8)
        latents = torch.randn(latents_shape, generator=None, device=pipeline.device)
        
        layers = [layer.expand(width, height) for layer in layers]

        sched.set_timesteps(steps)
        ts_tensor = sched.timesteps.to(pipeline.device)

        for i, t in enumerate(pipeline.sd_pipeline.progress_bar(ts_tensor)):
            if (i==len(ts_tensor)-1):
                nextt = 0
            else:
                nextt = ts_tensor[i+1]
            latents_prev = latents
            for ldx, layer in enumerate(layers):
                step_forward = layer.step(latents_prev, t, nextt, cfg, pipeline, width, height)
                if (ldx==0):
                    latents = step_forward
                else:
                    if (layer.prompt):
                        mask = layer.get_mask_tensor(width, height, device=pipeline.device)
                        latents = latents*(1-mask) + step_forward*mask
                    elif (layer.image):
                        mask = layer.get_mask_tensor(width, height, device=pipeline.device)
                        if (layer.preserve_mean):
                            mean_mask = mask!=0
                            shape = mask.shape
                            div = mean_mask.sum()/4
                            latents_mean = (latents*mean_mask).sum(axis=-1, keepdim=True).sum(axis=-2, keepdim=True)/div
                            image_mean = (step_forward*mean_mask).sum(axis=-1, keepdim=True).sum(axis=-2, keepdim=True)/div
                            # print(latents_mean, image_mean)
                            step_forward = step_forward-image_mean+latents_mean



                        mask_orig = 1-mask
                        mask_orig = mask_orig**(1/steps)
                        latents = latents*mask_orig + step_forward*(1-mask_orig)
        
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = pipeline.sd_pipeline.numpy_to_pil(image)
        return image[0]
if (__name__=="__main__"):
    load_model("RMHF", r"D:\StableDiffusion\BaseModels\RMHF\RMHF-Anime-V4\rmhf.safetensors")
    _model = model_mgr._INFER_MODEL
    model = CustomPipeline(_model, lora_load_path=r"E:\Diffusion-FastAPI\Models\LoRA")
    
    w = 768
    w, h = w, int(w/5*7)

    prompt_sample = ["concat",
        ["branch",
            "cat ears, pink hair, odd eyes, heterochromia, red eye, blue eye, 1girl, /*2girls, multiple girls, multiple views, */",
            # "cat ears, blue eyes, ",
            "black hair, red eyes, cat ears, petite, short hair, low twintails, /*mature female, milf, */", 
            "wolf ears, animal ears fluff, tiara, gray hair, shawl, "
        ],
        "best quality, absurdres, /*worst quality, bad anatomy, lowres, broken arms, broken legs, bad perspective, error, mutation, */",
        ["branch",
            "dappled sunlight, window shade, indoors, dramatic shadow, shade, rays of light, ",
            "forest, trees, dark, dramatic shadow, dappled sunlight, rays of light, shade, ",
            "seaside, beach, white clouds, sea, ocean, water, front-tie bikini top, front-tie bikini bra, side-tie bikini panties, side-tie knots, front-tie knots, criss-cross halter, halterneck, "
        ],
        ["branch",
            "<lora:cute_yuki_mini:0.8><lora:next_gen_mini:0.2><lora:fantexi_mini:0.15>",
            "",
            "<lora:cute_yuki_mini:0.2><lora:next_gen_mini:0.8><lora:fantexi_mini:0.15>"
        ]
    ]
    def foo(lo, hi):
        def inner(image, beta):
            arr = np.array(image.convert("L")).astype(np.float16)/255
            arr = lo+arr*(hi-lo)
            arr = (arr*255*beta).astype(np.uint8)
            return Image.fromarray(arr)
        return inner
    def randfloat(lo, hi):
        return lo + (hi-lo)*random.random()
    def clamp(lo, hi, x):
        return min(max(x, lo), hi)
    learn = False
    learn_beta_0 = 0.85
    learn_beta_1 = 0.85
    lr = 0.5
    i = -1
    niter = 30

    
    for i in range(niter):
        info = ["%04d"%i]
        qrw = int(w * randfloat(0.33, 0.55))
        info.append("qr_size=%d"%qrw)
        if (random.random()<0.67):
            xpos = 1
            ypos = 1
        else:
            xpos = randfloat(0, 1)
            ypos = randfloat(0, 1)
        info.append("pos=(%.2f,%.2f)"%(xpos, ypos))

        image, mask = make_qr_with_mask("https://b23.tv/spIRKln")
        if (i%2):
            if (learn):
                beta_dark = clamp(0.1, 1, learn_beta_0)
            else:
                beta_dark = randfloat(0.5, 0.9)
            beta_light = beta_dark*0.7
            learn01 = 0
            mask = foo(beta_dark, beta_light)(image, 1)
            info.append("beta_dark=%.2f_beta_light=%.2f"%(beta_dark, beta_light))
        else:
            if (learn):
                beta_inner = clamp(0.1, 1, learn_beta_1)
            else:
                beta_inner = randfloat(0.5, 0.9)
            beta_outer =0.7*beta_inner
            learn01 = 1
            fdist = lambda x: beta_inner+(beta_outer-beta_inner)*x
            image, mask = make_qr_with_mask("https://b23.tv/spIRKln", fdist=fdist)
            info.append("beta_inner=%.2f_beta_outer=%.2f"%(beta_inner, beta_outer))
        image = image.resize((qrw, qrw))
        mask = mask.resize((qrw, qrw), Image.LANCZOS)
        prompt = sample_text(prompt_sample)
        layer0 = Layer(prompt=prompt)
        layer1 = Layer(image=image, beta_mask=mask, beta=1, xpos=xpos, ypos=ypos, preserve_mean=True)
        result = layered_diffusion(model, [layer0, layer1], width=w, height=h)
        os.makedirs("test_ld", exist_ok=True)
        pth = path.join("test_ld", "%04d_mask.png"%i)
        mask.save(pth)
        pth = path.join("test_ld", "_".join(info)+".png")
        result.save(pth)
        print(pth)
        if (learn):
            import cv2
            qcd = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(np.array(result))
            if (retval):
                if (learn01==0):
                    learn_beta_0 -= lr
                else:
                    learn_beta_1 -= lr
                lr *= 0.95
            else:
                if (learn01==0):
                    learn_beta_0 += lr
                else:
                    learn_beta_1 += lr
                lr *= 0.95
            print("detect", retval, decoded_info)

elif (__name__=="__main__"):
    load_model("RMHF", r"D:\StableDiffusion\BaseModels\RMHF\RMHF-Anime-V4\rmhf.safetensors")
    _model = model_mgr._INFER_MODEL
    model = CustomPipeline(_model, lora_load_path=r"E:\Diffusion-FastAPI\Models\LoRA")
    
    w = 768
    w, h = w, int(w/5*7)

    prompt_sample = ["concat",
        ["branch",
            "cat ears, pink hair, odd eyes, heterochromia, red eye, blue eye, 1girl, /*2girls, */",
            "cat ears, blue eyes, ",
            "black hair, red eyes, cat ears, petite, /*mature female, milf, */"
        ],
        "best quality, absurdres, /*worst quality, bad anatomy, lowres, */",
        "dappled sunlight, window shade, indoors, dramatic shadow, shade, rays of light, ",
        ["branch",
            "<lora:cute_yuki_mini:0.8><lora:next_gen_mini:0.2><lora:fantexi_mini:0.15>",
            "",
            "<lora:cute_yuki_mini:0.2><lora:next_gen_mini:0.8><lora:fantexi_mini:0.15>"
        ]
    ]
    def foo(lo, hi):
        def inner(image, beta):
            arr = np.array(image.convert("L")).astype(np.float16)/255
            arr = lo+arr*(hi-lo)
            arr = (arr*255*beta).astype(np.uint8)
            return Image.fromarray(arr)
        return inner
    def randfloat(lo, hi):
        return lo + (hi-lo)*random.random()
    
    for i in range(25):
        qrw = int(w * randfloat(0.3, 0.5))
        xpos, ypos = random.random(), random.random()
        ypos = 1-ypos*ypos
        prompt = sample_text(prompt_sample)
        layer0 = Layer(prompt=prompt)
        
        beta_dark = 1
        beta_light = randfloat(0.3, 0.7)

        mixture = foo(beta_dark, beta_light)
        beta = randfloat(0.65, 0.95)

        print(prompt)
        if (random.random()<0.5):
            info_layer1 = "beta=(%.3f,%.3f)_size=%d"%(beta*beta_dark, beta*beta_light, qrw)
            img = Image.open(r"E:\Download\qr_to_baidu.png").resize((qrw, qrw)).convert("RGB")
            layer1 = Layer(image=img, beta=beta, mixture=mixture, xpos=xpos, ypos=ypos, preserve_mean=True)
        else:
            beta_light = randfloat(0.3, 0.7)
            info_layer1 = "makeqr_beta=(%.2f, %.2f)"%(beta*beta_dark, beta*beta_light)
            
            img, mask = make_qr_with_mask("https://baidu.com", fdist = lambda x:beta*(beta_dark+x*(beta_light-beta_dark)))
            img = img.resize((qrw, qrw)).convert("RGB")
            mask = mask.resize((qrw, qrw), Image.LANCZOS)
            mask.save("tmp_mask.png")
            layer1 = Layer(image=img, beta_mask=mask, xpos=xpos, ypos=ypos, preserve_mean=True, beta=beta)
        result = layered_diffusion(model, [layer0, layer1], width=w, height=h)
        os.makedirs("test_ld", exist_ok=True)

        pth=path.join("test_ld", "%04d_%s.png"%(i, info_layer1))
        result.save(pth)
        print(pth)
elif (__name__=="__main__"):
    load_model("RMHF", r"D:\StableDiffusion\BaseModels\RMHF\RMHF-Anime-V4\rmhf.safetensors")
    _model = model_mgr._INFER_MODEL
    model = CustomPipeline(_model, lora_load_path=r"E:\Diffusion-FastAPI\Models\LoRA")
    w = 768
    w, h = w, int(w/5*7)
    
    
    chara = "cat ears, pink hair, odd eyes, heterochromia, red eyes, blue eyes, 1girl, /*2girls, */"
    # chara = "cha ears, blue eyes, "
    quality_control = "best quality, absurdres, /*worst quality, bad anatomy, lowres, */"
    scena = "dappled sunlight, window shade, indoors, dramatic shadow, shade, "
    style = "<lora:cute_yuki_mini:0.8><lora:next_gen_mini:0.2><lora:fantexi_mini:0.15>"
    p = chara + quality_control + scena + style
    layer0 = Layer(prompt=p)

    

    def foo(lo, hi):
        def inner(image, beta):
            arr = np.array(image.convert("L")).astype(np.float16)/255
            arr = lo+arr*(hi-lo)
            arr = (arr*255*beta).astype(np.uint8)
            return Image.fromarray(arr)
        return inner
    
    def border(lo, hi):
        def inner(image: Image.Image, beta):
            w, h = image.size
            rad = ((w*w+h*h)**0.5)/100
            print('radius', rad)
            im_blur = image.filter(ImageFilter.GaussianBlur(rad))
            arr0 = np.array(image).astype(np.float32)
            arr1 = np.array(im_blur).astype(np.float32)
            diff = arr0-arr1
            diff = np.sqrt((diff**2).sum(axis=-1))
            diff = (diff-diff.min())/(diff.max()-diff.min())
            diff = diff*(hi-lo)+lo
            diff = diff*255*beta
            ret = Image.fromarray(diff.astype(np.uint8))
            return ret
        return inner

    idx = 0
    lo = 0.7
    hi = 0.8
    steps = 25
    for pm in [True]:
        for i in range(steps):
            beta = lo+(hi-lo)*random.random()
            xpos = random.random()
            ypos = random.random()
            for mixture in (foo(1, 0.6),):


                qrw = int(w*(0.3+random.random()*0.4))
                img = Image.open(r"E:\Download\qr_to_baidu.png").resize((qrw, qrw)).convert("RGB")

                layer1 = Layer(image=img, beta=beta, mixture=mixture, xpos=xpos, ypos=ypos, preserve_mean=pm)
                result = layered_diffusion(model, [layer0, layer1], width=w, height=h)
                os.makedirs("test_ld", exist_ok=True)

                pth=path.join("test_ld", "%04d_beta=%.3f_pm=%s_x=%.2f_y=%.2f.png"%(idx, beta, pm, xpos, ypos))
                result.save(pth)
                idx += 1
                print(pth)
