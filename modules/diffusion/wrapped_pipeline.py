from .parse_weights import WeightedPrompt
from dataclasses import dataclass
from .misc import normalize_weights
from ..utils.debug_vram import debug_vram
from ..utils.load_tensor import load as load_tensor
from diffusers import StableDiffusionPipeline
from functools import partial
import numpy as np
import torch, json
from torch import nn
from os import path
from glob import glob
from threading import Lock
from ..utils.candy import locked
from ..utils.misc import normalize_resolution 
from PIL import Image
from math import log, ceil, sqrt
from ..utils.lcs import LCS
import os
from typing import Callable, Iterable, Dict, Union, List
import safetensors
from .model_mgr import LoRA
from ..utils.misc import DEVICE as DEFAULT_DEVICE
def combine_noise(noise: torch.tensor, use_noise:np.ndarray, use_noise_alpha:float):
    dtype = noise.dtype
    device = noise.device
    use_noise = torch.from_numpy(use_noise).to(device=DEFAULT_DEVICE)
    a = 1-use_noise_alpha
    b = use_noise_alpha
    norm = (a**2+b**2)**0.5
    noise = (noise*a+use_noise*b)/norm
    return noise.to(dtype).to(device)

def preprocess_image(img, resolution=None):
    size = normalize_resolution(*img.size, resolution=resolution)
    return img.resize(size, Image.Resampling.LANCZOS)

def image_as_tensor(image):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2) # channel, y, x
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0 # to -1 ~ 1

def preprocess_mask(img, mask):
    w, h=img.size
    return mask.resize((w//8, h//8), Image.Resampling.LANCZOS)
def mask_as_numpy(mask):
    mask = mask.convert("L")
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    return mask
def mask_as_tensor(mask):
    mask = torch.from_numpy(mask_as_numpy(mask))
    return mask

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0
NOISE_PRED_BATCH = 1
class IDBucket:
    def __init__(self, max_length, weight, start_id, end_id):
        self.ids = np.zeros((1, max_length), np.int64) + end_id
        self.ids[0, 0] = start_id
        self.weight = weight
        self.used = 0
        self.max_length = max_length-2
    def reinit(self):
        self.used = 0
        end_id = self.ids[0, -1]
        start_id = self.ids[0, 0]
        self.ids = np.zeros((1, self.max_length+2), np.int64) + end_id
        self.ids[0, 0] = start_id
    def add_ids(self, weight, *ids):
        n = len(ids)
        if(n+self.used > self.max_length):
            return False
        if(weight !=self.weight):
            return False
        for idx, i in enumerate(ids):
            self.ids[0, self.used+1] = i
            self.used+=1
        return True
    def pad_wrap(self, target):
        orig_used = self.used
        if(not orig_used):
            return
        while(self.used < target):
            n = self.used%orig_used
            self.add_ids(self.weight, self.ids[0, n+1])
        return

class Reproducable:
    def __init__(self, func, result: Union[Image.Image, List[Image.Image]], *args, **kwargs):
        self.func=func
        self.result=result
        self.args=args
        self.kwargs=kwargs
    def reproduce(self, **kwargs):
        kwa = {}
        kwa.update(self.kwargs)
        kwa.update(kwargs)
        return self.func(*self.args, **kwa)
    @property
    def noise(self) -> np.ndarray:
        return self.kwargs["use_noise"]
class NameAssign:
    def __init__(self):
        self.used = set()
        self.next = dict()
    def __call__(self, name):
        idx = self.next.get(name, 0)
        def concat():
            nonlocal idx, name
            return "%s_%s"%(name, idx)
        while(concat() in self.used):
            idx += 1
        ret = concat()
        self.next[name] = idx+1
        self.used.add(ret)
        return ret
def combine(weights, inputs):
    ret = 0
    for idx, w in enumerate(weights):
        ret+=w*inputs[idx]
    return ret
def split_batch(inp, batch_size=0, torch_cat=False):
    ret = []
    le = len(inp) if isinstance(inp, list) else inp.shape[0]
    for i in range(0, le, batch_size):
        bs = min(batch_size, le-i)
        batch = inp[i:i+bs]
        if(torch_cat):
            batch = torch.cat(batch)
        ret.append(batch)
    return ret
TI_NEAREST_ORIG_WORD = False




class CustomPipeline:
    def __init__(self, sd_pipeline: StableDiffusionPipeline, ti_autoload_path=None, lora_load_path=None):
        self.sd_pipeline = sd_pipeline
        
        self.raw_tokenize = partial(sd_pipeline.tokenizer,
            padding=False,
            return_tensors="np"
        )
        
        self.tokenizer = sd_pipeline.tokenizer
        self.text_encoder = sd_pipeline.text_encoder
        self.unet = sd_pipeline.unet
        self.sched = sd_pipeline.scheduler
        self.device = sd_pipeline.device

        self.hijack_lock = Lock()
        self.ti_autoload_path = ti_autoload_path
        self.lora_load_path = lora_load_path
        self.ti_alias = {}
        self.ti_alias["remote_random"] = self.random_prompt
        self.ti_loaded = {}
        self.loras = {}
        self.lora_info = {}
        self.lora_file = {}
        self.ti_ph_assign = NameAssign()
        self.orig_embd_weights = self.text_encoder.text_model.embeddings.token_embedding.weight.numpy(force=True)
        self.auto_load_ti()
        
        self.auto_load_lora()
        self.inference_lock = Lock()




    def random_prompt(self):
        st, ed = self.tokenizer("")["input_ids"]
        hi = min(st, ed)
        ids = np.random.randint(0, hi, (1, 30))
        return self.tokenizer.batch_decode(ids)
    @torch.no_grad()
    def auto_load_lora(self):
        if(not self.lora_load_path):
            return
        pth = path.join(self.lora_load_path)
        files = list()
        files.extend(glob(path.join(pth, "*.pt")))
        files.extend(glob(path.join(pth, "*.safetensors")))
        for i in files:
            name = path.splitext(path.basename(i))[0]
            if(name in self.loras):
                continue
            self.lora_file[name] = i
            pt = load_tensor(i)
            self.loras[name] = pt
        files = list()
        files.extend(glob(path.join(pth, "*", "*.pt")))
        files.extend(glob(path.join(pth, "*", "*.safetensors")))
        for i in files:
            dn = path.dirname(i)
            name = path.basename(dn)
            if(name in self.loras):
                continue
            self.lora_file[name] = i
            pt = load_tensor(i)
            self.loras[name] = pt
            
            meta = path.join(dn, "info.json")
            if(path.exists(meta)):
                with open(meta) as f:
                    j = json.load(f)
                    self.lora_info[name] = j

        for k in list(self.loras):
            if (k in self.lora_file):
                f = self.lora_file[k]
                if(not path.exists(f)):
                    self.loras.pop(k)
                    if(k in self.lora_info):
                        self.lora_info.pop(k)

    def wrap_lora(self, *args):
        self.auto_load_lora()
        state_dicts = []
        scales = []
        for w, name in args:
            if(name in self.loras):
                state_dicts.append(self.loras[name])
                scales.append(w)
        if(state_dicts):
            return LoRA.WrapLoRAUNet(self.unet, state_dicts, scales), LoRA.WrapLoRATextEncoder(self.text_encoder, state_dicts, scales)
        else:
            return LoRA.Empty(), LoRA.Empty()
    @torch.no_grad()
    def auto_load_ti(self):
        if(not self.ti_autoload_path):
            return
        tokenizer = self.tokenizer
        text_embeddings = self.text_encoder.text_model.embeddings
        ann = None
        orig_embd_weights = self.orig_embd_weights
        def nearest_words(emb):
            nonlocal ann, orig_embd_weights
            n, m = emb.shape
            if(ann is None):
                from annoy import AnnoyIndex
                ann = AnnoyIndex(m, "euclidean")
                st, ed = tokenizer("")["input_ids"]
                
                for i in range(max(st, ed)+1):
                    vec = orig_embd_weights[i]
                    print("vec id", i, end="\r")
                    ann.add_item(i, vec)
                ann.build(5)
            ret = []
            for i in range(n):
                nn = ann.get_nns_by_vector(emb[i], 1)[0]
                ret.append(nn)
            return tokenizer.batch_decode(np.array(ret).reshape((1, -1)))
        to_cat = []
        mes = []
        ls = list(glob(path.join(self.ti_autoload_path, "*.pt")))
        ls.extend(glob(path.join(self.ti_autoload_path, "*.safetensors")))

        for i in ls:
            with locked(self.hijack_lock):
                if(i in self.ti_loaded):
                    continue
                if(i.endswith("pt")):
                    pt = torch.load(i)
                else:
                    
                    pt = {}
                    with safetensors.safe_open(i, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            pt[k] = f.get_tensor(k)
                    if("emb_params" in pt):
                        bn = path.splitext(path.basename(i))[0]
                        pt["string_to_param"] = {"bn": pt["emb_params"]}
                        pt["name"] = bn
                    else:
                        print(list(pt))
                key = next(iter(pt["string_to_param"])) # first element
                embeddings = pt["string_to_param"][key].cpu().numpy() # (size, 768)
                
                placeholder_length, dim = embeddings.shape
                
                # orig_embd_weights = text_embeddings.token_embedding.weight.cpu().numpy()
                alias_key = pt["name"]
                alias_value = []
                for j in range(placeholder_length):
                    placeholder = self.ti_ph_assign("token")
                    added = tokenizer.add_tokens(placeholder)
                    while(not added):
                        placeholder = self.ti_ph_assign("token")
                        added = tokenizer.add_tokens(placeholder)
                    assert added == 1, "more than 1 id is added for placeholder %s"%placeholder
                    alias_value.append(placeholder)
                alias_value = " ".join(alias_value)
                added_ids = tokenizer(alias_value)["input_ids"]
                st, ed, added_ids = added_ids[0], added_ids[-1], added_ids[1:-1]
                to_cat.append(embeddings)
                mes.append("%s"%(alias_key,))
                if(TI_NEAREST_ORIG_WORD):
                    print(alias_key, "near", nearest_words(embeddings))
                self.ti_alias[alias_key] = alias_value
                self.ti_loaded[i] = i
        if(mes):
            print(*mes, sep=", ")
        if(to_cat):
            embd_weights = np.concatenate([orig_embd_weights]+to_cat)
            self.orig_embd_weights = embd_weights
            embd_weights = torch.tensor(embd_weights, device=DEFAULT_DEVICE)
            del text_embeddings.token_embedding
            text_embeddings.token_embedding = nn.Embedding.from_pretrained(embd_weights)

    
    def get_txt2img_multiplier(self, prompt, cfg=7.5, steps=20, width=512, height=768, neg_prompt = "", noise_pred_batch_size=NOISE_PRED_BATCH, use_noise=None, use_noise_alpha=1):
        """
        Calculate estimated runtime multiplier.
        """
        weights, ids = self.get_ids(prompt, neg_prompt = neg_prompt)
        ret = len(weights)
        ret *= width*height*steps
        return ret
    
    def get_img2img_multiplier(self, prompt, orig_image, cfg=7.5, steps=20, alpha=0.7, beta=0.9, neg_prompt="", noise_pred_batch_size=NOISE_PRED_BATCH, eta=0, use_noise=None, use_noise_alpha=1):
        weights, ids = self.get_ids(prompt, neg_prompt = neg_prompt)
        ret = len(weights)
        img = preprocess_image(orig_image)
        width, height = img.size
        ret *= width*height*steps*alpha
        return ret

    def get_inpaint_multiplier(self, prompt, orig_image, mask_image, cfg=7.5, steps=20, neg_prompt="", noise_pred_batch_size=NOISE_PRED_BATCH, use_noise=None, use_noise_alpha=1, mode=1):
        weights, ids = self.get_ids(prompt, neg_prompt = neg_prompt)
        nweights = len(weights)
        w, h = preprocess_image(orig_image).size
        return w*h*nweights*steps
    
    @torch.no_grad()
    def inpaint(self, prompt, orig_image, mask_image, cfg=7.5, steps=20, neg_prompt="", noise_pred_batch_size=NOISE_PRED_BATCH, use_noise=None, use_noise_alpha=1, mode=1):
        with torch.autocast("cuda"), locked(self.inference_lock):
            text_encoder = self.text_encoder
            unet = self.unet
            sched = self.sched
            vae = self.sd_pipeline.vae
            
            orig_image = preprocess_image(orig_image)
            orig_tensor = image_as_tensor(orig_image)
            mask_image = preprocess_mask(orig_image, mask_image)
            mask_numpy = mask_as_numpy(mask_image)
            mask_tensor = mask_as_tensor(mask_image)

            # prepare weights
            weights, ids = self.get_ids(prompt, neg_prompt = neg_prompt)
            sqrt_weightn = len(weights)**0.5
            weights = normalize_weights(weights, std=cfg/sqrt_weightn)
            self.debug_ids(weights, ids)
            conds = []
            for i in ids:
                cond = text_encoder(torch.tensor(i).to(device=DEFAULT_DEVICE))[0]
                conds.append(cond)
            cond_batches = split_batch(conds, noise_pred_batch_size, torch_cat=True)
            nconds = len(conds)

            
            # prepare latents
            latents_dtype = cond.dtype
            orig_tensor = orig_tensor.to(device=self.device, dtype=latents_dtype)
            init_latent_dist = vae.encode(orig_tensor).latent_dist  # vae distribution, vae mean and stddev...
            init_latents = init_latent_dist.sample(generator=None) # vae sample by stddev
            init_latents = 0.18215 * init_latents

            
            # prepare timestep
            init_timestep = int(steps)
            sched.set_timesteps(steps)
            init_timestep = min(init_timestep, steps)
            timesteps = sched.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps], device=self.device)

            # prepare added noise
            noise = torch.randn(init_latents.shape, generator=None, device=self.device, dtype=latents_dtype)
            
            if(use_noise is not None):
                noise = combine_noise(noise, use_noise, use_noise_alpha)
            use_noise = noise.cpu().numpy()
            orig_latents = init_latents
            init_latents = sched.add_noise(init_latents, noise, timesteps)

            latents = init_latents
            
            timesteps = sched.timesteps[0:].to(self.device)
            if(True):
                it = list(enumerate(self.sd_pipeline.progress_bar(timesteps)))
            else:
                it = enumerate(timesteps)

            # denoising
            if(mode==0):
                mask_numpy = 1-(1-mask_numpy)**(1/steps)
                mask_tensor = torch.from_numpy(mask_numpy).to(device=DEFAULT_DEVICE)
            for i, t in it:
                noise_pred = self._predict_noise(latents, weights, cond_batches, nconds, noise_pred_batch_size, t)
                if(i==len(it)-1):
                    latents_proper = orig_latents
                else:
                    _, pt = it[i+1]
                    latents_proper = sched.add_noise(orig_latents, noise, torch.tensor([pt]))
                latents = sched.step(noise_pred, t, latents).prev_sample
                if(mode==0):
                    pass
                    # latents = latents*mask_tensor+latents_proper*(1-mask_tensor)
                else:
                    current = 1-i/steps
                    mask_current = (mask_numpy>=current).astype(np.float32) # * (mask_numpy**(1/steps))
                    mask_tensor=torch.from_numpy(mask_current).to(device=DEFAULT_DEVICE)
                latents = latents*mask_tensor+latents_proper*(1-mask_tensor)
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            del latents
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.sd_pipeline.numpy_to_pil(image)
            torch.cuda.empty_cache()
            return Reproducable(self.inpaint, image[0],
                prompt, orig_image, mask_image,
                steps=steps,
                cfg=cfg,
                neg_prompt=neg_prompt,
                use_noise=use_noise,
                use_noise_alpha=1
            )

            


    @torch.no_grad()
    def img2img(self, prompt, orig_image, cfg=7.5, steps=20, alpha=0.7, beta=1, neg_prompt="", noise_pred_batch_size=NOISE_PRED_BATCH, return_noise=False, use_noise=None, use_noise_alpha=1, print_progress=True, eta=0):
        """
        alpha: the rate of num_denoising_step/steps
        """
        loras = WeightedPrompt(prompt).loras
        loras = [(w, name) for name, w in loras.items()]
        lora_u, lora_t = self.wrap_lora(*loras)
        with torch.autocast("cuda"), locked(self.inference_lock), lora_u, lora_t:
            text_encoder = self.text_encoder
            unet = self.unet
            sched = self.sched
            vae = self.sd_pipeline.vae
            if alpha < 0 or alpha > 1:
                raise ValueError(f"The value of alpha should in [0.0, 1.0] but is {alpha}")
            
            
            # prepare orig image
            orig_image = preprocess_image(orig_image)
            orig_tensor = image_as_tensor(orig_image)
            # orig_tensor = preprocess(orig_image)


            # prepare weights
            weights, ids = self.get_ids(prompt, neg_prompt = neg_prompt)
            sqrt_weightn = len(weights)**0.5
            weights = normalize_weights(weights, std=cfg/sqrt_weightn)
            self.debug_ids(weights, ids)
            cond_num = len(weights)
            conds = []
            for i in ids:
                cond = text_encoder(torch.tensor(i).to(device=DEFAULT_DEVICE))[0]
                conds.append(cond)
            cond_batches = dict()
            for i in range(0, cond_num, noise_pred_batch_size):
                bs = min(noise_pred_batch_size, cond_num-i)
                cond = torch.cat(conds[i:i+bs])
                cond_batches[i] = cond
            
            # prepare latents
            latents_dtype = cond.dtype
            orig_tensor = orig_tensor.to(device=self.device, dtype=latents_dtype)
            init_latent_dist = vae.encode(orig_tensor).latent_dist  # vae distribution, vae mean and stddev...
            init_latents = init_latent_dist.sample(generator=None) # vae sample by stddev
            init_latents = 0.18215 * init_latents
            
            # prepare timestep
            sched.set_timesteps(steps)
            offset = sched.config.get("steps_offset", 0)
            init_timestep = int(steps * alpha) + offset
            init_timestep = min(init_timestep, steps)
            timesteps = sched.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps], device=self.device)
            if (True):
                init_numpy = init_latents.cpu().numpy()
                n = 0
                iter_count = 10
                for i in range(iter_count):
                    _noise = np.random.normal(size=init_numpy.shape)
                    if ((_noise*init_numpy).sum()<0):
                        n -= _noise
                    else:
                        n += _noise
                n /= sqrt(iter_count)
                print("std/mean", np.std(n), np.mean(n))
                noise = torch.from_numpy(n).to(device=self.device, dtype=latents_dtype)
            else:
                noise = torch.randn(init_latents.shape, generator=None, device=self.device, dtype=latents_dtype)
            
            if(use_noise is not None):
                use_noise = torch.tensor(use_noise).to(device=DEFAULT_DEVICE)
                noise = use_noise*use_noise_alpha+noise*(1-use_noise_alpha)
                sqrsum = use_noise_alpha**2+(1-use_noise_alpha)**2
                norm = sqrsum**0.5
                noise = noise/norm
            init_noise = noise.cpu().numpy()
            orig_latents = init_latents
            
            init_latents = sched.add_noise(init_latents, noise, timesteps)
            if(eta):
                noise = torch.randn(init_latents.shape, generator=None, device=self.device, dtype=latents_dtype)
                sqrsum = eta**2 + (1-eta)**2
                norm = sqrsum**0.5
                init_latents = init_latents*(1-eta) + noise*eta
                init_latents/=norm
            # denoising
            extra_step_kwargs={}
            latents = init_latents
            t_start = max(steps - init_timestep + offset, 0)
            timesteps = sched.timesteps[t_start:].to(self.device)
            rate_latent = beta
            rate_latent = beta**(1/steps/alpha)
            if(print_progress):
                it = list(enumerate(self.sd_pipeline.progress_bar(timesteps)))
            else:
                it = list(enumerate(timesteps))
            low_freq_fix = True
            hi_freq_fix = False
            for i, t in it:
                noise_pred = self._predict_noise(latents, weights, cond_batches, cond_num, noise_pred_batch_size, t)
                latents: torch.Tensor = sched.step(noise_pred, t, latents).prev_sample
                if(beta!=1):
                    if(i+1<len(it)):
                        i1, t1 = it[i+1]
                        latents_proper = sched.add_noise(orig_latents, noise, torch.tensor([t1]))
                    else:
                        latents_proper = orig_latents
                    if (low_freq_fix):
                        now_mean = latents.mean(axis=-1, keepdim=True).mean(axis=-2, keepdim=True)
                        tar_mean = latents_proper.mean(axis=-1, keepdim=True).mean(axis=-2, keepdim=True)
                        if (True):
                            latents_proper = latents_proper-tar_mean+now_mean
                        else:
                            latents = latents+(tar_mean-now_mean)*(1-rate_latent)
                    if (hi_freq_fix):
                        now_std = latents.std(axis=-1, keepdim=True).std(axis=-2, keepdim=True)
                        tar_std = latents_proper.std(axis=-1, keepdim=True).std(axis=-2, keepdim=True)
                        tar_mean = latents_proper.mean(axis=-1, keepdim=True).mean(axis=-2, keepdim=True)
                        if (True):
                            latents_proper = (latents_proper-tar_mean)/tar_std*now_std + tar_mean
                        
                    latents = (latents*rate_latent)+(latents_proper*(1-rate_latent))
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.sd_pipeline.numpy_to_pil(image)
            torch.cuda.empty_cache()
            return Reproducable(self.img2img, image[0], orig_image=orig_image, cfg=cfg, steps=steps, alpha=alpha, beta=beta, neg_prompt=neg_prompt, noise_pred_batch_size=noise_pred_batch_size, use_noise=init_noise, use_noise_alpha=1)

    def _predict_noise(self, latents, weights, cond_batches, cond_num, noise_pred_batch_size, t):
        unet = self.unet
        noises = []
        for idx in range(0, cond_num, noise_pred_batch_size):
            bs = min(noise_pred_batch_size, cond_num-idx)
            lat = torch.cat([latents]*bs)
            cond = cond_batches[idx]
            noise_pred = unet(lat, t, encoder_hidden_states=cond).sample
            for jdx in range(bs):
                noises.append((weights[idx+jdx], noise_pred[jdx]))
        noise_pred = None
        for w, noise in noises:
            if(noise_pred is None):
                noise_pred = noise*w
            else:
                noise_pred = noise_pred+noise*w
        return noise_pred
    def get_txt2img_interpolation_ids(self, prompt0, prompt1):
        SEP = ","
        wp0 = WeightedPrompt(prompt0).as_dict()
        wp1 = WeightedPrompt(prompt1).as_dict()
        keys = set(wp0.keys)+set(wp1.keys)
        for k in keys:
            w0 = k if k in wp0 else min(k, 0)
            w1 = k if k in wp1 else min(k, 0)
            s0 = wp0.get(k, "").split(SEP)
            s1 = wp1.get(k, "").split(SEP)
            lcs = LCS(s0, s1)
            
        

    def get_txt2img_interpolation_multiplier(self, prompt0, prompt1, cfg=7.5, steps=15, nframes=10, width=512, height=512, noise_pred_batch_size=NOISE_PRED_BATCH, use_noise=None, use_noise_alpha=1):
        weights0, ids0 = self.get_ids(prompt0)
        weights1, ids1 = self.get_ids(prompt1)
        nweights = len(weights0)+len(weights1)
        nfr = nframes if isinstance(nframes, int) else len(nframes)
        return nweights*width*height*steps*nfr
    @torch.no_grad()
    def txt2img_interpolation(self, prompt0, prompt1, cfg=7.5, steps=15, nframes=10, width=512, height=512, noise_pred_batch_size=NOISE_PRED_BATCH, return_noise = False, use_noise=None, use_noise_alpha=1):
        with torch.autocast("cuda"), locked(self.inference_lock):
            text_encoder = self.text_encoder
            unet = self.unet
            sched = self.sched
            vae = self.sd_pipeline.vae

            weights0, ids0 = self.get_ids(prompt0)
            weights1, ids1 = self.get_ids(prompt1)
            weights0 = normalize_weights(weights0, std=cfg/sqrt(len(weights0)))
            self.debug_ids(weights0, ids0)
            weights1 = normalize_weights(weights1, std=cfg/sqrt(len(weights1)))
            self.debug_ids(weights1, ids1)
            # cat_weights0 = np.concatenate([weights0, np.minimum(weights1, 0)], axis=0)
            cat_weights0 = np.concatenate([weights0, weights1*0], axis=0)
            # cat_weights1 = np.concatenate([np.minimum(weights0, 0), weights1], axis=0)
            cat_weights1 = np.concatenate([weights0*0, weights1], axis=0)
            len_embd = ids0[0].shape[-1]
            cat_ids = np.concatenate([ids0, ids1], axis=0).reshape((-1, len_embd))
            cat_ids = torch.tensor(cat_ids).to(device=DEFAULT_DEVICE)
            print("DEBUG:", cat_ids.shape)
            
            conds = text_encoder(cat_ids)[0]
            conds_batches = split_batch(conds, noise_pred_batch_size)

            latents_shape = (1, unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, generator=None, device=self.device, dtype=conds.dtype)
            if(use_noise is not None):
                latents = combine_noise(latents, use_noise, use_noise_alpha)
            init_noise = latents.cpu().numpy()
            
            if(True):
                print("DEBUG: using noise", init_noise.std(), init_noise.mean())
            # frame_latents = latents.repeat_interleave(nframes, dim=0)
            
            frame_weigths_batch = []
            if(isinstance(nframes, int)):
                for i in range(nframes):
                    rate = i/(nframes-1)
                    weight = cat_weights0*(1-rate)+cat_weights1*rate
                    weight = normalize_weights(weight, std=None)
                    frame_weigths = weight
                    frame_weigths_batch.append(split_batch(frame_weigths, noise_pred_batch_size))
            elif(isinstance(nframes, list)):
                for rate in nframes:
                    print("DEBUG: use rate", rate)
                    weight = cat_weights0*(1-rate)+cat_weights1*rate
                    weight = normalize_weights(weight, std=None)
                    print("DEBUG: weights", weight)
                    frame_weigths = weight
                    frame_weigths_batch.append(split_batch(frame_weigths, noise_pred_batch_size))
            frame_latents = [latents for i in frame_weigths_batch]
            print(frame_weigths_batch)
            sched.set_timesteps(steps)
            timesteps_tensor = sched.timesteps.to(self.device)
            for i, t in enumerate(self.sd_pipeline.progress_bar(timesteps_tensor)):
                for iframe, frame_latent in enumerate(frame_latents):
                    noise_preds = []
                    for bdx, cond in enumerate(conds_batches):
                        weights_batch = frame_weigths_batch[iframe][bdx]
                        lat = frame_latent.repeat_interleave(len(weights_batch), dim=0)
                        noise_pred_batch = unet(lat, t, encoder_hidden_states=cond).sample
                        for bdx, w in enumerate(weights_batch):
                            noise_pred = noise_pred_batch[bdx]
                            noise_preds.append((w, noise_pred))
                    ws = [i for i, j in noise_preds]
                    nps = [j for i, j in noise_preds]
                    noise_pred = combine(ws, nps)
                    latents = sched.step(noise_pred, t, frame_latent).prev_sample
                    frame_latents[iframe] = latents
            images = []
            for frame_latent in frame_latents:
                latent = 1/0.18215*frame_latent
                image = vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = self.sd_pipeline.numpy_to_pil(image)
                images.append(image[0])

            torch.cuda.empty_cache()
            return Reproducable(self.txt2img_interpolation, images, prompt0, prompt1, cfg=cfg, steps=steps, nframes=nframes, width=width, height=height, noise_pred_batch=NOISE_PRED_BATCH, use_noise=init_noise, use_noise_alpha=1)
        
    def get_multi_diffusion_multiplier(self, masks, prompts, steps=25, cfg=7.5, width=512, height=768, neg_prompt = ""):
        n_noises = 0
        for idx, p in enumerate(prompts):
            weight, ids = self.get_ids(p, neg_prompt=neg_prompt)
            n_noises+=len(ids)*len(masks)
        return n_noises*width*height*steps
        
    @torch.no_grad()
    def multi_diffusion(self, masks, prompts, steps=25, cfg=7.5, width=512, height=768, neg_prompt = "", use_noise=None, use_noise_alpha=1, noise_pred_batch_size=NOISE_PRED_BATCH):
        with torch.autocast("cuda"), locked(self.inference_lock):
            tencoder = self.text_encoder
            unet = self.unet
            vae = self.sd_pipeline.vae
            sched = self.sched

            conds = []
            _masks = []
            for idx, mask in enumerate(masks):
                mask = mask_as_numpy(mask.resize((width//8, height//8), Image.LANCZOS))
                prompt = prompts[idx]
                weights, idss = self.get_ids(prompt, neg_prompt=neg_prompt)
                for jdx, w in enumerate(weights):
                    ids = idss[jdx]
                    cond = tencoder(torch.tensor(ids).to(device=DEFAULT_DEVICE))[0]
                    conds.append(cond)
                    _masks.append(mask*w)
            masks_arr = np.stack(_masks, axis=-1)
            center: np.ndarray = masks_arr-masks_arr.mean(axis=-1, keepdims=True)
            # norm = center/(center.std(axis=-1, keepdims=True)+1e-6)*cfg
            norm = center/(center.std()+1e-6)*cfg
            
            masks_arr = norm+1/norm.shape[-1]
            print(masks_arr.shape)
            # assert np.all(masks_arr.sum(axis=-1,keepdims=True)==1), masks_arr.sum(axis=-1)
            masks = [torch.from_numpy(masks_arr[:, :,:,:,idx]).to(self.device) for idx in range(masks_arr.shape[-1])]
            # for mask in masks:
            #     print("weight", mask[0, 0, 0, 0])
            
            latents_shape = (1, unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, generator=None, device=self.device, dtype=cond.dtype)

            sched.set_timesteps(steps)
            timesteps_tensor = sched.timesteps.to(self.device)
            

            for i, t in enumerate(self.sd_pipeline.progress_bar(timesteps_tensor)):
                noises = []
                for idx, cond in enumerate(conds):
                    noise_pred = unet(latents, t, encoder_hidden_states=cond).sample
                    noises.append(noise_pred)
                noise = torch.zeros_like(latents)
                for idx, mask in enumerate(masks):
                    # print(mask.shape)
                    noise += noises[idx]*mask
                
                latents = sched.step(noise, t, latents).prev_sample

            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.sd_pipeline.numpy_to_pil(image)
            
            torch.cuda.empty_cache()
            return Reproducable(self.multi_diffusion, image[0], masks, prompts, steps=steps, cfg=cfg, width=width, height=height, neg_prompt=neg_prompt)

    @torch.no_grad()
    def txt2img(self, prompt, cfg=7.5, steps=20, width=512, height=768, neg_prompt = "", noise_pred_batch_size=NOISE_PRED_BATCH, return_noise=False, use_noise=None, use_noise_alpha=1):
        loras = WeightedPrompt(prompt).loras
        loras = [(w, name) for name, w in loras.items()]
        lora_u, lora_t = self.wrap_lora(*loras)
        with torch.autocast("cuda"), locked(self.inference_lock), lora_u, lora_t:

            text_encoder = self.text_encoder
            unet = self.unet
            sched = self.sched
            vae = self.sd_pipeline.vae
            weights, ids = self.get_ids(prompt, neg_prompt = neg_prompt)
            sqrt_weightn = len(weights)**0.5
            weights = normalize_weights(weights, std=cfg/sqrt_weightn)
            self.debug_ids(weights, ids)
            cond_num = len(weights)
            conds = []
            for i in ids:
                print("te device", text_encoder.device)
                cond = text_encoder(torch.tensor(i).to(device=DEFAULT_DEVICE))[0]
                conds.append(cond)
            for idx, w in enumerate(weights):
                cond = conds[idx]
                token_ids = ids[idx]
                # tokens = self.tokenizer.batch_decode(token_ids)
                
            latents_shape = (1, unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, generator=None, device=self.device, dtype=cond.dtype)
            if(use_noise is not None):
                latents = combine_noise(latents, use_noise, use_noise_alpha)
            init_noise = latents.cpu().numpy()
            sched.set_timesteps(steps)
            timesteps_tensor = sched.timesteps.to(self.device)
            
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * sched.init_noise_sigma
            cond_batches = dict()
            for i in range(0, cond_num, noise_pred_batch_size):
                bs = min(noise_pred_batch_size, cond_num-i)
                cond = torch.cat(conds[i:i+bs])
                cond_batches[i] = cond

            # accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            # if accepts_eta:
            # extra_step_kwargs["eta"] = eta


            for i, t in enumerate(self.sd_pipeline.progress_bar(timesteps_tensor)):
                noise_pred = self._predict_noise(latents, weights, cond_batches, cond_num, noise_pred_batch_size, t)
                
                latents = sched.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = self.sd_pipeline.numpy_to_pil(image)
            
            torch.cuda.empty_cache()
            debug_vram("after txt2img")
            return Reproducable(self.txt2img, image[0], prompt, cfg=cfg, steps=steps, width=width, height=height, neg_prompt=neg_prompt, noise_pred_batch_size=noise_pred_batch_size, use_noise=init_noise, use_noise_alpha=1)

    def debug_ids(self, weights, ids):
        stid, edid = self.raw_tokenize("").input_ids[0]
        _ids = []
        for _ in ids:
            no_pad = []
            for id in _[0]:
                # print(_, stid, edid)
                if(id!=stid and id != edid):
                    no_pad.append(id)
            _ids.append([no_pad])
        ids = np.array(_ids)
        for idx, w in enumerate(weights):
            prompt = self.tokenizer.batch_decode(ids[idx])
            print("len=", len(ids[idx][0]), "weight =", w, "prompt =", prompt)

    def get_ids_v1(self, prompt, neg_prompt = ""):
        self.auto_load_ti()
        tokenizer = self.tokenizer
        max_length = self.tokenizer.model_max_length
        
        stid, edid = self.raw_tokenize("").input_ids[0]
        buckets = []
        def add_ids(weight, *ids):
            nonlocal buckets
            for b in buckets:
                if(b.add_ids(weight, *ids)):
                    return b
            bucket = IDBucket(max_length, weight, stid, edid)
            assert (bucket.add_ids(weight, *ids))
            buckets.append(bucket)
            return
        def add_sentence(weight, sentence):
            aliases = list(self.ti_alias.items())
            for k, v in sorted(aliases, key=lambda x:-len(x[0])):
                if(k in sentence):
                    print("%s -> %s"%(k, v))
                    clear_sentences = sentence.split(k)
                    for idx, i in enumerate(clear_sentences):
                        if(idx):
                            add_sentence(weight, v)
                        add_sentence(weight, i)
                    return
            if(", " in sentence):
                for s in sentence.split(", "):
                    add_sentence(weight, s)
                return
            ids = self.raw_tokenize(sentence).input_ids[0][1:-1] # 1:-1 ignore startoftext, endoftext
            if(len(ids)<=max_length-2):
                add_ids(weight, *ids)
            else:
                for id in ids:
                    add_ids(weight, id)
        for w, s in WeightedPrompt(prompt + "/*"+neg_prompt+"*/"):
            add_sentence(w, s)
        for b in buckets:
            b.pad_wrap(min((max_length-2)/3, b.used*2))
        # buckets = [b for b in buckets if b.used]
        # f_used = lambda u:(log(u)-log(max_length))/log(max_length)+2
        f_used = lambda u:u**0.5
        weights = np.array([b.weight*f_used(b.used) for b in buckets])
        # fewer token guidance may lead to worse quality and too strong guidance
        ids = [b.ids for b in buckets]
        return weights, ids
    def get_ids_v2(self, prompt, neg_prompt = ""):
        self.auto_load_ti()
        tokenizer = self.tokenizer
        max_length = self.tokenizer.model_max_length
        
        stid, edid = self.raw_tokenize("").input_ids[0]
        buckets = []
        self.auto_load_ti()
        tokenizer = self.tokenizer
        max_length = self.tokenizer.model_max_length
        
        stid, edid = self.raw_tokenize("").input_ids[0]
        buckets = []
        def add_ids(weight, *ids):
            nonlocal buckets
            for b in sorted(buckets, key=lambda x:x.used):
                
                if(b.add_ids(weight, *ids)):
                    return b
            bucket = IDBucket(max_length, weight, stid, edid)
            assert (bucket.add_ids(weight, *ids))
            buckets.append(bucket)
            return bucket
        def add_sentence(weight, sentence, spl = True):
            aliases = list(self.ti_alias.items())
            for k, v in sorted(aliases, key=lambda x:-len(x[0])):
                if(k in sentence):
                    print("%s -> %s"%(k, v))
                    clear_sentences = sentence.split(k)
                    for idx, i in enumerate(clear_sentences):
                        if(idx):
                            add_sentence(weight, v)
                        add_sentence(weight, i)
                    return
            
            if(spl and ", " in sentence):
                for idx, s in enumerate(sentence.split(", ")):
                    if(idx):
                        add_sentence(weight, ", "+s, spl=False)
                    else:
                        add_sentence(weight, s, spl=False)
                return
            
            ids = self.raw_tokenize(sentence).input_ids[0][1:-1] # 1:-1 ignore startoftext, endoftext
            if(len(ids)<=max_length-2):
                add_ids(weight, *ids)
            else:
                for id in ids:
                    add_ids(weight, id)
        if(isinstance(prompt, str)):
            wp = WeightedPrompt(prompt + "/*"+neg_prompt+"*/")
        else:
            wp = prompt
        for w, s in wp:
            add_sentence(w, s)
        
        nids = {}
        for b in buckets:
            nids[b.weight] = nids.get(b.weight, 0)+b.used
        buckets = []
        for w, nid in nids.items():
            n = max(ceil(nid/max_length*1.08), 1)
            for i in range(n):
                buckets.append(IDBucket(max_length, w, stid, edid))
        
        for w, s in wp:
            add_sentence(w, s)
        f_used = lambda u:u**0.1

        weights = np.array([b.weight*f_used(b.used) for b in buckets])
        # fewer token guidance may lead to worse quality and too strong guidance
        ids = [b.ids for b in buckets]
        return weights, ids
    def get_ids(self, prompt, neg_prompt = ""):
        self.auto_load_ti()
        tokenizer = self.tokenizer
        max_length = self.tokenizer.model_max_length
        
        stid, edid = self.raw_tokenize("").input_ids[0]
        buckets = []
        self.auto_load_ti()
        tokenizer = self.tokenizer
        max_length = self.tokenizer.model_max_length
        
        stid, edid = self.raw_tokenize("").input_ids[0]
        buckets = []
        def add_ids(weight, *ids):
            nonlocal buckets
            for b in sorted(buckets, key=lambda x:x.used):
                
                if(b.add_ids(weight, *ids)):
                    return b
            bucket = IDBucket(max_length, weight, stid, edid)
            assert (bucket.add_ids(weight, *ids))
            buckets.append(bucket)
            return bucket
        def add_sentence(weight, sentence, spl = True):
            aliases = list(self.ti_alias.items())
            for k, v in sorted(aliases, key=lambda x:-len(x[0])):
                if(callable(v)):
                    v = v()
                if(k in sentence):
                    print("%s -> %s"%(k, v))
                    clear_sentences = sentence.split(k)
                    for idx, i in enumerate(clear_sentences):
                        if(idx):
                            add_sentence(weight, v)
                        add_sentence(weight, i)
                    return
            sep = ","
            if(spl and sep in sentence):
                for idx, s in enumerate(sentence.split(sep)):
                    if(idx):
                        add_sentence(weight, sep+s, spl=False)
                    else:
                        add_sentence(weight, s, spl=False)
                return
            
            ids = self.raw_tokenize(sentence).input_ids[0][1:-1] # 1:-1 ignore startoftext, endoftext
            if(len(ids)<=max_length-2):
                add_ids(weight, *ids)
            else:
                print("TOO LONG", sentence)
                for id in ids:
                    add_ids(weight, id)
        if(isinstance(prompt, str)):
            wp = WeightedPrompt(prompt + "/*"+neg_prompt+"*/")
        else:
            wp = prompt
        extra_bucket = True
        while(extra_bucket):
            LIMIT = 30
            extra_bucket = False
        
            for idx, b in enumerate(buckets):
                b.reinit()
            for w, s in wp:
                n = len(buckets)
                add_sentence(w, s)
                m = len(buckets)
                extra_bucket = extra_bucket or (n!=m)
            if(len(buckets)>=LIMIT):
                f_used = lambda u:u**0.1
                weights = np.array([b.weight*f_used(b.used) for b in buckets])
                ids = [b.ids for b in buckets]
                self.debug_ids(weights, ids)
                assert len(buckets) < LIMIT
        f_used = lambda u:u**0.1

        weights = np.array([b.weight*f_used(b.used) for b in buckets])
        # fewer token guidance may lead to worse quality and too strong guidance
        ids = [b.ids for b in buckets]
        return weights, ids

            
if(__name__=="__main__"):
    from modules.diffusion.model_mgr import pipe as diffusion_pipe, models, get_model
    mask0 = Image.open(r"E:\Pics\gradient.png").convert("L")
    mask1 = Image.fromarray(255-np.array(mask0))
    prompt0 = "2girls, pink hair, cat ears"
    prompt1 = "2girls, blue hair, cat ears"
    nprompt = "worst quality"
    model = get_model()
    im = model.multi_diffusion([mask0, mask1], [prompt0, prompt1], width=512, height=512, neg_prompt=nprompt)
    im.show()

