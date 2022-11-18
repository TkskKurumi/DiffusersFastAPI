import traceback
from enum import Enum
from types import NoneType
from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import Response, JSONResponse
from uuid import uuid4 as rnd_id
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from modules.utils.candy import locked
from modules.utils.timer import Timer, get_eta
from diffusers import StableDiffusionPipeline
from modules.utils.myhash import hashi
from modules.utils import pil2jpegbytes, normalize_resolution
from modules.utils.misc import default_neg_prompt, upfile2img, DEFAULT_RESOLUTION
from modules.diffusion.default_pipe import pipe as diffusion_pipe
from modules.superres import upscale
from PIL import Image
app = FastAPI()

pool = ThreadPoolExecutor()

diffusion_infer_lock = Lock()

images = {}


def add_img(img):
    if(len(images)>256):
        k = next(iter(images))
        images.pop(k)
    hashed = hex(hashi(img))[2:]
    images[hashed] = img
    return hashed


class TicketStatus(Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    RUNNING = "running"
    FINISHED = "finished"


class Ticket:
    tickets = {}

    def __init__(self, purpose, token=None):
        self.purpose = purpose
        self.token = token
        self.id = str(rnd_id())
        self.status = TicketStatus.CREATED
        Ticket.tickets[self.id] = self

    def run(self):
        raise NotImplementedError()

    def param(self, **kwargs):
        raise NotImplementedError()

    def eta(self):
        raise NotImplementedError()
    
    @property
    def result(self):
        if (self.status == TicketStatus.CREATED):
            eta = self.submit()["data"]["eta"]
        return self._task.result()
    def submit(self):
        self.status = TicketStatus.SUBMITTED
        eta = self.eta()
        print("Running %s, %.1f seconds estimated"%(self, eta))
        self._task = pool.submit(self.run)
        data = {}
        data["eta"] = eta
        if(hasattr(self, "eta_this") and callable(self.eta_this)):
            eta_this = self.eta_this()
            data["eta_this"] = eta_this
        return {
            "status":0, "message":"ok", "data": data
        }


class SRTicket(Ticket):
    tickets = {}

    def __init__(self, token=None):
        super().__init__("super-resolution", token)
        SRTicket.tickets[self.id] = self
        raise NotImplementedError()

class SeqIMG2IMGTicket(Ticket):
    tickets = {}
    STEP = 20
    LOCK = diffusion_infer_lock
    TIMER_KEY = "diffusion_infer_resolution_steps"
    def __init__(self, token=None):
        super().__init__("img2img_seq", token)
        IMG2IMGTicket.tickets[self.id] = self
        self.params = {}
    @property
    def accepted_params(self):
        return ["prompt", "neg_prompt", "guidance", "alpha"]
    def param(self, **kwargs):
        for key in self.accepted_params:
            if (key in kwargs):
                self.params[key] = kwargs[key]
        if ("neg_prompt" not in self.params):
            self.params["neg_prompt"] = default_neg_prompt()
    def __str__(self):
        ret = []
        for key in self.accepted_params:
            if(key in self.params):
                v = str(self.params[key])
                if(len(v)>50):
                    v = v[:50]+"..."
                ret.append("%s=%s"%(key, v))
        return "<img2img_seq %s>"%(", ".join(ret))
    def auth(self):
        return True
    
    def eta_this(self):
        return get_eta(IMG2IMGTicket.TIMER_KEY, self.get_n(), True)

    def eta(self):
        return get_eta(IMG2IMGTicket.TIMER_KEY, self.get_n())
    def form_pipe_kwargs(self):
        pro = self.params["prompt"]
        neg_pro = self.params["neg_prompt"]
        orig_image =self.params["orig_image"]
        orig_image = orig_image.resize(normalize_resolution(*orig_image.size), Image.Resampling.LANCZOS)
        alpha = max(self.params.get("alpha", 0.68), 0.01)
        guidance = self.params.get("guidance") or 7.5/alpha
        eta = self.params.get("ddim_noise", 0) # eta for ddim
        steps = int(max(25/alpha, IMG2IMGTicket.STEP))
        make_ret = lambda *args, **kwargs:(args, kwargs)
        return make_ret(pro, orig_image, alpha=alpha, steps=steps, neg_prompt=neg_pro, cfg=guidance, eta=eta)
    def get_n(self):
        args, kwargs = self.form_pipe_kwargs()
        return diffusion_pipe.get_img2img_multiplier(*args, **kwargs)
    def run(self):
        t = Timer(IMG2IMGTicket.TIMER_KEY, self.get_n())
        try:
            with locked(IMG2IMGTicket.LOCK):
                with t:
                    self.status = TicketStatus.RUNNING
                    args, kwargs = self.form_pipe_kwargs()
                    img = diffusion_pipe.img2img(*args, **kwargs)[0]
                    img = upscale(img)
                    self._result = {
                        "status": 0,
                        "message": "ok",
                        "data": {
                            "image": add_img(img),
                            "type": "image"
                        }
                    }
                    return self._result
        except Exception as e:
            traceback.print_exc()
            self._result = {"status": -500, "message": "failed",
                            "reason": str(e)}
            
            return self._result
class IMG2IMGTicket(Ticket):
    tickets = {}
    STEP = 25
    LOCK = diffusion_infer_lock
    TIMER_KEY = "diffusion_infer_resolution_steps"
    def __init__(self, token=None, strength=0.75):
        super().__init__("img2img", token)
        IMG2IMGTicket.tickets[self.id] = self
        self.params = {}
    @property
    def accepted_params(self):
        return ["prompt", "neg_prompt", "guidance", "alpha", "ddim_noise"]
    def param(self, **kwargs):
        for key in self.accepted_params:
            if (key in kwargs):
                self.params[key] = kwargs[key]
        if ("neg_prompt" not in self.params):
            self.params["neg_prompt"] = default_neg_prompt()
    def __str__(self):
        ret = []
        for key in self.accepted_params:
            if(key == "orig_image"):continue
            if(key in self.params):
                
                v = str(self.params[key])
                if(len(v)>50):
                    v = v[:50]+"..."
                ret.append("%s=%s"%(key, v))
        return "<img2img %s>"%(", ".join(ret))
    def auth(self):
        return True
    
    def eta_this(self):
        return get_eta(IMG2IMGTicket.TIMER_KEY, self.get_n(), True)

    def eta(self):
        return get_eta(IMG2IMGTicket.TIMER_KEY, self.get_n())
    def form_pipe_kwargs(self):
        pro = self.params["prompt"]
        neg_pro = self.params["neg_prompt"]
        orig_image =self.params["orig_image"]
        orig_image = orig_image.resize(normalize_resolution(*orig_image.size), Image.Resampling.LANCZOS)
        alpha = max(self.params.get("alpha", 0.68), 0.01)
        guidance = self.params.get("guidance") or 7.5/alpha
        eta = self.params.get("ddim_noise", 0) # eta for ddim
        steps = int(max(25/alpha, IMG2IMGTicket.STEP))
        make_ret = lambda *args, **kwargs:(args, kwargs)
        return make_ret(pro, orig_image, alpha=alpha, steps=steps, neg_prompt=neg_pro, cfg=guidance, eta=eta)
    def get_n(self):
        args, kwargs = self.form_pipe_kwargs()
        return diffusion_pipe.get_img2img_multiplier(*args, **kwargs)
    def run(self):
        t = Timer(IMG2IMGTicket.TIMER_KEY, self.get_n())
        try:
            with locked(IMG2IMGTicket.LOCK):
                with t:
                    self.status = TicketStatus.RUNNING
                    args, kwargs = self.form_pipe_kwargs()
                    img = diffusion_pipe.img2img(*args, **kwargs)[0]
                    img = upscale(img)
                    self._result = {
                        "status": 0,
                        "message": "ok",
                        "data": {
                            "image": add_img(img),
                            "type": "image"
                        }
                    }
                    return self._result
        except Exception as e:
            traceback.print_exc()
            self._result = {"status": -500, "message": "failed",
                            "reason": str(e)}
            
            return self._result
        return self._result
class TXT2IMGPromptInterp(Ticket):
    tickets = {}
    TIMER_KEY = "diffusion_infer_resolution_steps"
    LOCK = diffusion_infer_lock
    
    RESOLUTION = DEFAULT_RESOLUTION/1.8
    def __init__(self, token=None):
        super().__init__("txt2img_interp")
        TXT2IMGPromptInterp.tickets[self.id]=self
        self.params = {}
    @property
    def accepted_params(self):
        return ["prompt", "prompt1", "guidance", "aspect", "nframes"]
    def param(self, **kwargs):
        for key in self.accepted_params:
            if (key in kwargs):
                self.params[key] = kwargs[key]
    
    def eta_this(self):
        return get_eta(TXT2IMGPromptInterp.TIMER_KEY, self.get_n(), True)
        
    def eta(self):
        return get_eta(TXT2IMGPromptInterp.TIMER_KEY, self.get_n())
    def auth(self):
        return True
    def get_n(self):
        args, kwargs = self.form_pipe_kwargs()
        return diffusion_pipe.get_txt2img_interpolation_multiplier(*args, **kwargs)
    def form_pipe_kwargs(self):
        pro = self.params["prompt"]
        pro1 = self.params["prompt1"]
        cfg = self.params.get("guidance", 12)
        aspect = self.params.get("aspect", 9/16)
        nfr = self.params.get("nframes", 8)
        nfr = min(max(nfr, 3), 12)
        steps = int(100/nfr)
        
        w, h = normalize_resolution(aspect, 1, TXT2IMGPromptInterp.RESOLUTION)
        
        make_ret = lambda *args, **kwargs:(args, kwargs)
        
        return make_ret(pro, pro1, cfg=cfg, width=w, height=h, nframes=nfr, steps=steps)
    def run(self):
        t = Timer(TXT2IMGPromptInterp.TIMER_KEY, self.get_n())
        try:
            with locked(TXT2IMGPromptInterp.LOCK):
                with t:
                    self.status = TicketStatus.RUNNING
                    args, kwargs = self.form_pipe_kwargs()
                    imgs = diffusion_pipe.txt2img_interpolation(*args, **kwargs)
                    imgs = [upscale(img) for img in imgs]
                    self._result = {
                        "status": 0,
                        "message": "ok",
                        "data": {
                            "images": imgs,
                            "type": "image_sequence"
                        }
                    }
                    return self._result
        except Exception as e:
            traceback.print_exc()
            self._result = {"status": -500, "message": "failed",
                            "reason": str(e)}
            
            return self._result
        return self._result

class TXT2IMGTicket(Ticket):
    tickets = {}
    TIMER_KEY = "diffusion_infer_resolution_steps"
    LOCK = diffusion_infer_lock
    STEPS = 30

    def __init__(self, token=None):
        super().__init__("txt2img", token)
        TXT2IMGTicket.tickets[self.id] = self
        self.params = {}
    @property
    def accepted_params(self):
        return ["prompt", "neg_prompt", "guidance", "aspect"]
    def param(self, **kwargs):
        for key in self.accepted_params:
            if (key in kwargs):
                self.params[key] = kwargs[key]
        if ("neg_prompt" not in self.params):
            self.params["neg_prompt"] = default_neg_prompt()

    def auth(self):
        return True

    def eta_this(self):
        return get_eta(TXT2IMGTicket.TIMER_KEY, self.get_n(), True)

    def eta(self):
        
        return get_eta(TXT2IMGTicket.TIMER_KEY, self.get_n())

    def get_n(self):
        args, kwargs = self.form_pipe_kwargs()
        return diffusion_pipe.get_txt2img_multiplier(*args, **kwargs)

    def __str__(self):
        ret = []
        for key in self.accepted_params:
            if(key in self.params):
                v = str(self.params[key])
                if(len(v)>50):
                    v = v[:50]+"..."
                ret.append("%s=%s"%(key, v))
        return "<txt2img %s>"%(", ".join(ret))
    def form_pipe_kwargs(self):
        pro = self.params.get("prompt", "1girl")
        cfg = self.params.get("guidance", 12)
        neg = self.params.get("neg_prompt", "bad anatomy, bad perspective, bad proportion")
        wid, hei = normalize_resolution(self.params.get("aspect", 9/16), 1)
        make_ret = lambda *args, **kwargs:(args, kwargs)
        return make_ret(pro, neg_prompt = neg, cfg=cfg, width=wid, height=hei, steps=TXT2IMGTicket.STEPS)
    def run(self):
        t = Timer(TXT2IMGTicket.TIMER_KEY, self.get_n())
        try:
            with locked(TXT2IMGTicket.LOCK):
                with t:
                    self.status = TicketStatus.RUNNING
                    args, kwargs = self.form_pipe_kwargs()
                    img = diffusion_pipe.txt2img(*args, **kwargs)[0]
                    img = upscale(img)
                    self._result = {
                        "status": 0,
                        "message": "ok",
                        "data": {
                            "image": add_img(img),
                            "type": "image"
                        }
                    }
                    return self._result
        except Exception as e:
            traceback.print_exc()
            self._result = {"status": -500, "message": "failed",
                            "reason": str(e)}
            
            return self._result
        return self._result

class InpaintTicket(Ticket):
    tickets = {}
    STEP = 30
    LOCK = diffusion_infer_lock
    TIMER_KEY = "diffusion_infer_resolution_steps"
    def __init__(self, token=None, strength=0.75):
        super().__init__("inpaint", token)
        InpaintTicket.tickets[self.id] = self
        self.params = {}
    @property
    def accepted_params(self):
        return ["prompt", "neg_prompt", "guidance", "alpha", "mode"]
    def param(self, **kwargs):
        for key in self.accepted_params:
            if (key in kwargs):
                self.params[key] = kwargs[key]
        if ("neg_prompt" not in self.params):
            self.params["neg_prompt"] = default_neg_prompt()
    def __str__(self):
        ret = []
        for key in self.accepted_params:
            if(key == "orig_image"):continue
            if(key in self.params):
                
                v = str(self.params[key])
                if(len(v)>50):
                    v = v[:50]+"..."
                ret.append("%s=%s"%(key, v))
        return "<img2img %s>"%(", ".join(ret))
    def auth(self):
        return True
    def eta_this(self):
        return get_eta(InpaintTicket.TIMER_KEY, self.get_n(), True)
    def eta(self):
        return get_eta(InpaintTicket.TIMER_KEY, self.get_n())
    def form_pipe_kwargs(self):
        pro = self.params["prompt"]
        neg_pro = self.params["neg_prompt"]
        orig_image =self.params["orig_image"]
        orig_image = orig_image.resize(normalize_resolution(*orig_image.size), Image.Resampling.LANCZOS)
        mask_image = self.params["mask_image"]
        guidance = self.params.get("guidance", 12)
        mode = self.params.get("mode", 0)
        steps = InpaintTicket.STEP
        make_ret = lambda *args, **kwargs:(args, kwargs)
        return make_ret(pro, orig_image, mask_image, steps=steps, neg_prompt=neg_pro, cfg=guidance, mode=mode)
    def get_n(self):
        args, kwargs = self.form_pipe_kwargs()
        return diffusion_pipe.get_inpaint_multiplier(*args, **kwargs)
    def run(self):
        t = Timer(InpaintTicket.TIMER_KEY, self.get_n())
        try:
            with locked(InpaintTicket.LOCK):
                with t:
                    self.status = TicketStatus.RUNNING
                    args, kwargs = self.form_pipe_kwargs()
                    repro = diffusion_pipe.inpaint(*args, **kwargs)
                    img = repro.result
                    img = upscale(img)
                    self._result = {
                        "status": 0,
                        "message": "ok",
                        "data": {
                            "image": add_img(img),
                            "type": "image"
                        }
                    }
                    return self._result
        except Exception as e:
            traceback.print_exc()
            self._result = {"status": -500, "message": "failed",
                            "reason": str(e)}
            
            return self._result

class TicketParam(BaseModel):
    prompt: str
    guidance: float | NoneType = None
    strength: float | NoneType = None
    neg_prompt: str | NoneType = None
    aspect: float | NoneType = None
    alpha: float | NoneType = None
    prompt1: str | NoneType = None
    nframes: int | NoneType = 8
    mode: int = 0
    ddim_noise : float = 0
def serialize_response(resp):
    if(isinstance(resp, list)):
        return [serialize_response(i) for i in resp]
    elif(isinstance(resp, dict)):
        return {k: serialize_response(v) for k, v in resp.items()}
    elif(isinstance(resp, Image.Image)):
        return add_img(resp)
    elif(isinstance(resp, str)):
        return resp
    elif(isinstance(resp, int) or isinstance(resp, float)):
        return resp
    else:
        print("unhandled", type(resp))
        return resp
        

@app.get("/images/{id}")
def get_image(id: str):
    if(id in images):
        return Response(pil2jpegbytes(images[id]), media_type='image/jpeg')
    else:
        return Response("not found", status_code=404)

@app.post("/ticket/{ticket_id}/upload_image")
def post_upload(ticket_id: str, fn: str="orig_image", data: UploadFile = File()):
    if (ticket_id in Ticket.tickets):
        ticket = Ticket.tickets[ticket_id]
        img = upfile2img(data)
        ticket.params[fn] = img
        ret ={"status":0, "message": "ok"}
    else:
        ret = {"status": -404, "message": "ticket doesn't exist"}
        return JSONResponse(ret, status_code=404)
@app.get("/ticket/{ticket_id}/result")
def get_ticket_result(ticket_id: str):
    if (ticket_id in Ticket.tickets):
        ticket = Ticket.tickets[ticket_id]
        result = ticket.result
        print(str(result))
        stat_code = -result["status"] if result["status"] < 0 else 200
        result = serialize_response(result)
        return JSONResponse(result, status_code=stat_code)
    else:
        ret = {"status": -404, "message": "ticket doesn't exist"}
        return JSONResponse(ret, status_code=404)
@app.get("/ticket/{ticket_id}/submit")
def get_ticket_result(ticket_id: str):
    if (ticket_id in Ticket.tickets):
        ticket = Ticket.tickets[ticket_id]
        result = ticket.submit()
        stat_code = -result["status"] if result["status"] < 0 else 200
        return JSONResponse(result, status_code=stat_code)
    else:
        ret = {"status": -404, "message": "ticket doesn't exist"}
        return JSONResponse(ret, status_code=404)

@app.post("/ticket/{ticket_id}/param")
def post_ticket_param(ticket_id: str, data: TicketParam):
    if (ticket_id in Ticket.tickets):
        ticket = Ticket.tickets[ticket_id]
        d = data.dict()
        params = {}
        for key in ticket.accepted_params:
            if(d.get(key) is not None):
                params[key] = d[key]
        ticket.param(**params)
        ret = dict()
        for k, v in ticket.params.items():
            if(isinstance(v, Image.Image)):
                ret[k] = add_img(v)
            else:
                ret[k] = v
        return JSONResponse({"status":0, "message":"ok", "data":ret})
    else:
        ret = {"status": -1, "message": "ticket doesn't exist"}
        return JSONResponse(ret, status_code=404)


@app.get("/ticket/create/{purpose}")
def post_ticket_create(purpose: str):
    ret = {}
    if (purpose == "txt2img"):
        t = TXT2IMGTicket()
    elif(purpose == "img2img"):
        t = IMG2IMGTicket()
    elif(purpose == "txt2img_interp"):
        t = TXT2IMGPromptInterp()
    elif(purpose == "inpaint"):
        t = InpaintTicket()
    else:
        ret["status"] = -1
        ret["message"] = "unknown purpose"
        return JSONResponse(ret, status_code=404)
    
    if(not t.auth()):
        ret["status"] = -1
        ret["message"] = "auth fail"
        ret["data"] = None
        return JSONResponse(ret)
    
    ret["status"] = 0
    ret["message"] = "ok"
    ret["data"] = {"id": t.id}
    return JSONResponse(ret)




if (__name__ == "__main__"):
    t1 = TXT2IMGTicket(1)
    t1.param(prompt="1girl, pink hair, blue eyes, serafuku")
    result = t1.run()
    print(result)
