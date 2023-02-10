import argparse
import os
import re
from modules.diffusion import model_mgr
from modules.diffusion.wrapped_pipeline import CustomPipeline
from modules.utils.candy import print_time
import torch
import shutil
import tqdm
from torch import nn
import numpy as np
from functools import partial


# this function can also interpolate more than 2 models
def interpolate_into(into: nn.Module, intersted_weights, *args):
    with print_time("interpolate module"):
        orig_state = into.state_dict()
        result_state = {}
        for k in orig_state:
            if (k in intersted_weights):
                sum_w = 0
                sum_weight = 0
                for w, state in args:
                    if (w):
                        sum_w += w
                        sum_weight += w*state[k]
                result_state[k] = sum_weight/sum_w
            else:
                result_state[k] = orig_state[k]
        into.load_state_dict(result_state)


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str,
                        required=True, help="specify model path", )
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--base-vae", type=str, default="")
    parser.add_argument("--target-vae", type=str, default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match-start", type=str,
                       default="", help="match weight starts with")
    group.add_argument("--match-regex", type=str, default="",
                       help="match weight regular expression")
    parser.add_argument("--list-weights", action="store_true")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=998244353)
    parser.add_argument("--interp_steps", type=int, default=15,
                        help="num of interpolation steps")
    parser.add_argument("--infer_steps", type=int, default=25,
                        help="num of inference steps")
    
    parser.add_argument("--output", type=str, default="model_interpolate")
    args = parser.parse_args()

    output = args.output
    if (os.path.exists(output)):
        inp = input("%s already exist, continue? (y/n)" % output)
        while (True):
            if (inp.startswith("y")):
                break
            elif (inp.startswith("n")):
                exit()
            else:
                inp = input("%s already exist, continue? (y/n)" % output)
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    base_model_id = args.base_model
    target_model_id = args.target_model

    base_model = model_mgr.load_model("base", base_model_id, args.base_vae)
    base_unet_state, base_vae_state = base_model.unet.state_dict(), base_model.vae.state_dict()
    target_model = model_mgr.load_model("target", target_model_id, args.target_vae)
    target_unet_state, target_vae_state = target_model.unet.state_dict(
    ), target_model.vae.state_dict()

    model_mgr.MASTER_MODEL = CustomPipeline(model_mgr._INFER_MODEL)

    orig_unet_state_dict = base_model.unet.state_dict()
    orig_vae_state_dict = base_model.vae.state_dict()

    with open("model_interpolate.log", "w") as f:

        builtin_print = print
        print_log = partial(print, file=f)
        print = lambda *args, **kwargs: (builtin_print(*
                                         args, **kwargs), print_log(*args, **kwargs))

        if (args.list_weights):
            print("weight listing is too verbose, see ./model_interpolate.log")
            print_log("unet weights:", len(orig_unet_state_dict))
            for k in orig_unet_state_dict:
                print_log("    "+k)
            print_log("vae weights:", len(orig_vae_state_dict))
            for k in orig_vae_state_dict:
                print_log("    "+k)
        interested_weights = set()
        for k in list(orig_unet_state_dict)+list(orig_vae_state_dict):
            if (args.match_start):
                if (k.startswith(args.match_start)):
                    interested_weights.add(k)
            elif (args.match_regex):
                if (re.match(args.match_regex, k)):
                    interested_weights.add(k)
        assert interested_weights, "No params are selected!"
        print("intersted weights:")
        for i in interested_weights:
            print("    "+i)
        if (args.dry):
            exit()

        torch.manual_seed(args.seed)
        wid, hei = args.width, args.height
        wid, hei = wid-wid % 8, hei-hei % 8
        noise = torch.randn((4, hei//8, wid//8)).cpu().numpy()
        for i in tqdm.tqdm(range(args.interp_steps)):
            rate = i/(args.interp_steps-1)
            interpolate_into(model_mgr.MASTER_MODEL.sd_pipeline.unet,
                             interested_weights,
                             (1-rate, base_unet_state),
                             (rate, target_unet_state)
                             )
            interpolate_into(model_mgr.MASTER_MODEL.sd_pipeline.vae,
                             interested_weights,
                             (1-rate, base_vae_state),
                             (rate, target_vae_state)
                             )
            rep = model_mgr.MASTER_MODEL.txt2img(
                args.prompt, width=wid, height=hei, use_noise=noise, cfg=12, steps=args.infer_steps)
            image = rep.result
            image.save(os.path.join(output, "%03d.png" % i))
        target_filesize = 3<<20
        rate = (target_filesize/(wid*hei*args.interp_steps*0.12))**0.5
        rate = min(1, rate)
        while(True):
            w, h = wid*rate, hei*rate
            input_images = os.path.join(output, "*.png")
            output_path = os.path.join(output, "output.gif")
            ret = os.system("gifski %s --width %d --height %d --fps 8 -o %s"%(input_images, w, h, output_path))
            if(ret!=0):
                print("using gifski to make gif but seem it's not installed")
                break
            cur = os.path.getsize(output_path)
            if(cur<target_filesize):
                break
            else:
                rate *= 0.97
                rate *= (target_filesize/cur)**0.5
        print(output_path)