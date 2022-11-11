from os import path
import os
home = path.expanduser("~")
base = os.environ.get("DIFFUSION_FAST_API_V3_DATABASE", "dev")
main_pth = path.join(home, ".diffusion_fast_api_v3", base)