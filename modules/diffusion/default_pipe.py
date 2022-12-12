import os
if(os.environ.get("DIFFUSION_MODEL") == "ANYTHING_V3"):
    from .default_pipe_anything_V3 import pipe
else:
    from .default_pipe_old import pipe
