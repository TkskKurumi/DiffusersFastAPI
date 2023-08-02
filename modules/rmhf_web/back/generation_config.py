from . import permute
import random
from ...diffusion.wrapped_pipeline import CustomPipeline
from ...utils.misc import normalize_resolution
class GenerationConfig:
    def __init__(self, model: CustomPipeline, prompt_permute):
        self.model = model
        self.prompt_permute = prompt_permute
        self._repro = None
    def generate(self):
        if(self._repro):
            return self._repro.reproduce()

        pos = permute.sample_text(self.prompt_permute)
        ar = random.choice([9/16, 5/7, 3/4, 5/8, 1, 4/3])
        width, height = normalize_resolution(ar, 1, 512*512*1.8)
        repro = self.model.txt2img(pos, steps=30, width=width, height=height)
        self.repro = repro
        return repro
