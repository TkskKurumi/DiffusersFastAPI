from . import learning
import numpy as np
def get_n_models():
    return len(learning.MODELS)

def get_model_index(name_or_index):
    if(isinstance(name_or_index, str)):
        index = get_n_models()+1
        for idx, model in enumerate(learning.MODELS):
            if(model.name == name_or_index):
                index = idx
    elif(name_or_index is None):
        index = get_n_models()+1
    else:
        assert isinstance(name_or_index, int)
        index = name_or_index
    return index
def apply_model(name_or_index):
    index = get_model_index(name_or_index)
    
    if(index<get_n_models()):
        learning.one_model_hot(index)
    else:
        for opt in learning.OPTS:
            opt.i.apply(opt.w)



class StableDraw:
    def __init__(self):
        self.repros = {}
        self.drawn = {}
    def __call__(self, model_idx, uniq_id=0, **kwargs):
        key = []
        key.append(uniq_id)
        key.extend((i, j) for i, j in kwargs.items())

        key_for_repro = tuple(key)

        model_idx = get_model_index(model_idx)

        if(model_idx<get_n_models()):
            key.append(learning.MODELS[model_idx].pth)
        else:
            for opt in learning.OPTS:
                w = opt.w
                w_model_avg = np.mean(w, axis=0)
                w_0 = w[0, :]
                key.append(tuple(w_model_avg))
                key.append(tuple(w_0))

        key_for_image = tuple(key)
        print(key_for_image)
        if(key_for_image in self.drawn):
            return self.drawn[key_for_image]
        
        apply_model(model_idx)
        repro = self.repros.get(key_for_repro)

        if(repro is None):
            repro = learning.MODEL.txt2img(**kwargs)
            image = repro.result
        else:
            image = repro.reproduce().result
        
        self.repros[key_for_repro] = repro
        self.drawn[key_for_image] = image

        return image
