from ...utils.candy import print_time
from tqdm import tqdm, trange
from ...utils.debug_vram import debug_vram
import numpy as np
def softmax(w: np.ndarray, axis=-1):
    exp = np.exp(w)
    expsum = exp.sum(axis=axis, keepdims=True)
    ret = exp/expsum
    return ret
class Interpolate:
    def __init__(self, name, dst_model, states, names):
        self.dst_model = dst_model
        self.name = name
        self.states = states
        self.names = names
        self.n_model = len(names)
        self.n_ch = len(states[0])
        self.keys = list(states[0].keys())
        self._current = None
        self._current_norm = None
    def display(self):
        print(self.name, *self.names, end="\n")
        print(" "*len(self.name), end="")
        for idx, name in enumerate(self.names):
            contribution = self._current_norm[:, idx].mean()
            n = len(name)
            # %n.2f%%
            format = "%" + str(n) + ".2f" + "%%"
            print(format%(100*contribution), end="")
        print()
        
    def zeros(self):
        return np.zeros((self.n_ch, self.n_model))
    def randn(self):
        return np.random.normal(size=(self.n_ch, self.n_model))
    
    def apply(self, w, do_softmax=True):
        EPS = 1e-5
        dry = False
        prev_w = self._current_norm
        prev_sd = self.dst_model.state_dict()
        if(do_softmax):
            self._current = w
            w = softmax(w)
            if(self._current_norm is not None):
                diff = np.abs(w-self._current_norm)
                if(np.all(diff<EPS)):
                    dry = True
            self._current_norm = w
        else:
            self._current = np.log(w)
            if(self._current_norm is not None):
                diff = np.abs(w-self._current_norm)
                if(np.all(diff<EPS)):
                    dry = True
            self._current_norm = w
        assert not (np.any(np.isnan(self._current_norm)))
        if(not dry):
            with print_time("interpolate model %s" % self.name):
                ret = {}
                ls = list(enumerate(self.keys))
                for key_idx, key in tqdm(ls):
                    _dry = False
                    if(prev_w is not None):
                        ws = w[key_idx]
                        prev_ws = prev_w[key_idx]
                        diff = np.abs(ws-prev_ws)
                        if(np.all(diff<EPS)):
                            tensor = prev_sd[key]
                            _dry = True
                    if(not _dry):
                        tensor = 0
                        for model_idx, sd in enumerate(self.states):
                            model_tensor = sd[key]
                            model_w = w[key_idx, model_idx]
                            if(abs(model_w)>1e-6):
                                tensor += model_tensor*model_w
                    ret[key] = tensor
                self.dst_model.load_state_dict(ret)
        self.display()
    def dump_detail(self, prt):
        rows = []
        rows.append((self.name, *self.names))
        rows.append(["-"]*len(rows[0]))
        for key_idx, key in enumerate(self.keys):
            ws = self._current_norm[key_idx]
            row = [self.name+"."+key]
            for w in ws:
                row.append("%.2f%%"%(w*100))
            rows.append(row)
        n_columns = len(rows[0])
        column_w = []
        for i in range(n_columns):
            w = 0
            for jdx, j in enumerate(rows):
                w = max(w, len(j[i]))
            column_w.append(w)
        for row_i, row in enumerate(rows):
            prt("| ", end="")
            for col_i, elem in enumerate(row):
                if(col_i):
                    prt(" ", end="")
                compensate = column_w[col_i]-len(elem)
                prt(elem+" "*compensate, end=" |")
            prt("")
        
