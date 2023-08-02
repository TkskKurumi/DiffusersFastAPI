from typing import Any
import json
from os import path

class CFG:
    _ignored = {"_batch_mode", "_ignored"}
    def _save(self):
        raise NotImplementedError()
    def __setattr__(self, __name: str, __value: Any) -> None:
        if(__name in self._ignored):
            return super().__setattr__(__name, __value)
        if(isinstance(__value, dict)):
            __value = ChildCFG(self, __name, __value)
        ret = super().__setattr__(__name, __value)
        if(not self._batch_mode):
            self._save()
        return ret
    def _asdict(self):
        ret = {}
        for k, v in self.__dict__.items():
            if(k in self._ignored):
                continue
            if(isinstance(v, CFG)):
                v = v._asdict()
            ret[k] = v
        return ret
    def _save(self):
        raise NotImplementedError()
    
    def default(self, val, default):
        if(val in self.__dict__):
            return self.__dict__[val]
        setattr(self, val, default)
        return default
    def __getitem__(self, key):
        return self.__dict__[key]
class ChildCFG(CFG):
    _ignored = {"_batch_mode", "_fa", "_key"}
    def __init__(self, fa, key, d):
        self._batch_mode = True
        for k, v in d.items():
            setattr(self, k, v)
        self._batch_mode = False
        self._fa = fa
        self._key = key
    def _save(self):
        setattr(self._fa, self._key, self)
class FileCFG(CFG):
    _ignored = {"_batch_mode", "_ignored", "_pth"}
    def __init__(self, pth):
        self._batch_mode = True
        if(path.exists(pth)):
            with open(pth, "r") as f:
                d = json.load(f)
            for k, v in d.items():
                setattr(self, k, v)
        self._pth = pth
        self._batch_mode = False
    def _save(self):
        with open(self._pth, "w") as f:
            json.dump(self._asdict(), f)
AppCFG = FileCFG("./rmhf_web.json")
if(__name__=="__main__"):
    cfg = FileCFG("test.json")
    print(cfg._asdict())
    print(cfg.foo)
    cfg.foo = "bar"
    cfg.sub = {}
    cfg.sub.aa = "meow"
    cfg.default("subcfg", {"meow": "meowmeow"})