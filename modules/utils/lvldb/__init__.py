from leveldb import LevelDB as BaseLevelDB
import time
from ..candy import locked, lockedmethod, FakeLock, released
from threading import Lock
from io import BytesIO
import json
from .serialize import asbytes, list_as_bytes, list_from_bytes
from .serialize import base_from_types
from os import path
import os

class TypedLevelDB:
    opened = dict()
    opener_lock = Lock()
    @classmethod
    def open(cls, pth, from_bytes = None):
        with locked(cls.opener_lock):
            if(pth in cls.opened):
                ret = cls.opened[pth]
            else:
                ret = TypedLevelDB(pth, from_bytes)
                cls.opened[pth] = ret
        return ret
    def __init__(self, pth, from_bytes=None):
        self.lck = Lock()
        if(not path.exists(pth)):
            os.makedirs(pth)
        self.db = BaseLevelDB(path.join(pth, "data"))
        self.db_type = BaseLevelDB(path.join(pth, "type"))
        self.from_bytes = base_from_types
        if(from_bytes is not None):
            self.from_bytes.update(from_bytes)
    @lockedmethod
    def pop(self, key, default):
        type_k, b_k = asbytes(key)
        try:
            ret = self.db.Get(b_k)
        except KeyError:
            ret = default
        return ret
    @lockedmethod
    def __setitem__(self, key, value):
        type_k, b_k = asbytes(key)
        type_v, b_v = asbytes(value)
        b_type_item = list_as_bytes([type_k, type_v])
        self.db.Put(b_k, b_v)
        self.db_type.Put(b_k, b_type_item)

    @lockedmethod
    def __getitem__(self, key):
        type_k, b_k = asbytes(key)
        b_v = self.db.Get(b_k)
        b_type_item = self.db_type.Get(b_k)
        type_k, type_v = list_from_bytes(b_type_item)
        ret = self.from_bytes[type_v](b_v)
        return ret
    def get(self, key, default):
        if(key in self):
            return self[key]
        return default
    def items(self):
        for b_k, b_v in self.db.RangeIter():
            b_type_item = self.db_type.Get(b_k)
            type_k, type_v = list_from_bytes(b_type_item)
            key = self.from_bytes[type_k](b_k)
            value = self.from_bytes[type_v](b_v)
            yield (key, value)
    def __iter__(self):
        for b_k in self.db.RangeIter(include_value = False):
            b_type_item = self.db_type.Get(b_k)
            type_k, type_v = list_from_bytes(b_type_item)
            key = self.from_bytes[type_k](b_k)
            yield key
    @lockedmethod
    def __contains__(self, key):
        type_k, b_k = asbytes(key)
        try:
            b_v = self.db.Get(b_k)
            ret = True
        except KeyError:
            ret = False
        return ret
if(__name__ == "__main__"):
    import time
    test = TypedLevelDB("./tmpdb")
    test[int(time.time())] = "nowtime %d" % time.time()
    test["foo"] = "bar"
    test["dict"] = {"foo": "bar", "hello": "world"}
    print(test["foo"])
    for k, v in test.items():
        print(k, ":", v)
    for k in test:
        print(k)
    print("niuzi" in test)
    print("foo" in test)