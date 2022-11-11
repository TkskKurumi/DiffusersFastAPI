from io import BytesIO, StringIO
import json


def dict_as_bytes(j):
    with StringIO() as sio:
        json.dump(j, sio, ensure_ascii = True)
        sio.seek(0)
        bytes = sio.read().encode("ascii")
    return bytes


def dict_from_bytes(bytes):
    with BytesIO() as bio:
        bio.write(bytes)
        bio.seek(0)
        j = json.load(bio)
    return j


def list_as_bytes(bytes):
    return dict_as_bytes(bytes)


def list_from_bytes(bytes):
    return dict_from_bytes(bytes)


def str_as_bytes(s):
    return s.encode("utf-8")


def str_from_bytes(b):
    return b.decode("utf-8")


def bytes_from_bytes(b):
    return b


def int_to_bytes(i):
    return i.to_bytes((i.bit_length() + 7) // 8, byteorder='little')


def int_from_bytes(b):
    return int.from_bytes(b, "little")


def asbytes(obj):
    if(hasattr(obj, "asbytes") and callable(obj.asbytes)):
        return obj.asbytes()
    elif(isinstance(obj, bytes)):
        return ("bytes", obj)
    elif(isinstance(obj, int)):
        return ("int", int_to_bytes(obj))
    elif(isinstance(obj, str)):
        return ("str", obj.encode("utf-8"))
    elif(isinstance(obj, dict) or isinstance(obj, list)):
        return ("dict", dict_as_bytes(obj))
    else:
        raise TypeError("Cannot intepret %s as bytes" % (type(obj), ))
base_from_types = {
    "int": int_from_bytes,
    "str": str_from_bytes,
    "dict": dict_from_bytes
}