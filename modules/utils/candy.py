import time
class locked:
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        return
def lockedmethod(func, lck=None):
    def inner(self, *args, **kwargs):
        with locked(self.lck):
            ret = func(self, *args, **kwargs)
        return ret
    if(lck is None):
        return inner
    def inner1(*args, **kwargs):
        nonlocal lck
        with locked(lck):
            ret = func(*args, **kwargs)
        return ret
    return inner1
class FakeLock:
    # for debug
    def __init__(self):
        pass
    def acquire(self):
        print(self, ".acquire")
    def release(self):
        print(self, ".release")
class released:
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        self.lock.release()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.acquire()
        return
class print_time:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

    def __enter__(self):
        self.time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time()-self.time
        if(t > 1 and self.enabled):
            print("%s uses %.1f seconds" % (self.name, t))