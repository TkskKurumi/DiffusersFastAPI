from .paths import main_pth
from os import path
from .lvldb import TypedLevelDB
from threading import Lock
import time
from .candy import locked
running_count = {}
running_count_n = {}
db = TypedLevelDB.open(path.join(main_pth, "timer"))
lck = Lock()
db_lck = Lock()
windows_size = 10*60  # 10 minutes


def do_load_record(name):
    tm = time.time()
    with locked(db_lck):
        info = db.get(name, {})
        load_record = info.get("load_record", [])
        idx = 0
        for idx, i in enumerate(load_record):
            tm1, load1 = i
            if (tm1 > tm-windows_size):
                break
        load_record = load_record[idx:]
        load_record.append([tm, running_count[name]])
        info["load_record"] = load_record
        db[name] = info


def do_speed_record(name, n, elapsed):
    tm = time.time()
    with locked(db_lck):
        info = db.get(name, {})
        speed_record = info.get("speed_record", [])
        idx = 0
        for idx, i in enumerate(speed_record):
            tm1, n1, ela1 = i
            if (tm1 > tm-windows_size):
                break
        speed_record = speed_record[idx:]
        speed_record.append((tm, n, elapsed))
        
        info["speed_record"] = speed_record
        db[name] = info
def get_speed(name):
    info = db.get(name, {})
    speed_record = info.get("speed_record", [])
    tot_cnt = 0
    tot_ela = 0
    for tm, cnt, ela in speed_record:
        tot_cnt += cnt
        tot_ela += ela
    
    return tot_cnt/(tot_ela+1e-7)

def get_eta(name, n):
    return (n+running_count_n.get(name, 0))/(get_speed(name)+1e-7)



class Timer:
    def __init__(self, name, n=1):
        self.name = name
        with locked(lck):
            running_count[name] = running_count.get(name, 0)+1
            running_count_n[name] = running_count_n.get(name, 0)+n
        do_load_record(name)
        self.n = n

    def start(self):
        self.start_time = time.time()

    def end(self, success=True):
        with locked(lck):
            running_count[self.name] -= 1
            running_count_n[self.name] -= self.n
        
        do_load_record(self.name)
        if(success):
            elapsed = time.time()-self.start_time
            do_speed_record(self.name, self.n, elapsed)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if(exc_type is None):
            self.end()
        else:
            self.end(success=False)

