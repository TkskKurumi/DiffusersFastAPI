HISTORY = []
LAST = ""
def log(*args, end="\n", sep=" "):
    global LAST, HISTORY
    LAST = sep.join([str(i) for i in args])
    HISTORY.append(LAST)
    HISTORY.append(end)