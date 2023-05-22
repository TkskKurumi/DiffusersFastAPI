
def none_or(*args):
    for i in args:
        if(i is not None):
            return i
    return None

def prepare_contents(i, extract_callable=5, **kwargs):
    while(extract_callable>0):
        if(callable(i)):
            i = i(**kwargs)
            extract_callable -= 1
        else:
            break
    
    if(isinstance(i, list)):
        ret = []
        for j in i:
            ret.append(prepare_contents(j, extract_callable=extract_callable, **kwargs))
        return ret
    return i

class BaseWidget:
    def __call__(self, **kwargs):
        return self.render(**kwargs)

    def render(self, **kwargs):
        return NotImplemented
    def _get(self, key, *args, **kwargs):
        assert hasattr(self, key)
        if(getattr(self, key, None) is not None):
            ret = getattr(self, key)
        elif(kwargs.get(key) is not None):
            ret = kwargs[key]
        elif(args):
            ret = args[0]
        else:
            raise KeyError(key)
        ret = prepare_contents(ret, **kwargs)
        return ret
        
    