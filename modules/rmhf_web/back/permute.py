from __future__ import annotations
from functools import partial
from typing import List
from types import NoneType
import random
from math import ceil


class Status:
    def __init__(self, texts=None, sub=None):
        self.texts = [] if texts is None else texts
        self.sub = {} if sub is None else sub
    def copy(self):
        return Status(list(self.texts), dict(self.sub))

class Node:
    def __init__(self, name="unknown", apply=None, revert=None):
        self.to = []
        self.name = name
        self.f_apply = apply
        self.f_revert = revert
    def connect(self, other: Node):
        self.to.append(other)
        return other
    def __repr__(self):
        return "<Node name=%s>"%(self.name)
    def apply(self, S):
        if(callable(self.f_apply)):
            self.f_apply(self, S)
        else:
            pass
    def revert(self, S):
        if(callable(self.f_revert)):
            self.f_revert(self, S)
        else:
            pass



class Block:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
    def __repr__(self):
        return "<Block %s-%s>"%(self.head.name, self.tail.name)
class Branch(Block):
    def __init__(self, contents: List[Block]):
        super().__init__()
        for i in contents:
            self.head.connect(i.head)
            i.tail.connect(self.tail)

class Concat(Block):
    def __init__(self, contents: List[Block]):
        super().__init__()
        last = None
        for i in contents:
            if(last):
                last.tail.connect(i.head)
            else:
                self.head.connect(i.head)
            last = i
        last.tail.connect(self.tail)
class Single(Block):
    def __init__(self, content: Node):
        super().__init__()
        self.head.connect(content).connect(self.tail)

def _add_text(text):
    def apply(node: Node, S: Status):
        S.texts.append(text)
    def revert(node: Node, S: Status):
        S.texts.pop()
    return apply, revert
def _set_sub(*args):
    d_apply = {}
    d_revert = {}
    for i in range(1, len(args), 2):
        d_apply[args[i-1]] = args[i]    
    def apply(node: Node, S: Status):
        for k, v in d_apply.items():
            if(k in S.sub):
                d_revert[k] = S.sub[k]
            S.sub[k] = v
    def revert(node: Node, S: Status):
        for k, v in d_apply.items():
            if(k in d_revert):
                S.sub[k] = d_revert[k]
            else:
                S.sub.pop(k)
    return apply, revert
def _has_tag(tag, to0, to1):
    def apply(node: Node, S: Status):
        if(tag in S.sub):
            node.to = to1
        else:
            node.to = to0
    def revert(node: Node, S: Status):
        node.to = None
    return apply, revert
class IF(Block):
    def __init__(self, tag, then: Block, _else: Block|NoneType=None):
        self.tail = Node()
        if(_else is not None):
            _else.tail.connect(self.tail)
            to0 = [_else.head]
        else:
            to0 = [self.tail]
        
        then.tail.connect(self.tail)
        to1 = [then.head]
        app, rev = _has_tag(tag, to0, to1)

        self.head = Node(apply = app, revert=rev)
        

def Build(ls, route=None) -> Block:
    if(route is None):
        route = []
    nm = "_".join(["%s"%i for i in route])
    if(isinstance(ls, list)):
        typ = ls[0]
        if(typ.startswith("sub")):
            apply, revert = _set_sub(*ls[1:])
            node = Node(name="sub_"+nm, apply=apply, revert=revert)

            ret = Single(node)
            ret.head.name = node.name+"_pre"
            ret.tail.name = node.name+"_post"
            return ret
        elif(typ.startswith("branch")):
            contents = []
            for idx, i in enumerate(ls[1:]):
                route.append(idx)
                contents.append(Build(i, route))
                route.pop()
            ret = Branch(contents)
            ret.head.name = "branch_h_"+nm
            ret.tail.name = "branch_t_"+nm
            return ret
        elif(typ.startswith("concat")):
            contents = []
            for idx, i in enumerate(ls[1:]):
                route.append(idx)
                contents.append(Build(i, route))
                route.pop()
            ret = Concat(contents)
            ret.head.name = "concat_"+nm+"_h"
            ret.tail.name = "concat_"+nm+"_t"
            return ret
        elif(typ.startswith("if")):
            tag = ls[1]
            then = Build(ls[2])
            if(len(ls)>3):
                _else = Build(ls[3])
            else:
                _else = None
            ret = IF(tag, then, _else)
            return ret
            
        else:
            raise TypeError(typ)
    elif(isinstance(ls, str)):
        apply, revert = _add_text(ls)
        node = Node(name="text_"+nm, apply=apply, revert=revert)
        ret = Single(node)
        ret.head.name = nm+"_pre"
        ret.tail.name = nm+"_post"
        return ret
    else:
        raise TypeError(type(ls))
    
def sample_text(ls):
    if(isinstance(ls, list)):
        top = Build(ls)
    else:
        raise TypeError(type(ls))
    end = top.tail
    stack = []
    ret = None
    S = Status()
    def out():
        nonlocal stack, ret, S
        texts = []
        for t in S.texts:
            cnt = 0
            while(True):
                stop = True
                for k, v in S.sub.items():
                    if(k in t):
                        newt = t.replace(k, v)
                        if(newt!=t):
                            stop = False
                        t = newt
                        cnt += 1
                if(cnt>4096):
                    print(t, S.sub)
                    raise RecursionError("Potential circular substitution")
                if(stop):
                    break
            texts.append(t)
        # print(stack, texts)
        return "".join(texts)
    
    def recur(u: Node):
        nonlocal stack, end, S
        if(u is end):
            return out()
        for v in stack:
            assert v is not u, "%s %s"%(stack, u)
        stack.append(u)
        u.apply(S)
        to = u.to
        v = random.choice(to)
        return recur(v)
        stack.pop()
        u.revert(S)
    return recur(top.head)
def get_texts(ls, mx=2048):
    if(isinstance(ls, list)):
        top = Build(ls)
    elif(isinstance(ls, Block)):
        top = ls
    else:
        raise TypeError(type(ls))
    end = top.tail
    stack = []
    ret = []
    S = Status()
    def out():
        nonlocal stack, ret, S
        texts = []
        for t in S.texts:
            cnt = 0
            while(True):
                stop = True
                for k, v in S.sub.items():
                    if(k in t):
                        newt = t.replace(k, v)
                        if(newt!=t):
                            stop = False
                        t = newt
                        cnt += 1
                if(cnt>4096):
                    print(t, S.sub)
                    raise RecursionError("Potential circular substitution")
                if(stop):
                    break
            texts.append(t)
        # print(stack, texts)
        ret.append("".join(texts))
    def recur(u: Node, max_branch):
        nonlocal stack, end, S
        if(u is end):
            out()
        for v in stack:
            assert v is not u, "%s %s"%(stack, u)
        stack.append(u)
        u.apply(S)
        to = u.to
        if(len(to)>max_branch):
            to = random.sample(to, max(1, ceil(max_branch)))
        for v in to:
            recur(v, max_branch/len(to))
            # if(len(ret)>mx):break
        stack.pop()
        u.revert(S)
    recur(top.head, mx)
    if(len(ret)>mx):
        ret = random.sample(ret, mx)
    else:
        random.shuffle(ret)
    return ret


if(__name__=="__main__"):
    import json
    with open("rmhf2/prompt.permute", "r") as f:
        ls = json.load(f)
    for i in range(10):
        print(sample_text(ls))