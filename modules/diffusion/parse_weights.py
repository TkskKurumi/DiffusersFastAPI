import re


class WeightedPrompt:
    def __init__(self, prompt):
        if (isinstance(prompt, str)):
            self.init_str(prompt)
        else:
            raise ValueError("Unsupported prompt type%s" % type(prompt))

    def init_str(self, prompt: str):
        pattern = r"[{}]|/\*|\*/"
        ret = []
        operators = re.findall(pattern, prompt)
        segments = re.split(pattern, prompt)
        curr_weight = 1
        for idx, seg in enumerate(segments):
            if(seg.strip(" ")):
                ret.append((curr_weight, seg))
            if (idx < len(operators)):
                op = operators[idx]
                if (op == "{"):
                    curr_weight *= 1.1
                elif (op == "}"):
                    curr_weight /= 1.1
                else:
                    curr_weight *= -1
        self.sentences = ret

    def __iter__(self):
        return iter(self.sentences)
    def as_dict(self):
        ret = {}
        for w, s in self:
            ret[w] = ret.get(w, [])
            if(ret[w]):
                ret[w]+=", "
            ret[w] += s
        return ret
if(__name__=="__main__"):
    prompt = "hello, {world}, /*bad guy*/, ok"
    print(list(WeightedPrompt(prompt)))