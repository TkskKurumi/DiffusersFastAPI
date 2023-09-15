import re


class WeightedPrompt:
    def __init__(self, prompt):
        if (isinstance(prompt, str)):
            self.init_str(prompt)
        else:
            raise ValueError("Unsupported prompt type%s" % type(prompt))

    def init_str(self, prompt: str):
        self.loras = {}

        def extract_lora(pro):
            pattern = r"(<lora:([\s\S]+?):(-?\d+|-?\d*\.\d*)>)"
            loras = re.findall(pattern, pro)
            for full, name, ratio in loras:
                self.loras[name] = self.loras.get(name, 0) + float(ratio)
                pro = pro.replace(full, "")
            return pro
        prompt = extract_lora(prompt)

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
            ret[w].append(s)
        for k, v in ret.items():
            ret[k] = ", ".join(v)
            ret[k] = re.sub("( *, *)+", ", ", ret[k])
        return ret
if(__name__=="__main__"):
    prompt = "<lora:foo1:-1><lora:foo2:1.0>hello, {world}, /*bad guy*/, ok"
    WP = WeightedPrompt(prompt)
    print(list(WP))
    print(WP.as_dict())
    print(WP.loras)