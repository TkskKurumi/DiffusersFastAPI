from .base import BaseWidget, none_or, prepare_contents
from .resize import limit as limit_size
from .resize import resize
from PIL import Image


class Row(BaseWidget):
    def __init__(self, contents, margin=None, padding=None, limit_height=None, same_height=None, bg=None, align_y=None):
        self.contents = contents
        self.margin = margin
        self.padding = padding
        self.limit_height = limit_height
        self.align_y = align_y
        self.bg = bg
        self.same_height = same_height
        

    def render(self, **kwargs):
        def get(key, *args, **kwa):
            nonlocal kwargs, self
            if(kwa):
                kwargs = dict(kwargs)  # copied
                kwargs.update(kwa)
            return self._get(key, *args, **kwargs)
        contents = get("contents")
        limit_height = get("limit_height", None)
        margin = get("margin", 10)
        padding = get("padding", 10)
        same_height = get("same_height", None)
        bg = get("bg", (0, 0, 0, 0))
        align_y = get("align_y", 0.5)

        contents = prepare_contents(contents, **kwargs)
        if(same_height is not None):
            if(same_height == "max"):
                same_height = max([i.size[1] for i in contents])
            elif(same_height == "min"):
                same_height = min([i.size[1] for i in contents])
            elif(isinstance(same_height, int) or isinstance(same_height, float)):
                pass
            else:
                raise ValueError("same_height %.2f" % (same_height))
            contents = [resize(i, height=same_height) for i in contents]
        if(limit_height is not None):
            contents = [limit_size(i, height=limit_height) for i in contents]

        maxh = max([i.size[1] for i in contents])
        width = sum([i.size[0] for i in contents]) + \
            padding*2 + margin*(len(contents)-1)
        height = maxh + padding*2

        ret = Image.new("RGBA", (width, height), bg)
        left = padding
        for idx, i in enumerate(contents):
            w, h = i.size
            top = int((maxh-h)*align_y + padding)
            has_mask = "A" in i.mode
            if(has_mask):
                ret.paste(i, box=(left, top), mask=i)
            else:
                ret.paste(i, box=(left, top))
            left += w+margin
        return ret
if (__name__ == "__main__"):
    from glob import glob
    contents = []
    for i in glob("./*.png"):
        # contents.append(i+"\n")
        contents.append(Image.open(i))
    RT = Row(contents, bg=(255, 255, 255, 255), align_y=0.5)
    RT.render().save("temp.png")
