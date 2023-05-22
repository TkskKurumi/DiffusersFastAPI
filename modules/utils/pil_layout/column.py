from .base import BaseWidget, none_or, prepare_contents
from .resize import limit as limit_size
from .resize import resize
from PIL import Image


class Column(BaseWidget):
    def __init__(self, contents, margin=None, padding=None, limit_width=None, same_width=None, bg=None, align_x=None):
        self.contents = contents
        self.margin = margin
        self.padding = padding
        self.limit_width = limit_width
        self.align_x = align_x
        self.bg = bg
        self.same_width = same_width
        

    def render(self, **kwargs):
        def get(key, *args, **kwa):
            nonlocal kwargs, self
            if(kwa):
                kwargs = dict(kwargs)  # copied
                kwargs.update(kwa)
            return self._get(key, *args, **kwargs)
        contents = get("contents")
        limit_width = get("limit_width", None)
        margin = get("margin", 10)
        padding = get("padding", 10)
        same_width = get("same_width", None)
        bg = get("bg", (0, 0, 0, 0))
        align_x = get("align_x", 0.5)

        contents = prepare_contents(contents, **kwargs)
        if(same_width is not None):
            if(same_width == "max"):
                same_width = max([i.size[1] for i in contents])
            elif(same_width == "min"):
                same_width = min([i.size[1] for i in contents])
            elif(isinstance(same_width, int) or isinstance(same_width, float)):
                pass
            else:
                raise ValueError("same_width %.2f" % (same_width))
            contents = [resize(i, height=same_width) for i in contents]
        if(limit_width is not None):
            contents = [limit_size(i, width=limit_width) for i in contents]

        maxw = max([i.size[0] for i in contents])
        height = sum([i.size[1] for i in contents]) + \
            padding*2 + margin*(len(contents)-1)
        width = maxw + padding*2

        ret = Image.new("RGBA", (width, height), bg)
        top = padding
        for idx, i in enumerate(contents):
            w, h = i.size
            left = int((maxw-w)*align_x + padding)
            has_mask = "A" in i.mode
            if(has_mask):
                ret.paste(i, box=(left, top), mask=i)
            else:
                ret.paste(i, box=(left, top))
            top += h+margin
        return ret
if (__name__ == "__main__"):
    from glob import glob
    contents = []
    for i in glob("./*.png"):
        # contents.append(i+"\n")
        contents.append(Image.open(i))
    RT = Column(contents, bg=(255, 255, 255, 255), align_x=0.25, same_width="min")
    RT.render().save("temp.png")
