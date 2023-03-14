from .base import BaseWidget, none_or, prepare_contents
from .resize import limit as limit_size
from .resize import resize
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import re


class _LineFeed:
    pass


LF = _LineFeed()
DEBUG_SEARCH_FONT = True
def search_font(size):
    global DEBUG_SEARCH_FONT
    verbose = DEBUG_SEARCH_FONT
    DEBUG_SEARCH_FONT = False
    ls = ["DejaVuSansMono.ttf", "/usr/share/fonts/DejaVuSansMono.ttf"]
    for i in ls:
        if(verbose):
            print("look for %s"%i)
        try:
            return ImageFont.truetype(i, size=size)
        except Exception:
            print("not found")
    print("default font")
    return ImageFont.load_default()
class RichText(BaseWidget):
    def __init__(self, contents, fill=None, bg=None, width=None, font=search_font, font_size=None, padding=None, margin=None, align_x=None, align_y=None, limit_height=None, limit_width=None):
        self.contents = contents
        self.bg = bg
        self.fill = fill
        self.width = width
        self.font_size = font_size
        self.padding = padding
        self.margin = margin
        self.align_x = align_x
        self.align_y = align_y
        self.font = font
        self.limit_height = limit_height
        self.limit_width = limit_width

    def render(self, **kwargs):
        def get(key, *args, **kwa):
            nonlocal kwargs, self
            if(kwa):
                kwargs = dict(kwargs)  # copied
                kwargs.update(kwa)
            return self._get(key, *args, **kwargs)
        contents = get("contents")
        margin = get("margin", 5)
        bg = get("bg", (0, 0, 0, 0))
        fill = get("fill", (0, 0, 0, 255))
        font_size = get("font_size", 24)
        font = get("font", extract_callable=0)
        align_x = get("align_x", 0)
        align_y = get("align_y", 1)
        canvas_width = get("width", 512)
        padding = get("padding", 0)
        limit_height = get("limit_height", font_size*5)
        limit_width = get("limit_width", canvas_width-padding*2)
        # prepare font
        if(isinstance(font, str)):
            font = ImageFont.truetype(font, size=font_size)
        elif(callable(font)):
            font = font(size=font_size)

        # process line_feed
        _contents = []
        for i in contents:
            if(isinstance(i, str)):
                for jdx, j in enumerate(re.split("[\n\r]", i)):
                    if(jdx):
                        _contents.append(LF)
                    _contents.append(j)
            else:
                i = limit_size(i, width=limit_width, height=limit_height)
                _contents.append(i)
        contents = _contents

        def render_line(line, return_width=False):
            line = [i for i in line if i is not LF]
            if(not line):
                if(return_width):
                    return 1
                else:
                    return Image.new("RGBA", (1, font_size), (0, 0, 0, 0))

            # prepare contents
            _line = []
            width = 0
            st = ""
            for i in line:
                if(isinstance(i, Image.Image)):
                    if(st):
                        _line.append(st)
                    _line.append(i)
                elif(isinstance(i, str)):
                    st += i
                else:
                    raise TypeError(type(i))
            if (st):
                _line.append(st)
            line = _line

            # prepare shape
            width = 0
            height = 0
            for idx, i in enumerate(line):
                if(idx):
                    width += margin
                if(isinstance(i, str)):
                    le, up, ri, lo = font.getbbox(i)

                    width += ri
                    height = max(height, lo)
                elif(isinstance(i, Image.Image)):
                    width += i.size[0]
                    height = max(height, i.size[1])
                else:
                    raise TypeError(type(i))
            if(return_width):
                return width

            ret = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            dr = ImageDraw.Draw(ret)
            left = 0
            for idx, i in enumerate(line):
                if(idx):
                    left += margin
                if(isinstance(i, str)):
                    le, up, ri, lo = font.getbbox(i)
                    top = int((height-lo)*align_y)
                    dr.text((left, top), i, font=font, fill=fill)
                    left += ri
                elif(isinstance(i, Image.Image)):
                    w, h = i.size
                    top = int((height-h)*align_y)

                    has_mask = "A" in i.mode
                    if(has_mask):
                        ret.paste(i, (left, top), mask=i)
                    else:
                        ret.paste(i, (left, top))
                    left += w
                else:
                    raise TypeError(type(i))
            return ret

        def should_feed(line):
            if(line and line[-1] is LF):
                return True
            return render_line(line, True) > canvas_width-padding*2
        lines = []
        cur_line = []
        for i in contents:
            if(should_feed(cur_line+[i])):
                rendered = render_line(cur_line)
                lines.append(rendered)
                cur_line = []
            if(i is not LF):
                cur_line.append(i)
        if(cur_line):
            rendered = render_line(cur_line)
            lines.append(rendered)

        # width = 0
        height = padding*2
        for idx, i in enumerate(lines):
            if(idx):
                height += margin
            height += i.size[1]

        ret = Image.new("RGBA", (canvas_width, height), bg)
        top = padding

        for idx, i in enumerate(lines):
            w, h = i.size
            left = int(padding + (canvas_width - 2*padding - w)*align_x)
            if(idx):
                top += margin
            has_mask = "A" in i.mode
            if(has_mask):
                ret.paste(i, box=(left, top), mask=i)
            else:
                ret.paste(i, box=(left, top))
            top += i.size[1]
        return ret


if (__name__ == "__main__"):
    from glob import glob
    contents = []
    for i in glob("./*.png"):
        contents.append(i+"\n")
        contents.append(Image.open(i))
    RT = RichText(contents, bg=(255, 255, 255, 255), align_x=0.5)
    RT.render().save("temp.png")
