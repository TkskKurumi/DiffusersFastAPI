import requests
from PIL import Image
from io import BytesIO
import os
HOST = "http://localhost:8000"

def img2bio(img):
    bio = BytesIO()
    if("A" not in img.mode):
        img.save(bio, "JPEG")
    else:
        img.save(bio, "PNG")
    bio.seek(0)
    return bio
class DiffuserFastAPITicket:
    def __init__(self, purpose):
        r = requests.get(HOST+"/ticket/create/%s" % purpose)
        self.ticket_id = r.json()["data"]["id"]
        self.ticket_url = HOST+"/ticket/"+self.ticket_id

    def param(self, **kwargs):
        data = {"data": kwargs}
        r = requests.post(self.ticket_url+"/param", json=kwargs)
        return r.json()
    @property
    def result(self):
        r = requests.get(self.ticket_url+"/result")
        return r.json()
    def submit(self):
        r = requests.get(self.ticket_url+"/submit")
        return r.json()
    def upload_image(self, image, name="orig_image"):
        url = self.ticket_url+"/upload_image?fn="+name
        bio = img2bio(image)
        return requests.post(url, files={"data":bio}).json()
    def get_image_seq(self):
        result = self.result
        data = result["data"]
        data_type = data["type"]
        assert data_type == "image_sequence", "Response Data is not Image Sequence"
        ret = []
        for i in data["images"]:
            r = requests.get(HOST+"/images/"+i)
            bio = BytesIO()
            bio.write(r.content)
            bio.seek(0)
            im = Image.open(bio)
            ret.append(im)
        return ret
    def get_image(self):
        result = self.result
        data = result["data"]
        data_type = data["type"]
        if(data_type == "image"):
            image = data["image"]
        elif(data_type.startswith("image_seq")):
            image = data["images"][0]
        else:
            raise TypeError(data_type)
        r = requests.get(HOST+"/images/"+image)
        bio = BytesIO()
        bio.write(r.content)
        bio.seek(0)
        im = Image.open(bio)
        return im

