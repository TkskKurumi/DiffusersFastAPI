from client import DiffuserFastAPITicket
if(__name__=="__main__"):
    prompt = "seaside sunset"
    t = DiffuserFastAPITicket("txt2img")
    t.param(prompt=prompt, aspect=1.78)
    print(t.submit())
    print(t.result)
    t.get_image().save("tmp.png")