from .client import DiffuserFastAPITicket

if(__name__=="__main__"):
    pr = prompt="seaside, sunshine, coconut tree"
    t1 = DiffuserFastAPITicket("txt2img")
    t1.param(prompt=pr, aspect=1, fast_eval=True)
    
    result = t1.result
    noise = result["data"]["noise"]

    t2 = DiffuserFastAPITicket("txt2img")
    t2.param(prompt=pr, aspect=1, fast_eval=True, use_noise=noise, use_noise_alpha=1)

    t3 = DiffuserFastAPITicket("txt2img")
    t3.param(prompt=pr, aspect=1, fast_eval=True, use_noise=noise, use_noise_alpha=0.2)

    t1.get_image().save("tmp0.png")
    t2.get_image().save("tmp1.png")
    t3.get_image().save("tmp2.png")
