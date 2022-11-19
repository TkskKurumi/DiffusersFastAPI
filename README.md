# Diffusers-FastAPI
The project is intended to provide Stable-Diffusion web service as simple as possible.
## Features
+ extra-long prompt overcoming clip model's 75-word limit
+ + simply seprate prompt into slices under 75 words, and combine multiple prompt condition in diffusion
+ txt2img
+ img2img
+ txt2img with prompt interpolation
+ + this generate a sequence of image (video), smoothly alter from original prompt to target prompt.

## Client
Reading codes in examples folder is enough for understanding APIs, maybe, I think.
### "Ticket"
The idea is to memorize each image generation requests in specific IDs.

This may help _future_ features like reproducing result from previous image-generation; change noise/latent by little scale each frame for image-sequence (video) generation. etc.