import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
import os
#os.environ["HF_HOME"] = "/d1/hyomin/.cache3"

model = "pwnhyo/instruct-pix2pix-model"
image = load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_1.png")

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
pipeline.safety_checker = None
generator = torch.Generator("cuda").manual_seed(422)

prompt = "Add a santa hat"

edited_image = pipeline(
   prompt,
   image=image,
   num_inference_steps=20,
   image_guidance_scale=1.5,
   guidance_scale=10,
   generator=generator,
).images[0]
edited_image.save("edited_image_test11.png")

#CUDA_VISIBLE_DEVICES=1 python inference.py