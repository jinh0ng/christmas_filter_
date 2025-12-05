import gradio as gr
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.utils import load_image

model_id = "pwnhyo/instruct-pix2pix-model"

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipeline.safety_checker = None

def edit_image(
    input_image, 
    prompt, 
    num_inference_steps, 
    image_guidance_scale, 
    guidance_scale,
    seed
):
    generator = torch.Generator("cuda").manual_seed(int(seed))
    edited_image = pipeline(
        prompt,
        image=input_image,
        num_inference_steps=int(num_inference_steps),
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    return edited_image


demo = gr.Interface(
    fn=edit_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Textbox(lines=1, value="Add a santa hat", label="Prompt"),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=20,
            step=1,
            label="Steps"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=5,
            value=1.5,
            step=0.1,
            label="Image CFG"
        ),
        gr.Slider(
            minimum=1.0,
            maximum=15,
            value=7.5,
            step=0.5,
            label="Guidance Scale"
        ),
        gr.Number(
            value=422,
            label="Seed",
            precision=0,
        )
    ],
    outputs=gr.Image(type="pil", label="Edited Image"),
    title="InstructPix2Pix Demo",
    description="Upload an image and provide an instruction to edit it."
)


if __name__ == "__main__":
    demo.launch(share=True) 