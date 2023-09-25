import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionXLInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

def resize(value, img):
    img = Image.open(img)
    img = img.resize((value, value))
    return img

def predict(source_img, prompt, negative_prompt, num_inference_steps, seed):
    torch.manual_seed(seed)


    imageio.imwrite("data.png", source_img["image"])
    imageio.imwrite("data_mask.png", source_img["mask"])
    src = resize(1080, "data.png")
    src.save("src.png")
    mask = resize(1080, "data_mask.png")
    mask.save("mask.png")
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=src, mask_image=mask, num_inference_steps=num_inference_steps).images[0]
    return image

title='DIAMONIK7777 - SD - Avatar&Selfie - Inpainting'
description="<p style='text-align: center'>Будь в курсе обновлений <a href='https://vk.com/public221489796'>ПОДПИСАТЬСЯ</a></p>"
article="<br><br><p style='text-align: center'>Генерация индивидуальной модели с собственной внешностью <a href='https://vk.com/im?sel=-221489796'>ПОДАТЬ ЗАЯВКУ</a></p><br><br><br><br><br>"

inputs = [
    gr.inputs.Image(source="upload", tool="sketch", label="Source Image"),
    gr.inputs.Textbox(label="Что вы хотите, чтобы ИИ генерировал в области маски?"),
    gr.inputs.Textbox(label='Что вы не хотите, чтобы ИИ генерировал в области маски?', default='(Face disproportionate to body, deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, old face'),
    gr.inputs.Slider(1, 50, 1, default=20, label="Количество итераций"),
    gr.inputs.Number(default=7096774193548387, label="Точка старта функции")
]

interface = gr.Interface(fn=predict, inputs=inputs, outputs="image", title=title, description=description, article=article)
interface.launch(debug=True, max_threads=True, share=True, inbrowser=True)
