import torch
from diffusers import StableDiffusionPipeline
import datetime
import math
from PIL import Image, ImageChops


def generate_image_from_text(prompt, save_path):
    prompt = prompt if prompt!="" else '''blush, bridal gauntlets, capelet, white long dress, winged capelet, yellow hair, hair band, hair between eyes, hair ornament, jewelry, extra short hair, beautiful detailed background, upper body, shoulder wing, white gold theme, indoor, royal palace, glowing light, wind, flowers'''
    try:
        pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        
        pipeline = pipeline.to("cuda")
    
        image = pipeline(prompt).images
        dateTime = math.ceil(datetime.datetime.now().timestamp())
        filename = f'{save_path}/newImage_{dateTime}.png'
        image[0].save(filename)
        
    except BaseException:
        filename = ""
    
    return filename

