import torch
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from matplotlib import pyplot as plt
import datetime
import math
from PIL import Image as img
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
#, use_safetensors=True, use_auth_token="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"
# pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
# pipeline = DiffusionPipeline.from_pretrained("eimiss/EimisAnimeDiffusion_1.0v",  use_auth_token="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")
#pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

pipeline = pipeline.to("cuda")
#image = pipeline("a girl, cute, fluffy blonde hair, blue eyes, castle, night, fireflies, stars").images[0]

#pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
init_image = load_image(img.open(r'./Hemi.png')).convert("RGB")
print("###########")
image = pipeline('''blush, bridal gauntlets, capelet, white long dress, winged capelet, yellow hair, hair band, hair between eyes, hair ornament, jewelry, extra short hair, beautiful detailed background, upper body, shoulder wing, white gold theme, indoor, royal palace, glowing light, wind, flowers''', image=init_image).images
print("=======================")
print(torch.cuda.is_available())
print("=======================")
print(len(image))
# plt.imshow(image[0])
# plt.show()
dateTime = math.ceil(datetime.datetime.now().timestamp())
#plt.savefig(f'/downloads/newImage_{dateTime}.png', bbox_inches='tight')
#image.save(f'/downloads/newImage_{dateTime}.png')
image[0].save(f'./downloads/image_to_image/newImage_{dateTime}.png')
