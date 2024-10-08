import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import torch
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline, FlaxStableDiffusionPipeline
import datetime
import math
from PIL import Image, ImageChops


def generate_image_from_text(prompt, save_path):
    prompt = prompt if prompt!="" else '''blush, bridal gauntlets, capelet, white long dress, winged capelet, yellow hair, hair band, hair between eyes, hair ornament, jewelry, extra short hair, beautiful detailed background, upper body, shoulder wing, white gold theme, indoor, royal palace, glowing light, wind, flowers'''
    try:
        #pipeline = DiffusionPipeline.from_pretrained("eimiss/EimisAnimeDiffusion_1.0v")
        ##pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        ##pipeline = DiffusionPipeline.from_pretrained("xinsir/controlnet-union-sdxl-1.0")
        #pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        #pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        #pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
        print("0")
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="flax", dtype=jax.numpy.bfloat16)
        print("1")
        prng_seed = jax.random.PRNGKey(0)
        num_inference_steps = 50

        num_samples = jax.device_count()
        prompt = num_samples * [prompt]
        print("2")
        prompt_ids = pipeline.prepare_inputs(prompt)
        print("3")
        # shard inputs and rng
        params = replicate(params)
        prng_seed = jax.random.split(prng_seed, num_samples)
        prompt_ids = shard(prompt_ids)
        
        pipeline = pipeline.to("cuda")
        print("4")
        image = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
        image = pipeline.numpy_to_pil(np.asarray(image.reshape((num_samples,) + image.shape[-3:])))
        print("5")
    # pipeline = DiffusionPipeline.from_pretrained("eimiss/EimisAnimeDiffusion_2.0v", use_auth_token="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")
#,  use_auth_token="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"

        #pipeline = pipeline.to("cuda")
    
        #image = pipeline(prompt).images
        print("==========================")
        print(image[0])
        print("==========================")
        # img1 = image[0] #Image.open('downloads/text_to_image/newImage_1720387424.png')
        # img2 = Image.open('downloads/text_to_image/newImage_1720375841.png') 
        # diff = ImageChops.difference(img1, img2)   
        # if diff.getbbox():
        #     pass
        # else:
        #     raise ValueError("Potential NFSW Content")
        dateTime = math.ceil(datetime.datetime.now().timestamp())
        filename = f'{save_path}/newImage_{dateTime}.png'
        image[0].save(filename)
        
    except BaseException:
        print(BaseException.args)
        filename = ""
        
    finally:
        print("Finally chala")
            
    
    return filename

