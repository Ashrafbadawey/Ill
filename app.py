import streamlit as st
import torch
from PIL import Image
import random
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
import time

# Define your constants
BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

# Initialize both pipelines
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)

main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")

image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)


# Define your Streamlit app
st.title("Illusion Diffusion HQ 🌀")

# Input widgets
control_image = st.file_uploader("Upload Input Illusion", type=["jpg", "png"])
illusion_strength = st.slider("Illusion Strength", 0.0, 5.0, 0.8, 0.01)
prompt = st.text_input("Prompt", "Medieval village scene with busy streets and castle in the distance")
negative_prompt = st.text_input("Negative Prompt", "low quality")

# Advanced Options
guidance_scale = st.slider("Guidance Scale", 0.0, 50.0, 7.5, 0.25)
sampler = st.selectbox("Sampler", list(SAMPLER_MAP.keys()))
control_start = st.slider("Start of ControlNet", 0.0, 1.0, 0.0, 0.1)
control_end = st.slider("End of ControlNet", 0.0, 1.0, 1.0, 0.1)
strength = st.slider("Strength of the Upscaler", 0.0, 1.0, 1.0, 0.1)
seed = st.slider("Seed", -1, 9999999999, -1, 1)

# Run button
if st.button("Run"):
    if control_image is None:
        st.error("Please upload an Input Illusion")
    elif not prompt:
        st.error("Prompt is required")
    else:
        start_time = time.time()
        control_image_pil = Image.open(control_image).convert('RGB')
        control_image_small = center_crop_resize(control_image_pil)
        control_image_large = center_crop_resize(control_image_pil, (1024, 1024))

        main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
        my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
        generator = torch.Generator(device="cuda").manual_seed(my_seed)

        out = main_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image_small,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(illusion_strength),
            generator=generator,
            control_guidance_start=float(control_start),
            control_guidance_end=float(control_end),
            num_inference_steps=15,
            output_type="latent"
        )
        upscaled_latents = upscale(out, "nearest-exact", 2)
        out_image = image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image_large,
            image=upscaled_latents,
            guidance_scale=float(guidance_scale),
            generator=generator,
            num_inference_steps=20,
            strength=strength,
            control_guidance_start=float(control_start),
            control_guidance_end=float(control_end),
            controlnet_conditioning_scale=float(illusion_strength)
        )
        end_time = time.time()

        st.image(out_image["images"][0], caption='Generated Image', use_column_width=True)
        st.write(f"Inference took {end_time - start_time:.2f} seconds")
