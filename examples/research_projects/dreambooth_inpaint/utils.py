import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, Generator, List, Tuple, Union

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintClothesPipeline,
    UNet2DConditionModel
)
from diffusers.utils import check_min_version
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_clothes import _prepare_mask_and_masked_image
from diffusers.utils import randn_tensor

def add_clothes_channel_to_unet(unet: nn.Module) -> nn.Module:
    with torch.no_grad():
        unet.config.in_channels = 13
        block_out_channels = unet.config.block_out_channels
        conv_in_kernel = unet.config.conv_in_kernel
        conv_in_padding = (conv_in_kernel - 1) // 2
        conv_in = nn.Conv2d(
            unet.config.in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        conv_in.weight[:, 4:, :, :] = unet.conv_in.weight
        unet.conv_in = conv_in
    return unet

def get_tokenizer(tokenizer_name: str = None, pretrained_model_name: str = None) -> CLIPTokenizer:
    assert (
            tokenizer_name is not None and pretrained_model_name is None
            ) or (
                    tokenizer_name is None and pretrained_model_name is not None
                    ), "Only specify the tokenizer_name or pretrained_model_name. Both or None are specified"
    if tokenizer_name is not None:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    else: 
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
    return tokenizer

def get_models(torch_dtype: torch.type, pretrained_name: str, ignore_list: List[str] = [], 
                                   clothes_version: str = 'v1') -> Dict[str, Union[CLIPTextModel, AutoencoderKL, UNet2DConditionModel, StableDiffusionInpaintClothesPipeline]]:
    ignore_possibilites = ['text_encoder', 'vae', 'unet', 'pipeline']
    if len(ignore_list) > 0:
        sum_ignored = [ignore_elem in ignore_possibilites for ignore_elem in ignore_possibilites]
        assert sum(sum_ignored) > 0, "Some of the elements in the ignore list are not possible. Current ignore list is {} and possibilities are {}".format(ignore_list, ignore_possibilites)

    models_dict = {}
    if 'text_encoder' not in ignore_list:
        models_dict.update({'text_encoder': CLIPTextModel.from_pretrained(pretrained_name, subfolder="text_encoder")})
    else:
        models_dict.update({'text_encoder': None})
    
    if 'vae' not in ignore_list:
        models_dict.update({'vae': AutoencoderKL.from_pretrained(pretrained_name, subfolder="vae")})
    else: 
        models_dict.update({'vae': None})

    if 'unet' not in ignore_list:
        if clothes_version == 'v1':
            unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
            unet = add_clothes_channel_to_unet(unet)
            models_dict.update({'unet': unet})
        else:
            NotImplementedError("Version {} for modeling clothes is not yet implemented".format(clothes_version))

    else: 
        models_dict.update({'unet': None})

    if 'pipeline' not in ignore_list:
        pipeline = StableDiffusionInpaintClothesPipeline.from_pretrained(
                pretrained_name, torch_dtype=torch_dtype, safety_checker=None
            )
        pipeline.set_progress_bar_config(disable=True)
        models_dict.update({'pipeline': pipeline})
    else:
        models_dict.update({'pipeline': None})

    return models_dict

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def prepare_models_with_accelerator(accelerator: Accelerator, models: List[Callable]) -> List[Callable]:
    return accelerator.prepare(*models)

def place_on_device(device: str, weight_dtype: str, models: List[Callable]) -> None: 
    [model.to(device, weight_dtype) for model in models]

def get_collate_function(tokenizer: CLIPTokenizer) -> Callable:

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        target_pixel_values = [example["instance_targets"] for example in examples]
        clothes_pixel_values = [example["instance_clothes"] for example in examples]
        pixel_values = [example["instance_masked"] for example in examples]

        masks = []
        masked_images = []
        for example in examples:
            pil_image = example["PIL_instance_targets"]
            mask = example["PIL_instance_masks"]
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        clothes_pixel_values = torch.stack(clothes_pixel_values)
        clothes_pixel_values = clothes_pixel_values.to(memory_format=torch.contiguous_format).float()

        target_pixel_values = torch.stack(target_pixel_values)
        target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "target_pixel_values": target_pixel_values, "clothes_pixel_values": clothes_pixel_values, "masks": masks, "masked_images": masked_images}
        return batch
    
    return collate_fn

def get_training_params(train_dataloader: torch.utils.data.DataLoader, gradient_accumulation_steps: int, 
                        max_train_steps: int, num_train_epochs: int) -> Tuple[int]:
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    if overrode_max_train_steps:
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    return num_train_epochs, num_update_steps_per_epoch 

def initialize_pipeline_params(num_inference_steps: int, device: str, pipeline: StableDiffusionInpaintClothesPipeline, strength: float, unet: Callable) -> Tuple[int]:
    
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device) 
    num_images_per_prompt = 1
    strength = 1.0
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = pipeline.scheduler.timesteps[t_start * pipeline.scheduler.order :] #noise_scheduler.timesteps[t_start * noise_scheduler.order :]
    num_inference_steps = num_inference_steps - t_start
    
    batch_size = 1
    guidance_scale = 1.1
    generator = None 
    negative_prompt_embeds = None 
    negative_prompt = None 
    
    cross_attention_kwargs = None 
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    height = unet.config.sample_size * pipeline.vae_scale_factor
    width = unet.config.sample_size * pipeline.vae_scale_factor
    text_encoder_lora_scale = None 

    return num_images_per_prompt, timesteps, batch_size, strength, guidance_scale, generator, negative_prompt, negative_prompt_embeds, cross_attention_kwargs, do_classifier_free_guidance, height, width, text_encoder_lora_scale

def get_noise_loss_functions_dict(mse: bool = True, mse_weight: float = None) -> Dict[str, Tuple[float, Callable]]:
    loss_dict = {}

    
    if mse:
        def mse_loss(x, y):
            return F.mse_loss(x, y, reduction="mean")
        mse_weight = mse_weight if mse_weight is not None else 1.0 
        loss_dict['mse'] = (mse_weight, mse_loss)
    
    return loss_dict

def get_end_loss_functions_dict(device: str, ssim: bool = True, ssim_weight: float = None, l1: bool = True, l1_weight: float = None) -> Dict[str, Callable]:
    loss_dict = {}

    if ssim:
        def ssim_func():
            ssim_loss = StructuralSimilarityIndexMeasure(kernel_size=5).to(device)
            def calc_loss(x, y):
                 return 1 - (1 + ssim_loss(x, x)) / 2
            return calc_loss
        
        ssim_weight = ssim_weight if ssim_weight is not None else 1.0 
        loss_dict['ssim'] = (ssim_weight, ssim_func())
    
    if l1:
        l1 = nn.L1Loss()
        l1_weight = l1_weight if l1_weight is not None else 1.0
        loss_dict['l1'] = (l1_weight, l1)

    return loss_dict

def get_init_latents(
        vae: AutoencoderKL, pipeline: StableDiffusionInpaintClothesPipeline, unet: Callable, batch: Dict[str, torch.tensor],
        image: torch.tensor, mask_image: torch.tensor, height: float, width: float, generator: Generator, 
        prompt: str, device: str, num_images_per_prompt: int, do_classifier_free_guidance: bool, strength: float, 
        negative_prompt: str, lora_scale: Callable, prompt_embeds: torch.tensor, negative_prompt_embeds: torch.tensor, t: int, batch_size: int, weight_dtype: torch.type, 
        ) -> Tuple[torch.tensor]:

        prompt_embeds = pipeline._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale
        )
        latent_timestep = t.repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        mask, masked_image, init_image = _prepare_mask_and_masked_image(
                        image, mask_image, height, width, return_image=True
                    )
        
        num_channels_latents = vae.config.latent_channels
        num_channels_unet = unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        target_latents = vae.encode(batch["target_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        target_latents = target_latents * vae.config.scaling_factor

        clothes_latents = vae.encode(batch["clothes_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        clothes_latents = clothes_latents * vae.config.scaling_factor
        latents_outputs = pipeline.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents,
                        clothes_latents,
                        image=init_image,
                        timestep=latent_timestep,
                        is_strength_max=is_strength_max,
                        return_noise=True,
                        return_image_latents=return_image_latents,
                    )

        latents, clothes_latents, _ = latents_outputs

        mask, masked_image_latents = pipeline.prepare_mask_latents(
                            mask,
                            masked_image,
                            batch_size * num_images_per_prompt,
                            height,
                            width,
                            prompt_embeds.dtype,
                            device,
                            generator,
                            do_classifier_free_guidance,
                        )

        return clothes_latents, latents, mask, masked_image_latents, target_latents