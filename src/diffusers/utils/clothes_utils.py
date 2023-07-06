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
    UNet2DConditionModel, 
    UNet2DConditionClothesLatentsModel,
    UNet2DConditionClothesInterpLatentsModel,
    UNet2DConditionClothesConvLatentsModel
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

def get_clothes_unet(unet: nn.Module) -> nn.Module:
    clothes_unet = UNet2DConditionModel(sample_size=unet.config.sample_size, in_channels = 13, out_channels = unet.config.out_channels,
                          center_input_sample = unet.config.center_input_sample, flip_sin_to_cos = unet.config.flip_sin_to_cos, freq_shift = unet.config.freq_shift,
                          down_block_types = unet.config.down_block_types, mid_block_type = unet.config.mid_block_type, up_block_types = unet.config.up_block_types,
                          only_cross_attention = unet.config.only_cross_attention, block_out_channels = unet.config.block_out_channels, layers_per_block = unet.config.layers_per_block,
                          downsample_padding = unet.config.downsample_padding, mid_block_scale_factor = unet.config.mid_block_scale_factor, act_fn = unet.config.act_fn,
                          norm_num_groups = unet.config.norm_num_groups, norm_eps = unet.config.norm_eps, cross_attention_dim = unet.config.cross_attention_dim,
                          encoder_hid_dim = unet.config.encoder_hid_dim, encoder_hid_dim_type = unet.config.encoder_hid_dim_type, attention_head_dim = unet.config.attention_head_dim,
                          dual_cross_attention = unet.config.dual_cross_attention, use_linear_projection = unet.config.use_linear_projection, class_embed_type = unet.config.class_embed_type,
                          addition_embed_type = unet.config.addition_embed_type, num_class_embeds = unet.config.num_class_embeds, upcast_attention = unet.config.upcast_attention,
                          resnet_time_scale_shift = unet.config.resnet_time_scale_shift, resnet_skip_time_act = unet.config.resnet_skip_time_act, resnet_out_scale_factor = unet.config.resnet_out_scale_factor,
                          time_embedding_type = unet.config.time_embedding_type, time_embedding_dim = unet.config.time_embedding_dim, time_embedding_act_fn = unet.config.time_embedding_act_fn,
                          timestep_post_act = unet.config.timestep_post_act, time_cond_proj_dim = unet.config.time_cond_proj_dim , conv_in_kernel = unet.config.conv_in_kernel,
                          conv_out_kernel = unet.config.conv_out_kernel, projection_class_embeddings_input_dim = unet.config.projection_class_embeddings_input_dim,
                          class_embeddings_concat = unet.config.class_embeddings_concat, mid_block_only_cross_attention = unet.config.mid_block_only_cross_attention,
                          cross_attention_norm = unet.config.cross_attention_norm, addition_embed_type_num_heads = unet.config.addition_embed_type_num_heads)
    return clothes_unet

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

def get_models(torch_dtype: torch.dtype, pretrained_name: str, get_list: List[str], checkpoint: str = None,
                                   clothes_version: str = 'v1') -> Dict[str, Union[CLIPTextModel, AutoencoderKL, UNet2DConditionModel, StableDiffusionInpaintClothesPipeline]]:

    models_dict = {}
    if 'text_encoder' in get_list:
        models_dict.update({'text_encoder': CLIPTextModel.from_pretrained(pretrained_name, subfolder="text_encoder")})
    else:
        models_dict.update({'text_encoder': None})
    
    if 'vae' in get_list:
        models_dict.update({'vae': AutoencoderKL.from_pretrained(pretrained_name, subfolder="vae")})
    else: 
        models_dict.update({'vae': None})

    if 'unet' in get_list:
        if clothes_version == 'v1':
            unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
            unet = add_clothes_channel_to_unet(unet)
            models_dict.update({'unet': unet})
        elif clothes_version == 'v2':
            unet = UNet2DConditionClothesLatentsModel.from_pretrained(
                                                                        pretrained_name, 
                                                                        subfolder="unet", 
                                                                        low_cpu_mem_usage=False,
                                                                        device_map=None
                                                                    )
            models_dict.update({'unet': unet})
        elif clothes_version == "v3":
            unet = UNet2DConditionClothesInterpLatentsModel.from_pretrained(
                                                                        pretrained_name, 
                                                                        subfolder="unet", 
                                                                        low_cpu_mem_usage=False,
                                                                        device_map=None
                                                                    )
            models_dict.update({'unet': unet})
        elif clothes_version == "v4":
            unet = UNet2DConditionClothesConvLatentsModel.from_pretrained(
                                                                        pretrained_name, 
                                                                        subfolder="unet", 
                                                                        low_cpu_mem_usage=False,
                                                                        device_map=None
                                                                    )
            models_dict.update({'unet': unet})
        else:
            NotImplementedError("Version {} for modeling clothes is not yet implemented".format(clothes_version))

    else: 
        models_dict.update({'unet': None})

    if 'pipeline' in get_list:
        pipeline = StableDiffusionInpaintClothesPipeline.from_pretrained(
                pretrained_name, torch_dtype=torch_dtype, safety_checker=None
            )
        pipeline.set_progress_bar_config(disable=True)
        models_dict.update({'pipeline': pipeline})
    else:
        models_dict.update({'pipeline': None})

    if 'eval_clothes_v1' in get_list:
        unet = UNet2DConditionModel.from_pretrained(pretrained_name, subfolder="unet")
        clothes_unet = get_clothes_unet(unet)
        checkpoint = torch.load(checkpoint)
        clothes_unet.load_state_dict(checkpoint)
        clothes_unet.eval()
        models_dict.update({'eval_clothes_v1': clothes_unet})
    else: 
        models_dict.update({'eval_clothes_v1': None})

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

def get_collate_function(tokenizer: CLIPTokenizer, clothes_version: str = "v1") -> Callable:

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        if 'instance_targets' in examples[0].keys():
            target_pixel_values = [example["instance_targets"] for example in examples]
        clothes_pixel_values = [example["instance_clothes"] for example in examples]
        pixel_values = [example["instance_masked"] for example in examples]

        masks, clothes_masks = [], []
        masked_images = []
        for example in examples:
            pil_image = example["PIL_instance_targets"] if "PIL_instance_targets" in example.keys() else example["PIL_instance_masked"]
            mask = example["PIL_instance_masks"]
           
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
            masks.append(mask)
            masked_images.append(masked_image)

            if clothes_version in ['v2', 'v3', 'v4']:
                clothes_mask = example["PIL_instance_clothes_masks"]
                clothes_mask, _ = prepare_mask_and_masked_image(pil_image, clothes_mask)
                clothes_masks.append(clothes_mask)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        clothes_pixel_values = torch.stack(clothes_pixel_values)
        clothes_pixel_values = clothes_pixel_values.to(memory_format=torch.contiguous_format).float()


        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        if 'instance_targets' in example.keys():
            target_pixel_values = torch.stack(target_pixel_values)
            target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
            if clothes_version == "v1":
                batch = {"input_ids": input_ids, "pixel_values": pixel_values, "target_pixel_values": target_pixel_values, "clothes_pixel_values": clothes_pixel_values, "masks": masks, "masked_images": masked_images}
            elif clothes_version in ["v2", "v3", "v4"]:
                batch = {"input_ids": input_ids, "pixel_values": pixel_values, "target_pixel_values": target_pixel_values, "clothes_pixel_values": clothes_pixel_values, "instance_clothes_masks": clothes_masks, "masks": masks, "masked_images": masked_images}
            else:
                raise NotImplementedError("Batch return not implemented for clothes version {}".format(clothes_version))

        else:
            if clothes_version == "v1":
                batch = {"input_ids": input_ids, "pixel_values": pixel_values, "clothes_pixel_values": clothes_pixel_values, "masks": masks, "masked_images": masked_images}
            elif clothes_version == "v2":
                batch = {"input_ids": input_ids, "pixel_values": pixel_values, "clothes_pixel_values": clothes_pixel_values, "clothes_pixel_values": clothes_pixel_values, "masks": masks, "masked_images": masked_images}
            else:
                raise NotImplementedError("Batch return not implemented for clothes version {}".format(clothes_version))

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

def get_noise_loss_functions_dict(device: str, mse: bool = True, mse_weight: float = None) -> Dict[str, Tuple[float, Callable]]:
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
                 return 1 - (1 + ssim_loss(x, y)) / 2
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
        negative_prompt: str, lora_scale: Callable, prompt_embeds: torch.tensor, negative_prompt_embeds: torch.tensor, t: int, batch_size: int, weight_dtype: torch.dtype, 
       clothes_version: str = 'v1', noise_clothes_latents: bool = True) -> Tuple[torch.tensor]:

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
        if clothes_version in ['v2', 'v3', 'v4']:
           clothes_mask, clothes_masked_image, clothes_init_image = _prepare_mask_and_masked_image(
                        batch['clothes_pixel_values'], batch['instance_clothes_masks'][0], height, width, return_image=True
                    ) 
        
        num_channels_latents = vae.config.latent_channels
        num_channels_unet = unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        if "target_pixel_values" in batch.keys():
            target_latents = vae.encode(batch["target_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            target_latents = target_latents * vae.config.scaling_factor
        else: 
            target_latents = None

        clothes_latents = vae.encode(batch["clothes_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        clothes_latents = clothes_latents * vae.config.scaling_factor

        if clothes_version == "v1":
            latents_outputs = pipeline.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                clothes_latents=clothes_latents,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                return_noise=True,
                return_image_latents=return_image_latents,
            )
            latents, clothes_latents, _ = latents_outputs
        elif clothes_version in ['v2', 'v3', 'v4']:
            latents_outputs = pipeline.prepare_latents(
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            height,
                            width,
                            prompt_embeds.dtype,
                            device,
                            generator,
                            latents=latents,
                            image=init_image,
                            timestep=latent_timestep,
                            is_strength_max=is_strength_max,
                            return_noise=True,
                            return_image_latents=return_image_latents,
                        )
            latents, _ = latents_outputs

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
        
        if clothes_version in ['v2', 'v3', 'v4']:
            clothes_mask, clothes_masked_image_latents = pipeline.prepare_mask_latents(
                    clothes_mask,
                    clothes_masked_image,
                    batch_size * num_images_per_prompt,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )

        if clothes_version == 'v1':
            return clothes_latents, latents, mask, masked_image_latents, prompt_embeds, target_latents
        elif clothes_version in ['v2', 'v3', 'v4']:
            return clothes_latents, latents, mask, masked_image_latents, clothes_mask, clothes_masked_image_latents, prompt_embeds, target_latents