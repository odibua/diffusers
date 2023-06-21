import argparse
import gc
import os
from pathlib import Path
from typing import List

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import upload_folder
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchmetrics import StructuralSimilarityIndexMeasure
from transformers import CLIPTokenizer
from tqdm.auto import tqdm
from torchvision import transforms


from diffusers import (
    AutoencoderKL,
    StableDiffusionInpaintClothesPipeline,
    UNet2DConditionModel
)
from diffusers.utils import check_min_version
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_clothes import _prepare_mask_and_masked_image
from diffusers.utils import randn_tensor
from train_dreambooth_inpaint_clothes import ClothesDataset, prepare_mask_and_masked_image
from .clothes_utils import get_tokenizer

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--person_clothes_file",
        type=str,
        default=None,
        required=True,
        help="A text file containing file paths to person, clothes mask, and clothes images",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a person with a shirt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help=(
            "The eval batch size"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    
    if args.person_clothes_file is None:
        raise ValueError("You must specify a csv that gives relevant paths with headers masked_images,clothes_images,mask_images")

    return args

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

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    # TODO(odibua@): Module 2: Get and prepare models 
    ##################################################################
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    clothes_unet = get_clothes_unet(unet)
    checkpoint = torch.load(args.checkpoint)
    clothes_unet.load_state_dict(checkpoint)
    clothes_unet.eval()
    ##################################################################


    # Get the pipeline that is used for inpainting
    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    pipeline = StableDiffusionInpaintClothesPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
            )
    vae.requires_grad_(False)

    # Load the tokenizer
    tokenizer = get_tokenizer(pretrained_model_name=args.pretrained_model_name_or_path)


    inference_results_df = pd.read_csv(args.person_clothes_file)
    
    eval_dataset = ClothesDataset(
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        instance_masked_images_paths=inference_results_df['masked_images'].tolist(),
        instance_clothes_images_paths=inference_results_df['clothes_images'].tolist(),
        instance_masks_images_paths=inference_results_df['mask_images'].tolist(),
    )


    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        clothes_pixel_values = [example["instance_clothes"] for example in examples]
        pixel_values = [example["instance_masked"] for example in examples]

        masks = []
        masked_images = []
        for example in examples:
            pil_image = example["PIL_instance_targets"] if "PIL_instance_targets" in example.keys() else example["PIL_instance_masked"] #TODO(odibua@): Confirm that this can just be replaced by example["PIL_instance_masked"] 
            mask = example["PIL_instance_masks"]
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        clothes_pixel_values = torch.stack(clothes_pixel_values)
        clothes_pixel_values = clothes_pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "clothes_pixel_values": clothes_pixel_values, "masks": masks, "masked_images": masked_images}
        return batch

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    
    # clothes_unet, eval_dataloader = accelerator.prepare(
    #     clothes_unet, eval_dataloader
    # )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # TODO: Module 2: put relevant models on devices
    clothes_unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    clothes_unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype = weight_dtype)
    pipeline.to(accelerator.device, torch_dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("clothes", config=vars(args))

    # Train!

    logger.info("***** Running eval *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Num batches each epoch = {len(eval_dataloader)}")

    # Only show the progress bar once on each machine.
    # TODO (odibua@): Module 4 Initialize pipeline stuff for initializing latents (Maybe make a dict or namespace?)
    ######################################################################
    num_inference_steps = 25
    device = accelerator.device
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device) #noise_scheduler.set_timesteps(num_inference_steps, device=device)
    num_images_per_prompt = 1
    strength = 1.0
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = pipeline.scheduler.timesteps[t_start * pipeline.scheduler.order :] #noise_scheduler.timesteps[t_start * noise_scheduler.order :]
    num_inference_steps = num_inference_steps - t_start
    
    batch_size = 1
    return_dict=False
    output_type="latent"
    guidance_scale = 1.1
    eta = 0.0 
    generator = None 
    negative_prompt_embeds = None 
    callback = None
    num_in = 13
    negative_prompt = None 
    
    cross_attention_kwargs = None 
    height, width = None,  None
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    height = height or unet.config.sample_size * pipeline.vae_scale_factor
    width = width or unet.config.sample_size * pipeline.vae_scale_factor
    text_encoder_lora_scale = None 
    ######################################################################

    # 5. Preprocess mask and image
    inv_transform = transforms.ToPILImage()
    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            print(f"Batch {idx}")
            image=batch["pixel_values"]
            mask_image=batch["masks"][0][0]
            for i, t in enumerate(timesteps):
                # print(i, t)
                if i == 0:
                    prompt_embeds = None 
                # with accelerator.accumulate(unet):
                    prompt_embeds = pipeline._encode_prompt(
                        args.instance_prompt,
                        device,
                        num_images_per_prompt,
                        do_classifier_free_guidance,
                        negative_prompt,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        lora_scale=text_encoder_lora_scale
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

                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    clothes_latents = vae.encode(batch["clothes_pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
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

                latent_model_input = torch.cat([clothes_latents, latents], dim=1)
                latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input

                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)#noise_scheduler.scale_model_input(latent_model_input, t)

                # if pipeline.unet.config.in_channels == num_in:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                noise_pred = clothes_unet(latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                                    # compute the previous noisy sample x_t -> x_t-1
                kwargs = {'eta': 0, 'generator': None}
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = pipeline.scheduler.step(noise_pred, t, latents, **kwargs, return_dict=False)[0] #noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                clothes_latents = pipeline.scheduler.step(noise_pred, t, clothes_latents, **kwargs, return_dict=False)[0]


            image =  vae.decode(latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]

            output_img_path = os.path.join(args.output_dir,f"output-{idx}.png")
            masked_img_path = os.path.join(args.output_dir,f"masked-{idx}.png")
            mask_img_path = os.path.join(args.output_dir,f"mask-{idx}.png")
            clothes_img_path = os.path.join(args.output_dir,f"clothes-{idx}.png")
            inv_transform(image[0] * 0.5 + 0.5).save(output_img_path)
            inv_transform(batch["pixel_values"][0]  * 0.5 + 0.5).save(masked_img_path)
            inv_transform(batch["masks"][0][0] * 0.5 + 0.5).save(mask_img_path)
            inv_transform(batch["clothes_pixel_values"][0] * 0.5 + 0.5).save(clothes_img_path)


if __name__ == "__main__":
    main()
