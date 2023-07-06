import argparse
import os
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torchvision import transforms


from diffusers.utils import check_min_version
from train_dreambooth_inpaint_clothes import ClothesDataset
from diffusers.utils.clothes_utils import (
    get_collate_function,
    get_init_latents,
    get_models, 
    get_tokenizer, 
    initialize_pipeline_params,
    place_on_device, 
)

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
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--clothes_version",
        default="v1",
        type=str,
        help=(
            "Version of clothing model to be trained"
        )
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
    device = accelerator.device 

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = get_tokenizer(tokenizer_name=args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = get_tokenizer(pretrained_model_name=args.pretrained_model_name_or_path)
        

    weight_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    # Load models
    if args.clothes_version in ['v1', 'v2', 'v3', 'v4']:
        models_dict = get_models(torch_dtype=weight_dtype, pretrained_name=args.pretrained_model_name_or_path, checkpoint=args.checkpoint, get_list=['vae', 'pipeline', 'eval_clothes'], clothes_version=args.clothes_version)
        vae, pipeline, clothes_unet = models_dict['vae'], models_dict['pipeline'], models_dict['eval_clothes']
    else:
        raise NotImplementedError("Get model is not implemented for clothes model version {}".format(args.clothes_version))
    
    if args.clothes_version in ['v1', 'v2', 'v3', 'v4']:
        vae.requires_grad_(False)
    else:
        raise NotImplementedError("False gradient setting not impelmented for clothes version {}".format(args.clothes_version))


    collate_fn = get_collate_function(tokenizer, clothes_version=args.clothes_version)

    inference_results_df = pd.read_csv(args.person_clothes_file)
    
    eval_dataset = ClothesDataset(
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        instance_masked_images_paths=inference_results_df['masked_images'].tolist(),
        instance_clothes_images_paths=inference_results_df['clothes_images'].tolist(),
        instance_masks_images_paths=inference_results_df['mask_images'].tolist(),
        instance_clothes_masks_images_paths=inference_results_df['clothes_mask_images'].tolist(),
        clothes_version=args.clothes_version
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Place models on device
    place_on_device(device=device, weight_dtype=weight_dtype, models=[clothes_unet, vae, pipeline, pipeline.text_encoder])

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("clothes", config=vars(args))

    # Train!

    logger.info("***** Running eval *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Num batches each epoch = {len(eval_dataloader)}")

    # Only show the progress bar once on each machine.
    # Initialize pipeline stuff for initializing latents (Maybe make a dict or namespace?)
    ######################################################################
    num_images_per_prompt, timesteps, batch_size, strength, guidance_scale, generator, negative_prompt, negative_prompt_embeds, cross_attention_kwargs, do_classifier_free_guidance, height, width, text_encoder_lora_scale = initialize_pipeline_params(num_inference_steps=args.num_inference_steps, device=device, pipeline=pipeline, strength=1.0, unet=clothes_unet)
    # Do inference
    prompt="a person with a shirt"
    inv_transform = transforms.ToPILImage()
    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            print(f"Batch {idx}")
            if args.clothes_version in ['v1', 'v2', 'v3', 'v4']:
                image=batch["pixel_values"]
                mask_image=batch["masks"][0][0]
                batch["pixel_values"] = batch["pixel_values"].to(device, dtype=weight_dtype)
                batch["masks"] = batch["masks"].to(device, dtype=weight_dtype)
                batch["clothes_pixel_values"] = batch["clothes_pixel_values"].to(device, dtype=weight_dtype)
                batch["instance_clothes_masks"][0] = batch["instance_clothes_masks"][0].to(device, dtype=weight_dtype)
            for i, t in enumerate(timesteps):
                # print(i, t)
                if i == 0:
                    prompt_embeds = None 
                    if args.clothes_version == 'v1':
                        clothes_latents, latents, mask, masked_image_latents, prompt_embeds, _ = get_init_latents(
                            vae=vae, pipeline=pipeline, unet=clothes_unet, batch=batch, image=image, mask_image=mask_image, height=height, width=width, generator=generator, 
                            prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, strength=strength, 
                            negative_prompt=negative_prompt, lora_scale=text_encoder_lora_scale, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, t=t, batch_size=batch_size, weight_dtype=weight_dtype, 
                            )
                    elif args.clothes_version in ['v2', 'v3', 'v4']:
                        clothes_latents, latents, mask, masked_image_latents, _, _, prompt_embeds, _ = get_init_latents(
                            vae=vae, pipeline=pipeline, unet=clothes_unet, batch=batch, image=image, mask_image=mask_image, height=height, width=width, generator=generator, 
                            prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, strength=strength, 
                            negative_prompt=negative_prompt, lora_scale=text_encoder_lora_scale, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, t=t, batch_size=batch_size, weight_dtype=weight_dtype, 
                            clothes_version=args.clothes_version)
                    else: 
                        raise NotImplementedError("Get initial latents not implemented for clothes version {}".format(args.clothes_version))


                if args.clothes_version == 'v1':
                    latent_model_input = torch.cat([clothes_latents, latents], dim=1)
                    latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input

                    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                    # if pipeline.unet.config.in_channels == num_in:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                    noise_pred = clothes_unet(latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]
                elif args.clothes_version in ['v2', 'v3', 'v4']:
                    latent_model_input = torch.cat([latents], dim=1)
                    latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
                    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                    noise_pred = clothes_unet(latent_model_input,
                            t,
                            clothes_latent=torch.cat([clothes_latents, clothes_latents]),
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
                if args.clothes_version == 'v1':
                    clothes_latents = pipeline.scheduler.step(noise_pred, t, clothes_latents, **kwargs, return_dict=False)[0]
                elif args.clothes_version in ['v2', 'v3', 'v4']:
                    pass
                else:
                    raise NotImplementedError("Clothes Latent processing not implemented for clothes model version {}".format(args.clothes_version))


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
