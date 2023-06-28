import argparse
from contextlib import nullcontext
import itertools
import gc 
import os
import random
from pathlib import Path
from typing import List

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers.utils import check_min_version
from diffusers.utils import randn_tensor
from diffusers.utils.clothes_utils import (
    get_collate_function,
    initialize_pipeline_params, 
    get_init_latents,
    get_models, 
    get_tokenizer, 
    get_end_loss_functions_dict,
    get_noise_loss_functions_dict,
    get_training_params, 
    place_on_device, 
    prepare_models_with_accelerator,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)
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

# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask

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
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_target_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training target data of a person with clothes",
    )
    parser.add_argument(
        "--instance_masked_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the data of the person with masked clothes",
    )
    parser.add_argument(
        "--instance_clothes_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of clothes to be fit to the masked out part of the body",
    )
    parser.add_argument(
        "--instance_masks_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of mask that mask out the part of the body to be fit",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--add_clothes",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--clothes_version",
        default="v1",
        type=str,
        help=(
            "Version of clothing model to be trained"
        )
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1, type=float, help="Max gradient norm.")
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--add-clothes",
        action="store_true",
        help=(
            "By default this is false, but if placed then it is true"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_target_dir is None:
        raise ValueError("You must specify a train target data directory.")
    
    if args.instance_clothes_dir is None:
        raise ValueError("You must specify a train clothes data directory.")

    if args.instance_masks_dir is None:
        raise ValueError("You must specify a train mask data directory.")

    return args


class ClothesDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_masked_root: str = None,
        instance_target_root: str = None,
        instance_clothes_root: str = None,
        instance_mask_root: str = None,
        instance_prompt: str = None,
        tokenizer: str = None,
        instance_masked_images_paths: List[str] = None,
        instance_clothes_images_paths: List[str] = None,
        instance_masks_images_paths: List[str] = None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        if instance_masked_images_paths is None: 
            self.instance_masked_root = Path(instance_masked_root)
            if not self.instance_masked_root.exists():
                raise ValueError("Instance target root doesn't exists.")

            self.instance_target_root = Path(instance_target_root)
            if not self.instance_target_root.exists():
                raise ValueError("Instance target root doesn't exists.")

            self.instance_clothes_root = Path(instance_clothes_root)
            if not self.instance_clothes_root.exists():
                raise ValueError("Instance clothes root doesn't exists.")

            self.instance_mask_root = Path(instance_mask_root)
            if not self.instance_mask_root.exists():
                raise ValueError("Instance mask root doesn't exists.")

            self.instance_target_images_path = list(Path(instance_target_root).iterdir()) * 500
            instance_clothes_images_paths = [Path(instance_clothes_root) / target_image_path.name for target_image_path in self.instance_target_images_path] 
            instance_masks_images_paths = [Path(instance_mask_root) / target_image_path.name for target_image_path in self.instance_target_images_path] 
            instance_masked_images_paths = [Path(instance_masked_root) / target_image_path.name for target_image_path in self.instance_target_images_path] 
        else:
            self.instance_target_images_path = None

        self.instance_masked_images_path = instance_masked_images_paths  
        self.instance_clothes_images_path = instance_clothes_images_paths
        self.instance_masks_path = instance_masks_images_paths
        
        self.num_instance_masked_images = len(self.instance_masked_images_path)
        self.num_instance_clothes_images = len(self.instance_clothes_images_path)
        self.num_instance_masks_images = len(self.instance_masks_path)

        if self.instance_target_images_path is not None:
            self.num_instance_target_images = len(self.instance_target_images_path)
            assert self.num_instance_clothes_images == self.num_instance_target_images, "Number of masked images for inpainting: {} is not equal to the number of targt images: {}".format(self.num_instance_masks_images, self.num_instance_clothes_images)


        assert self.num_instance_clothes_images == self.num_instance_masked_images, "Number of images in clothes directory: {} is not equal to the number of masked images: {}".format(self.num_instance_clothes_images, self.num_instance_target_images)
        assert self.num_instance_clothes_images == self.num_instance_masks_images, "Number of mask images for inpainting: {} is not equal to the number of target images: {}".format(self.num_instance_masks_images, self.num_instance_clothes_images)
        

        self.instance_prompt = instance_prompt
        self._length = self.num_instance_clothes_images

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                 transforms.CenterCrop(size) 
            ]
        )

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        index = index % self.num_instance_clothes_images
        instance_target_image = Image.open(self.instance_target_images_path[index]) if self.instance_target_images_path is not None else None
        instance_masked_image = Image.open(self.instance_masked_images_path[index])
        instance_clothes = Image.open(self.instance_clothes_images_path[index])
        instance_masks = Image.open(self.instance_masks_path[index])

        if instance_target_image is not None:
            if not instance_target_image.mode == "RGB":
                instance_target_image = instance_target_image.convert("RGB")

            instance_target_image  = self.image_transforms_resize_and_crop(instance_target_image)
            example["PIL_instance_targets"] = instance_target_image
            example["instance_targets"] = self.image_transforms(instance_target_image)

        if not instance_clothes.mode == "RGB":
            instance_clothes = instance_clothes.convert("RGB")
        if not instance_masked_image.mode == "RGB":
            instance_masked_image = instance_masked_image.convert("RGB")
        if not instance_masks.mode == "L":
            instance_masks = instance_masks.convert("L")

        instance_masked_image = self.image_transforms_resize_and_crop(instance_masked_image)
        instance_clothes = self.image_transforms_resize_and_crop(instance_clothes)
        instance_masks = self.image_transforms_resize_and_crop(instance_masks)


        example["PIL_instance_masked"] = instance_masked_image
        example["PIL_instance_masks"] = instance_masks
        example["instance_masked"] = self.image_transforms(instance_masked_image)
        example["instance_clothes"] = self.image_transforms(instance_clothes)
        example["instance_masks"] = self.image_transforms(instance_masks)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 1) Create accelerator and get device 
    inv_transform = transforms.ToPILImage()

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

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 2) Set data type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 3) Load the tokenizer
    if args.tokenizer_name:
        tokenizer = get_tokenizer(tokenizer_name=args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = get_tokenizer(pretrained_model_name=args.pretrained_model_name_or_path)

    # 4) Load models and place ones that won't be trained on the device
    ###########################################################
    if args.clothes_version in ["v1", "v2"]:
        models_dict = get_models(torch_dtype=weight_dtype, pretrained_name=args.pretrained_model_name_or_path, get_list=['text_encoder', 'vae', 'unet', 'pipeline'], clothes_version=args.clothes_version)
        text_encoder, vae, unet, pipeline = models_dict['text_encoder'], models_dict['vae'], models_dict['unet'], models_dict['pipeline']
    elif args.clothes_version == "v3":
        models_dict = get_models(torch_dtype=weight_dtype, pretrained_name=args.pretrained_model_name_or_path, get_list=['text_encoder', 'unet', 'edit_latent', 'pipeline'], clothes_version=args.clothes_version)
        text_encoder, unet, pipeline, edit_latent = models_dict['text_encoder'], models_dict['unet'], models_dict['pipeline'], models_dict['edit_latent']
    elif args.clothes_version == "v4":
        models_dict = get_models(torch_dtype=weight_dtype, pretrained_name=args.pretrained_model_name_or_path, get_list=['text_encoder', 'unet', 'pipeline'], clothes_version=args.clothes_version)
        text_encoder, unet, pipeline = models_dict['text_encoder'], models_dict['unet'], models_dict['pipeline']
    else: 
        raise NotImplementedError("Get models not impelemented for clothes version {}".format(args.clothes_version))
    
    if args.clothes_version in ["v1", "v2", "v4"]:
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
    elif args.clothes_version == "v3":
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        text_encoder.requires_grad_(False)
        # unet.requires_grad_(False)
        pipeline.unet.requires_grad_(False) 

    else: 
        raise NotImplementedError("Require grad setting to False not implemented for clothes version {}".format(args.clothes_version))
    
    if args.clothes_version in ["v1", "v2"]:
        place_on_device(device=device, weight_dtype=weight_dtype, models=[vae, pipeline, pipeline.text_encoder])
    elif args.clothes_version == "v3":
        place_on_device(device=device, weight_dtype=weight_dtype, models=[unet, pipeline.vae, pipeline.text_encoder])
        # place_on_device(device=device, weight_dtype=weight_dtype, models=[unet, vae, pipeline, pipeline.text_encoder])
        # pipeline.to('cpu', weight_dtype)
        # place_on_device(device=device, weight_dtype=weight_dtype, models=[unet, vae, pipeline.text_encoder])
    else: 
        raise NotImplementedError("Place on device not implemented for clothes version {}".format(args.clothes_version))
    ###########################################################

    # 5) Set parameters for gradient checking and initialize optimizer/params to optimize
    ###########################################################
    if args.gradient_checkpointing:
        # unet.enable_gradient_checkpointing()
        edit_latent.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Set parameters to optimize
    if args.clothes_version in ['v1', 'v2']:
        params_to_optimize = itertools.chain(unet.parameters())
    elif args.clothes_version == "v3":
         import ipdb
         ipdb.set_trace()
         params_to_optimize = itertools.chain(edit_latent.parameters())
    else:
        raise NotImplementedError("Optimization of version {} for modeling clothes is not yet implemented".format(args.clothes_version))
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    ###########################################################

    # 6) Initialize dataset and dataloader, and prepare them along with the optimizer and unet
    ###########################################################
    collate_fn = get_collate_function(tokenizer)
    train_dataset = ClothesDataset(
        instance_target_root=args.instance_target_dir,
        instance_masked_root=args.instance_masked_dir,
        instance_clothes_root=args.instance_clothes_dir,
        instance_mask_root=args.instance_masks_dir,
        instance_prompt="a person with a shirt",
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    if args.clothes_version in ["v1", "v2"]:
        unet, optimizer, train_dataloader = prepare_models_with_accelerator(accelerator, [unet, optimizer, train_dataloader])
    elif args.clothes_version == "v3": 
        edit_latent, optimizer, train_dataloader = prepare_models_with_accelerator(accelerator, [edit_latent, optimizer, train_dataloader])
    else: 
        raise NotImplementedError("Preparation of device not impelemnted for clothes_version {}".format(args.clothes_version))
    
    ###########################################################
    # Scheduler and math around the number of training steps.
    num_train_epochs, num_update_steps_per_epoch  = get_training_params(train_dataloader=train_dataloader, gradient_accumulation_steps=args.gradient_accumulation_steps, 
                        max_train_steps=args.max_train_steps, num_train_epochs=args.num_train_epochs)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("clothes", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 7) Get parameters that will be used to determine inpainting inference and to initialize latents
    num_images_per_prompt, timesteps, batch_size, strength, guidance_scale, generator, negative_prompt, negative_prompt_embeds, cross_attention_kwargs, do_classifier_free_guidance, height, width, text_encoder_lora_scale = initialize_pipeline_params(num_inference_steps=args.num_inference_steps, device=device, pipeline=pipeline, strength=1.0, unet=unet)

     # 8) Get loss functions for the noisy loop and the end of the loop
    noise_loss_function_dict = get_noise_loss_functions_dict(device=device, mse=True, mse_weight = 1e-2)
    loss_function_dict = get_end_loss_functions_dict(device=device, ssim=True, ssim_weight=4, l1 = True, l1_weight = 0.5)
    prompt="a person with a shirt"
    ##########################################################################################
    # 9) Run training loop
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # 10) Set relevant model to train
            if args.clothes_version in ['v1', 'v2']:
                unet.train()
                pipeline.unet = unet
            elif args.clothes_version == "v3":
                edit_latent.train()
                # pipeline.unet = unet
            else:
                raise NotImplementedError("Training loop not implemented for clothes model version {}".format(args.clothes_version))
            image=batch["pixel_values"]
            mask_image=batch["masks"][0]
            
            with accelerator.accumulate(edit_latent): # if args.clothes_version == "v3" else (accelerator.accumulate(unet) if args.clothes_version in ["v1", "v2"] else nullcontext()) as gs:
                noise_loss = 0
                noise_loss_init_dict = {loss_key: 0 for loss_key in noise_loss_function_dict.keys()}
                logs = { "idx": None, "epoch": None, "timestep": None}
                for i, t in enumerate(timesteps):
                    logs["idx"], logs["epoch"], logs["timestep"] = i, epoch, t
                    print(i, t)
                    if i == 0:
                        # 12) Initialize latents
                        prompt_embeds = None 
                        clothes_latents, latents, mask, masked_image_latents, prompt_embeds, target_latents = get_init_latents(
                            vae=pipeline.vae, pipeline=pipeline, unet=unet, batch=batch, image=image, mask_image=mask_image, height=height, width=width, generator=generator, clothes_version=args.clothes_version, 
                            prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, strength=strength, 
                            negative_prompt=negative_prompt, lora_scale=text_encoder_lora_scale, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, t=t, batch_size=batch_size, weight_dtype=weight_dtype
                            )
                        noise = randn_tensor(target_latents.shape, generator=generator, device=device, dtype=weight_dtype)

                    # 13) Predict noise from latents and predict previous noisy sample
                    
                    if args.clothes_version in ["v1", "v2"]:
                        latent_model_input = torch.cat([clothes_latents, latents], dim=1)
                        latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input

                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                    elif args.clothes_version == "v3":
                        mod_clothes_latents = edit_latent(clothes_latents, t).to(dtype=weight_dtype)
                        latent_model_input = latents + mod_clothes_latents
                        latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input

                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                    else:
                        raise NotImplementedError("Training loop not implemented for clothes model version {}".format(args.clothes_version))

                    noise_pred = unet(latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]
                    kwargs = {'eta': 0, 'generator': None}
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = pipeline.scheduler.step(noise_pred, t, latents, **kwargs, return_dict=False)[0] #noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    clothes_latents = pipeline.scheduler.step(noise_pred, t, clothes_latents, **kwargs, return_dict=False)[0]
                    _target_latents = pipeline.scheduler.add_noise(target_latents, noise, max((t-1, 0))) * pipeline.scheduler.init_noise_sigma

                    # 14) Calculate noisy loss
                    for loss_key in noise_loss_function_dict.keys():
                        loss_weight, loss_func = noise_loss_function_dict[loss_key]
                        noise_loss_init_dict[loss_key] = noise_loss_init_dict[loss_key] + loss_weight * loss_func(latents.float(), _target_latents.float())

                    mod_clothes_latents = edit_latent(clothes_latents, t).to(dtype=weight_dtype)
                    latents = latents + mod_clothes_latents

                logs.update({loss_key: noise_loss_init_dict[loss_key] for loss_key in noise_loss_init_dict.keys()})
                # 15) Average noisy loss
                for loss_key in noise_loss_function_dict.keys():
                    noise_loss = noise_loss + noise_loss_init_dict[loss_key]
                noise_loss = noise_loss / len(timesteps)
                logs.update({'noise_loss': noise_loss})

                if args.clothes_version in ["v1", "v2"]:
                    _image =  vae.decode(latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
                    _target = vae.decode(target_latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
                elif args.clothes_version == "v3":
                    _image =  pipeline.vae.decode(latents.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    _target = pipeline.vae.decode(target_latents.to(weight_dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                else:
                    raise NotImplementedError("Decoding not implemented for version {}".format(args.clothes_version))
                

                # 16) Calculate loss
                loss = 0
                for loss_key in loss_function_dict.keys():
                    loss_weight, loss_func = loss_function_dict[loss_key]
                    _loss = loss_weight * loss_func(latents.float(), _target_latents.float())
                    loss = loss + _loss
                    logs.update({loss_key: _loss})

                # 16) Calculate backward loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.clothes_version in ['v1', 'v2']:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                    elif args.clothes_version == 'v3': 
                        params_to_clip = (
                            itertools.chain(edit_latent.parameters())
                        )
                    else: 
                        raise NotImplementedError("Parameters to clip not implemented for clothes version {}".format(args.clothes_version))
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # progress_bar.update(1)
                    global_step += 1

                # 17) Write columns and save models at checkpointings
                if epoch == 0:
                    cols = [key for key in logs.keys()]
                    with open(f"{os.path.join(args.output_dir,'loss_log.txt')}", "a") as fObj:
                            fObj.write("{}\n".format(','.join(cols)))
                print(f"{global_step}: {logs} ")

                if global_step % args.checkpointing_steps == 0 or global_step == 1 or global_step % 2 == 0:
                    if accelerator.is_main_process:
                        vals_to_write = [str(logs[key]) for key in cols]
                        with open(f"{os.path.join(args.output_dir,'loss_logg.txt')}", "a") as fObj:
                            fObj.write("{}\n".format(','.join(vals_to_write)))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{step}-{i}-{global_step}")
                        target_save_path = os.path.join(args.output_dir,f"target-{epoch}-{step}-{i}-{global_step}.png")
                        gen_save_path = os.path.join(args.output_dir,f"get-{epoch}-{step}-{i}-{global_step}.png")
                        if global_step % args.checkpointing_steps == 0:
                            accelerator.save_state(save_path)
                        inv_transform(_target[0]).save(target_save_path)
                        inv_transform(_image[0]).save(gen_save_path)
                        logger.info(f"Saved state to {save_path}")
        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
