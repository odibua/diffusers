import argparse
import gc
import itertools
import math
import os
import random
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
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
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DClothesConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_clothes import prepare_mask_and_masked_image


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def _prepare_mask_and_masked_image(image, mask):
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        instance_masked_root,
        instance_target_root,
        instance_clothes_root,
        instance_mask_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

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
        self.instance_clothes_images_path = [Path(instance_clothes_root) / target_image_path.name for target_image_path in self.instance_target_images_path] 
        self.instance_masks_path = [Path(instance_mask_root) / target_image_path.name for target_image_path in self.instance_target_images_path] 
        self.instance_masked_images_path = [Path(instance_masked_root) / target_image_path.name for target_image_path in self.instance_target_images_path] 

        self.num_instance_target_images = len(self.instance_target_images_path)
        self.num_instance_clothes_images = len(self.instance_clothes_images_path)
        self.num_instance_masks_images = len(self.instance_masks_path)
        self.num_instance_masked_images = len(self.instance_masked_images_path)

        assert self.num_instance_clothes_images == self.num_instance_target_images, "Number of images in clothes directory: {} is not equal to the number of target images: {}".format(self.num_instance_clothes_images, self.num_instance_target_images)
        assert self.num_instance_clothes_images == self.num_instance_masks_images, "Number of mask images for inpainting: {} is not equal to the number of target images: {}".format(self.num_instance_masks_images, self.num_instance_clothes_images)
        assert self.num_instance_clothes_images == self.num_instance_masked_images, "Number of masked images for inpainting: {} is not equal to the number of targt images: {}".format(self.num_instance_masks_images, self.num_instance_clothes_images)

        self.instance_prompt = instance_prompt
        self._length = self.num_instance_target_images

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                 transforms.CenterCrop(size) #if center_crop else -{epoch}-{global_step,
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
        index = index % self.num_instance_target_images
        instance_target_image = Image.open(self.instance_target_images_path[index])
        instance_masked_image = Image.open(self.instance_masked_images_path[index])
        instance_clothes = Image.open(self.instance_clothes_images_path[index])
        instance_masks = Image.open(self.instance_masks_path[index])

        if not instance_target_image.mode == "RGB":
            instance_target_image = instance_target_image.convert("RGB")
        if not instance_clothes.mode == "RGB":
            instance_clothes = instance_clothes.convert("RGB")
        if not instance_masked_image.mode == "RGB":
            instance_masked_image = instance_masked_image.convert("RGB")
        if not instance_masks.mode == "L":
            instance_masks = instance_masks.convert("L")

        instance_target_image  = self.image_transforms_resize_and_crop(instance_target_image)
        instance_masked_image = self.image_transforms_resize_and_crop(instance_masked_image)
        instance_clothes = self.image_transforms_resize_and_crop(instance_clothes)
        instance_masks = self.image_transforms_resize_and_crop(instance_masks)

        example["PIL_instance_targets"] = instance_target_image
        example["PIL_instance_masked"] = instance_masked_image
        example["PIL_instance_masks"] = instance_masks
        example["instance_targets"] = self.image_transforms(instance_target_image)
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
    
class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
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
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms_resize_and_crop(instance_image)

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms_resize_and_crop(class_image)
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


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

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion

    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # unet = UNet2DClothesConditionModel(unet=unet)
    # unet.half()
    if args.add_clothes:
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
            # unet.half()

    # Get the pipeline that is used for inpainting
    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    pipeline = StableDiffusionInpaintClothesPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
            )
    pipeline.set_progress_bar_config(disable=True)


    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

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

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else itertools.chain(unet.parameters())
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # TODO (odibua@): Create Dataset based on clothes, as well as the associated prompt
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

    # TODO (odibua@): Check collate_fn based on new dataset
    transform = transforms.Compose([transforms.PILToTensor()])
    inv_transform = transforms.ToPILImage()
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
            mask, masked_image = _prepare_mask_and_masked_image(pil_image, mask)

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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, 
        )
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    # if not args.train_text_encoder:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype = weight_dtype)
    # pipeline.text_encoder.requries_grad_(False)
    pipeline.to(accelerator.device, torch_dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

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
    num_inference_steps = 50
    device = accelerator.device
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    num_images_per_prompt = 1
    strength = 1.0
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start * noise_scheduler.order :]
    num_inference_steps = num_inference_steps - t_start
    
    batch_size = 1
    return_dict=False
    output_type="latent"
    guidance_scale = 7.5
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

    alpha_ssim = 2
    alpha_l1 = 0.5
    thresh = -1
    # 5. Preprocess mask and image
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            pipeline.unet = unet
            prompt="a person with a shirt"
            image=batch["pixel_values"]
            mask_image=batch["masks"][0]
           
            for i, t in enumerate(timesteps):
                prompt_embeds = None 
                with accelerator.accumulate(unet):
                    prompt_embeds = pipeline._encode_prompt(
                        prompt,
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

                    mask, masked_image, init_image = prepare_mask_and_masked_image(
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
                    print(latent_timestep)
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

                    target_latents_outputs = pipeline.prepare_latents(
                                        batch_size * num_images_per_prompt,
                                        num_channels_latents,
                                        height,
                                        width,
                                        prompt_embeds.dtype,
                                        device,
                                        generator,
                                        target_latents,
                                        clothes_latents,
                                        image=init_image,
                                        timestep=latent_timestep,
                                        is_strength_max=is_strength_max,
                                        return_noise=True,
                                        return_image_latents=return_image_latents,
                                    )
                    target_latents, _, _ = target_latents_outputs

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

                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                    # if pipeline.unet.config.in_channels == num_in:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                    noise_pred = unet(latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    target_latents = noise_scheduler.step(noise_pred, t, target_latents, return_dict=False)[0]
                    # clothes_latents = noise_scheduler.step(noise_pred, t, clothes_latents, return_dict=False)[0]

                    _image =  vae.decode(latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
                    _target = vae.decode(target_latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]

                    if i > thresh:
                        loss = F.mse_loss(_image.float(), _target.float(), reduction="mean")
                    else:
                        ssim = StructuralSimilarityIndexMeasure(kernel_size=3).to(accelerator.device)
                        # TODO (odibua@): Update loss to include SSIM between final image and image with clothes. https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
                        ssim_loss = ssim(_image.float(), _target.float())
                        l1= nn.L1Loss()
                        l1_loss = l1(_image.float(), _target.float())
                        # loss = l1_loss
                        loss = alpha_ssim * ssim_loss + alpha_l1 * l1_loss

                    # TODO (odibua@): LATER: Add Perception loss  VGG (https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49)
                    # TODOD (odibua@): LATER: Test perception loss (MDF https://github.com/gfxdisp/mdf)
                    # TODO (odibua@): LATER: Test segmentation loss (pixelwise softmax)
                    # TODO (odibua@): LATER: Test loss based on joints
                    # TODO (odibua@): LATER: Update to have adverserial loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # progress_bar.update(1)
                        global_step += 1

                    if i > thresh:
                        logs = { "idx": i, "loss": loss, "epoch": epoch, "timestep": t}
                    else:
                        logs = {"ssim-loss": ssim_loss.detach().item(), "idx": i, "l1 loss":  l1_loss, "loss": loss, "epoch": epoch, "timestep": t}
                    if i == 0 and epoch == 0:
                        with open(f"{os.path.join(args.output_dir,'loss_logg.txt')}", "a") as fObj:
                                fObj.write("epoch,i,t,loss,ssim-loss,l1_loss,mse_loss\n")
                        # progress_bar.set_postfix(**logs)
                    print(f"{global_step}: {logs} ")
                    # accelerator.log(logs, step=global_step)
                    if global_step % args.checkpointing_steps == 0 or global_step == 1 or global_step % 50 == 0:
                        if accelerator.is_main_process:
                            if i > thresh:
                                with open(f"{os.path.join(args.output_dir,'loss_logg.txt')}", "a") as fObj:
                                    fObj.write(f"{epoch},{i}.{t},0,0,0,{loss}\n")
                            else:
                                with open(f"{os.path.join(args.output_dir,'loss_logg.txt')}", "a") as fObj:
                                    fObj.write(f"{epoch},{i}.{t},{loss},{ssim_loss},{l1_loss},0\n")
                            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{step}-{i}-{global_step}")
                            target_save_path = os.path.join(args.output_dir,f"target-{epoch}-{step}-{i}-{global_step}.png")
                            gen_save_path = os.path.join(args.output_dir,f"get-{epoch}-{step}-{i}-{global_step}.png")
                            if global_step % args.checkpointing_steps == 0:
                                accelerator.save_state(save_path)
                            inv_transform(_target[0]).save(target_save_path)
                            inv_transform(_image[0]).save(gen_save_path)
                            logger.info(f"Saved state to {save_path}")
                    # del latents, init_image, latent_model_input, params_to_clip, clothes_latents, _image, _target, masked_image_latents, target_latents, target_latents_outputs, latents_outputs, noise_pred, noise_pred_text, prompt_embeds, image, mask_image
                    # gc.collect()
                    # torch.cuda.empty_cache()





        ######################################################
        # for step, batch in enumerate(train_dataloader):
        #     # Skip steps until we reach the resumed step
        #     if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
        #         if step % args.gradient_accumulation_steps == 0:
        #             progress_bar.update(1)
        #         continue
        #     # latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        #     # latents = latents * vae.config.scaling_factor
        #     # latents.retain_graph=True

        #     # clothes_latents = vae.encode(batch["clothes_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        #     # clothes_latents = clothes_latents * vae.config.scaling_factor
        #     # gen_latents_time = pipeline(
        #     #     prompt="a person with a shirt", 
        #     #     image=batch["pixel_values"], 
        #     #     mask_image=batch["masks"][0],
        #     #     latents=latents,
        #     #     clothes_latents=clothes_latents,
        #     #     return_dict=False,
        #     #     output_type="latent",
        #     #     num_inference_steps=num_inference_steps
        #     #     )

        #     latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        #     latents = latents * vae.config.scaling_factor

        #     target_latents = vae.encode(batch["target_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        #     target_latents = target_latents * vae.config.scaling_factor

        #     clothes_latents = vae.encode(batch["clothes_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        #     clothes_latents = clothes_latents * vae.config.scaling_factor
        #     for idx in range(num_inference_steps):
        #         with accelerator.accumulate(unet):
        #             # Convert images to latent space
        #             # TODO (odibua@): Add cloth image as latent 
        #             latents, target_latents, clothes_latents = pipeline(
        #                     prompt="a person with a shirt", 
        #                     image=batch["pixel_values"], 
        #                     mask_image=batch["masks"][0],
        #                     latents=latents,
        #                     clothes_latents=clothes_latents,
        #                     target_latents=target_latents,
        #                     return_dict=False,
        #                     output_type="latent",
        #                     num_inference_steps=num_inference_steps,
        #             )


        #             # image = transforms.Normalize([0.5], [0.5])(image)
        #             # noise = torch.randn_like(latents)
        #             # noisy_image_latent = pipeline.scheduler.add_noise(image_latent, noise, timestep)
        #             image = vae.decode(latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
        #             target = vae.decode(target_latents.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]

        #             # TODO (odibua@): Update to have l1 loss between images
        #             if idx < thresh:
        #                 loss = F.mse_loss(image.float(), target.float(), reduction="mean")
        #             else:
        #                 ssim = StructuralSimilarityIndexMeasure().to(accelerator.device)
        #                 # TODO (odibua@): Update loss to include SSIM between final image and image with clothes. https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
        #                 ssim_loss = ssim(image.float(), target.float())
        #                 l1= nn.L1Loss()
        #                 l1_loss = l1(image.float(), target.float())
        #                 loss = ssim_loss + 0.5 * l1_loss
        #             # loss = l1_loss + ssim_loss
                    
        #             # TODO (odibua@): LATER: Add Perception loss  VGG (https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49)
        #             # TODOD (odibua@): LATER: Test perception loss (MDF https://github.com/gfxdisp/mdf)
        #             # TODO (odibua@): LATER: Test segmentation loss (pixelwise softmax)
        #             # TODO (odibua@): LATER: Test loss based on joints
        #             # TODO (odibua@): LATER: Update to have adverserial loss

        #             accelerator.backward(loss)
        #             if accelerator.sync_gradients:
        #                 params_to_clip = (
        #                     itertools.chain(unet.parameters(), text_encoder.parameters())
        #                     if args.train_text_encoder
        #                     else unet.parameters()
        #                 )
        #                 accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             latents = latents.to(accelerator.device)
        #             target_latents = target_latents.to(accelerator.device)
        #             clothes_latents = clothes_latents.to(accelerator.device)
        #             # del image_latent, image, ssim 
        #             # gc.collect()
        #             # torch.cuda.empty_cache()
                
        #         # Checks if the accelerator has performed an optimization step behind the scenes
        #         if accelerator.sync_gradients:
        #             progress_bar.update(1)
        #             global_step += 1

        #             if global_step % args.checkpointing_steps == 0 or global_step == 1 or global_step % 10 == 0:
        #                 if accelerator.is_main_process:
        #                     save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        #                     target_save_path = os.path.join(args.output_dir,f"target-{epoch}-{global_step}.png")
        #                     gen_save_path = os.path.join(args.output_dir,f"get-{epoch}-{global_step}.png")
        #                     # accelerator.save_state(save_path)
        #                     inv_transform(target[0]).save(target_save_path)
        #                     inv_transform(image[0]).save(gen_save_path)
        #                     logger.info(f"Saved state to {save_path}")

        #         if (idx < thresh):
        #             logs = {"mse-loss": loss.detach().item(), "idx": idx}
        #         else:
        #             logs = {"ssim-loss": loss.detach().item(), "idx": idx, "l1 loss":  l1_loss, "loss": loss}
        #         progress_bar.set_postfix(**logs)
        #         accelerator.log(logs, step=global_step)

        #         # if global_step >= args.max_train_steps:
                #     break

            
        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
