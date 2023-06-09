
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_MASKED_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/image"
export INSTANCE_TARGET_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/target"
export INSTANCE_CLOTHES_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/clothes"
export INPAINT_MASK_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/inpaint_mask"
export INSTANCE_CLOTHES_MASK_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/clothes_mask"
export OUTPUT_DIR="/content/drive/MyDrive/clothestest-v3-6-samples"

# export INSTANCE_MASKED_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/swim_outlet/image"
# export INSTANCE_TARGET_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/swim_outlet/target"
# export INSTANCE_CLOTHES_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/swim_outlet/clothes"
# export INPAINT_MASK_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/swim_outlet/inpaint_mask"
# export OUTPUT_DIR="/content/drive/MyDrive/clothestest-v2-new"

accelerate launch train_dreambooth_inpaint_clothes.py \
  --clothes_version="v3" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_masked_dir=$INSTANCE_MASKED_DIR \
  --instance_target_dir=$INSTANCE_TARGET_DIR \
  --instance_clothes_dir=$INSTANCE_CLOTHES_DIR \
  --instance_masks_dir=$INPAINT_MASK_DIR \
  --instance_clothes_masks_dir=$INSTANCE_CLOTHES_MASK_DIR\
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a person with a shirt" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=5e-6 \
  --max_train_steps=400 \
  --add_clothes \
  --mixed_precision="fp16" \