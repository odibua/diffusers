
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_TARGET_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/image"
export INSTANCE_CLOTHES_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/clothes"
export INPAINT_MASK_DIR="/content/drive/Othercomputers/ohi-mac/diffusers/test_data/inpaint_mask"
export OUTPUT_DIR="test_train_inpaint"


accelerate launch train_dreambooth_inpaint_clothes.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_target_dir=$INSTANCE_TARGET_DIR \
  --instance_clothes_dir=$INSTANCE_CLOTHES_DIR \
  --instance_masks_dir=$INPAINT_MASK_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a person with a shirt" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --max_train_steps=800 \
  --add_clothes
```