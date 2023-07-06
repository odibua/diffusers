
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export CHECKPOINT="/content/drive/MyDrive/clothestest-v3-6-samples/checkpoint-0-59-24-300-4/pytorch_model.bin"
export PERSON_CLOTHES_FILE="/content/drive/Othercomputers/ohi-mac/diffusers/examples/research_projects/dreambooth_inpaint/inference_list_v3_6_samps.csv"
export INSTANCE_PROMPT="a person with a shirt"
export OUTPUT_DIR="/content/drive/MyDrive/eval-clothestest-v3"


accelerate launch inference_dreambooth_inpaint_clothes.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --checkpoint=$CHECKPOINT \
  --person_clothes_file=$PERSON_CLOTHES_FILE \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --mixed_precision="fp16" \
  --clothes_version="v3"