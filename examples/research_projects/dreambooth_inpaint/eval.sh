
export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export CHECKPOINT="/content/drive/MyDrive/clothestest-refactor/checkpoint-0-99-24-100/pytorch_model.bin"
export PERSON_CLOTHES_FILE="/content/drive/Othercomputers/ohi-mac/diffusers/examples/research_projects/dreambooth_inpaint/inference_list.csv"
export INSTANCE_PROMPT="a person with a shirt"
export OUTPUT_DIR="/content/drive/MyDrive/eval-clothestest-train-refactor"


accelerate launch inference_dreambooth_inpaint_clothes.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --checkpoint=$CHECKPOINT \
  --person_clothes_file=$PERSON_CLOTHES_FILE \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --mixed_precision="fp16" 