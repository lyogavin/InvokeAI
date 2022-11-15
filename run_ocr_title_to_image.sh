#!/bin/bash


# ref: https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb
# ref: https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-how-we-made-the-text-to-pokemon-model-at-lambda

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $run_ts"

echo "START TIME: $(date)"

ROOT_DIR=/home/ubuntu/cloudfs/saved_models/stable_diffusion/finetune_redbook_ocr/

if [ ! -d ${ROOT_DIR} ];then
  mkdir -p ${ROOT_DIR}
  echo ${ROOT_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_DIR} exist!!!!!!!!!!!!!!!
fi

ROOT_LOGS_DIR=/home/ubuntu/cloudfs/saved_models/stable_diffusion/finetune_redbook_ocr/logs

if [ ! -d ${ROOT_LOGS_DIR} ];then
  mkdir -p ${ROOT_LOGS_DIR}
  echo ${ROOT_LOGS_DIR} created!!!!!!!!!!!!!!
else
  echo ${ROOT_LOGS_DIR} exist!!!!!!!!!!!!!!!
fi



NAME=finetune_ocr_title_to_image_$run_ts

python3 ./main.py -t \
    --base ./configs/stable-diffusion/v1-finetune_ocr_title_to_image.yaml \
    --actual_resume ./models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt \
    --logdir $ROOT_LOGS_DIR \
    -n $NAME \
    --gpus 0,1,2,3,4,5,6,7 \
    --data_root $ROOT_DIR \
    --init_word 'cat'