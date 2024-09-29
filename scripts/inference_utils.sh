# Run this script from the root directory as
# bash scripts/inference_utils.sh

export HF_HOME=../cache/
# model_name="llava-hf/llava-v1.6-mistral-7b-hf"
# model_name="Lin-Chen/ShareGPT4V-7B"
# model_name="renjiepi/G-LLaVA-7B"
# model_name="llava-hf/llava-1.5-13b-hf"
# model_name="llava-hf/llava-v1.6-vicuna-13b-hf"
# model_name="Lin-Chen/ShareGPT4V-13B"
# model_name="Qwen/Qwen2-VL-2B-Instruct"
# model_name="Qwen/Qwen2-VL-7B-Instruct"
model_name="renjiepi/G-LLaVA-13B"
# model_name="llava-hf/llava-v1.6-34b-hf"

# This script aims to carry out manual distributed data inference
# using multi GPUs by splitting the data manually across GPUs
# based on operator preference.
#  export CUDA_VISIBLE_DEVICES=4
#  python scripts/inference_utils.py \
#  -dp "./datastore/" \
#  -mn "$model_name" \
#  -drs 0 \
#  -dre 30 \
#  -isq True \
#  --mode "inference"

export CUDA_VISIBLE_DEVICES=5
python scripts/inference_utils.py \
-dp "./datastore/" \
-mn "$model_name" \
-drs 30 \
-dre 60 \
--mode "inference"

# export CUDA_VISIBLE_DEVICES=6
# python scripts/inference_utils.py \
# -dp "./datastore/" \
# -mn "$model_name" \
# -drs 60 \
# -dre 90 \
# --mode "inference"

# export CUDA_VISIBLE_DEVICES=7
# python scripts/inference_utils.py \
# -dp "./datastore/" \
# -mn "$model_name" \
# -drs 90 \
# -dre 131 \
# --mode "inference"
