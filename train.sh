MODEL_NAME="timbrooks/instruct-pix2pix"
DATASET_ID="kyne0127/ip2p_christmas"

# export HF_HOME="/d1/hyomin/.cache3" 
# export HF_DATASETS_CACHE="/d1/hyomin/.cache3"

accelerate launch --gpu_ids=1 --num_processes=1 --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --resolution=256 \
    --random_flip \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=8000 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=1 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.0 \
    --mixed_precision=fp16 \
    --seed=42 \
    --edit_prompt_column="instruction" \
    --edited_image_column="output_image" \
    --push_to_hub