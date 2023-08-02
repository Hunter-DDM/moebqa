python3 run_headqa.py \
        --model_name_or_path model_name_or_path \
        --moe_bert \
        --routing_linear \
        --routing_dim 20 \
        --moe_arch 8-4-0 \
        --num_expert 2 \
        --routing_feat q_mean \
        --gate_type v2 \
        --balance_type acc_all \
        --balance_factor 0.001 \
        --train_file train_file \
        --validation_file validation_file \
        --output_dir output_dir \
        --logging_dir logging_dir \
        --do_train \
        --do_eval \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --seed 1234 \
        --per_gpu_eval_batch_size=2 \
        --per_device_train_batch_size=2 \
        --max_seq_length 512 \
        --overwrite_cache \
        --overwrite_output_dir \
        --save_steps 10000 \
        --logging_steps 10 \
        --warmup_ratio 0.1 \
        --evaluation_strategy epoch