CUDA_VISIBLE_DEVICES=2 python3 run_qa.py \
  --model_name_or_path /ssd3/damai/data/PubMedBERT-base-uncased-abs-full \
  --train_file /ssd3/damai/data/emrQA/hf_form_datasets/train_hf_form.json \
  --validation_file /ssd3/damai/data/emrQA/hf_form_datasets/dev_hf_form.json \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/