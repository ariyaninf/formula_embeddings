#!/bin/bash

C:/Users/nurul/miniconda3/envs/myenv/python.exe v6_order_emb.py \
  --dataset SENT_9vars_45clauses_140K_EF \
  --dataset_path dataset/EASY_dataset/EF_9vocabs \
  --batch_size 2 \
  --max_seq_length 32 \
  --bert_version bert_base_uncased \
  --bert_pooling cls \
  --sent_pooling min \
  --error_margin 2 \
  --threshold 1 \
  --mode_train RP \
  --model_path output/v6_order_embeddings \
  --patience 3
