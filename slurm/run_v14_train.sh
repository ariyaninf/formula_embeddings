#!/bin/bash
#SBATCH --job-name=v14_9vars_45cl_EF
#SBATCH --error=log/e.%x.%j
#SBATCH --output=log/o.%x.%j
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --account=scw1997

set -eu

module purge
module load anaconda
module list

source activate
source activate myenv

WORKDIR=/scratch/$USER/bert_bi-encoder
cd ${WORKDIR}

python3 v14_literal_loss.py \
  --dataset SENT_9vars_45clauses_140K_EF \
  --dataset_path dataset/EASY_dataset/EF_9vocabs \
  --batch_size 8 \
  --max_seq_length 32 \
  --bert_version bert_base_uncased \
  --bert_pooling cls \
  --sent_pooling min \
  --error_margin 2 \
  --threshold 1 \
  --patience 3
