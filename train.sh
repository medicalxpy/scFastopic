#!/bin/bash

dataset_name='PBMC8k'
  python train_fastopic.py \
      --embedding_file /root/autodl-tmp/scFastopic/results/cell_embedding/${dataset_name}_scvi.pkl \
      --adata_path /root/autodl-tmp/scFastopic/data/${dataset_name}.h5ad \
      --dataset ${dataset_name}_scVI \
      --n_topics 20 \
      --epochs 1000 \
      --lr 0.01 \
      --DT_alpha 1 \
      --TW_alpha 8 \
      --theta_temp 5 \



# python visualization.py \
#     /root/autodl-tmp/scFastopic/results/cell_embedding/PBMC12k_scVI.pkl \
#     --adata_path /autodl-fs/data/dataset/PBMC_12k.h5ad