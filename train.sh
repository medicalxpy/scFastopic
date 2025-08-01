#!/bin/bash

echo "🚀 Step 1: 提取Cell Embeddings"
python get_cell_emb.py \
    --input_data /autodl-fs/data/dataset/PBMC.h5ad \
    --dataset_name PBMC_geneformer \
    --use_geneformer \
    --model_path /root/autodl-tmp/Geneformer/Geneformer-V2-104M \
    --forward_batch_size 64 \
    --nproc 1 \
    --verbose

echo "🚀 Step 2: 训练FASTopic"
python train_fastopic.py \
    --embedding_file results/PBMC_geneformer_cell_embeddings.pkl \
    --genes_file results/PBMC_geneformer_selected_genes.pkl \
    --dataset PBMC_geneformer \
    --n_topics 20 \
    --epochs 200