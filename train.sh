#!/bin/bash

echo "🚀 scFASTopic 最佳配置训练流程"
echo "基于超参数调优得出的最佳参数"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="

echo "🧬 Step 1: 提取GenePT Cell Embeddings"
python get_cell_emb.py \
    --input_data /autodl-fs/data/dataset/PBMC_12k.h5ad \
    --dataset_name PBMC_12k \
    --embedding_type genept_strategy1 \
    --verbose

echo "🤖 Step 2: 使用最佳参数训练FASTopic"
echo "最佳参数配置:"
echo "  - 主题数量: 15"
echo "  - DT_alpha: 2.0"  
echo "  - TW_alpha: 2.0"
echo "  - theta_temp: 2.5"
echo "  - 训练轮数: 100"
echo "  - 学习率: 0.01"

python train_fastopic.py \
    --embedding_file results/cell_embedding/Wang_cell_embeddings.pkl \
    --adata_path /root/autodl-tmp/scFastopic/data/Wang.h5ad \
    --dataset Wang_clu \
    --n_topics 100 \
    --epochs 500 \
    --lr 0.01 \
    --DT_alpha 10 \
    --TW_alpha 10 \
    --theta_temp 0.2

echo "✅ 训练完成！结果保存在 results/ 目录"


python visualization.py \
    results/cell_embedding/*.pkl \
    --adata_path /autodl-fs/data/dataset/PBMC_12k.h5ad