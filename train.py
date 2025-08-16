#!/usr/bin/env python3
"""
scFASTopic 最佳配置训练脚本
基于超参数调优得出的最佳参数配置
"""

import subprocess
import sys

def main():
    """使用优化后的最佳参数进行训练"""
    
    # 最佳配置（基于超参数调优结果）
    cmd = [
        "python", "train_fastopic.py",
        "--embedding_file", "results/cell_embedding/PBMC_12k_genept_strategy1.pkl",
        "--adata_path", "/autodl-fs/data/dataset/PBMC_12k.h5ad",
        "--dataset", "PBMC_12k_optimized",
        "--n_topics", "15",
        "--epochs", "100", 
        "--lr", "0.01",
        "--DT_alpha", "2.0",
        "--TW_alpha", "2.0",
        "--theta_temp", "2.5"
    ]
    
    print("🚀 使用最佳配置训练 scFASTopic 模型")
    print("=" * 60)
    print("配置参数:")
    print("  - 主题数量: 15")
    print("  - DT_alpha: 2.0")  
    print("  - TW_alpha: 2.0")
    print("  - theta_temp: 2.5")
    print("  - 训练轮数: 100")
    print("  - 学习率: 0.01")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 训练完成！")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()