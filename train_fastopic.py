#!/usr/bin/env python3
"""
FASTopic训练脚本
使用预提取的cell embeddings和原始adata训练scFASTopic模型
"""
import argparse
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
import pickle
import time
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc


# 使用自定义的utils函数
def save_matrices(matrices, dataset_name, n_topics, output_dir):
    """保存矩阵到指定的子目录"""
    base_output_dir = Path(output_dir)
    
    # 定义矩阵类型到子目录的映射
    matrix_subdirs = {
        'cell_topic_matrix': 'cell_topic',
        'topic_gene_matrix': 'topic_gene', 
        'gene_embeddings': 'gene_embedding',
        'topic_embeddings': 'topic_embedding'
    }
    
    saved_files = []
    for matrix_name, matrix in matrices.items():
        # 创建对应的子目录
        subdir = matrix_subdirs.get(matrix_name, matrix_name)
        matrix_output_dir = base_output_dir / subdir
        matrix_output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{dataset_name}_{matrix_name}_{n_topics}.pkl"
        filepath = matrix_output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(matrix, f)
        
        saved_files.append(str(filepath))
        print(f"💾 Saved {matrix_name}: {filepath}")
    
    return saved_files
def validate_matrices(matrices):
    """验证矩阵形状和内容"""
    try:
        for name, matrix in matrices.items():
            if matrix is None:
                print(f"⚠️ Warning: {name} is None")
                return False
            if matrix.size == 0:
                print(f"⚠️ Warning: {name} is empty")
                return False
        return True
    except Exception as e:
        print(f"❌ Matrix validation error: {e}")
        return False
from dataclasses import dataclass


@dataclass
class FastopicConfig:
    embedding_file: Optional[str] = None
    adata_path: Optional[str] = None
    dataset: str = "PBMC"
    output_dir: str = "results"
    n_topics: int = 20
    epochs: int = 100
    learning_rate: float = 0.01
    DT_alpha: float = 1.0
    TW_alpha: float = 1.0
    theta_temp: float = 2.0
    verbose: bool = True
    seed: int = 42
    filter_genept: bool = True
    patience: int = 10
    min_delta: float = 1e-4


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train scFASTopic with pre-extracted cell embeddings')
    
    # 输入文件参数
    parser.add_argument('--embedding_file', type=str, required=True,
                       help='Path to cell embeddings pkl file')
    parser.add_argument('--adata_path', type=str, required=True,
                       help='Path to original adata file (.h5ad)')
    parser.add_argument('--dataset', type=str, default='PBMC',
                       help='Dataset name')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    # 模型参数
    parser.add_argument('--n_topics', type=int, default=20,
                       help='Number of topics')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    
    # FASTopic超参数
    parser.add_argument('--DT_alpha', type=float, default=1.0,
                       help='Dirichlet-tree alpha parameter')
    parser.add_argument('--TW_alpha', type=float, default=1.0,
                       help='Topic-word alpha parameter')
    parser.add_argument('--theta_temp', type=float, default=2.0,
                       help='Temperature parameter')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--no_genept_filter', action='store_true',
                       help='Disable GenePT gene filtering')
    
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> FastopicConfig:
    return FastopicConfig(
        embedding_file=args.embedding_file,
        adata_path=args.adata_path,
        dataset=args.dataset,
        output_dir=args.output_dir,
        n_topics=args.n_topics,
        epochs=args.epochs,
        learning_rate=args.lr,
        DT_alpha=args.DT_alpha,
        TW_alpha=args.TW_alpha,
        theta_temp=args.theta_temp,
        verbose=not args.quiet,
        seed=args.seed,
        filter_genept=not args.no_genept_filter,
        patience=args.patience,
    )


def load_genept_genes():
    """加载GenePT基因列表"""
    try:
        genept_path = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
        with open(genept_path, 'rb') as f:
            genept_dict = pickle.load(f)
        return set(genept_dict.keys())
    except Exception as e:
        print(f"⚠️ 无法加载GenePT基因列表: {e}")
        return None

def preprocess_adata(adata_path: str, verbose: bool = False, filter_genept: bool = True):
    """
    从adata中提取计数矩阵并进行预处理
    
    Args:
        adata_path: 单细胞数据路径
        verbose: 是否详细输出
        filter_genept: 是否过滤到GenePT共有基因
        
    Returns:
        expression_matrix: 预处理后的表达矩阵 (cells x genes)
        gene_names: 基因名称列表
    """
    if verbose:
        print(f"📁 Loading adata: {adata_path}")
    
    # 加载数据
    adata = sc.read_h5ad(adata_path)
    
    if verbose:
        print(f"原始数据维度: {adata.shape}")
    
    # 简单过滤
    # 过滤低质量细胞 (表达基因数 < 200)
    sc.pp.filter_cells(adata, min_genes=200)
    
    # 过滤低表达基因 (在 < 3个细胞中表达)
    sc.pp.filter_genes(adata, min_cells=3)
    
    if verbose:
        print(f"过滤后数据维度: {adata.shape}")
    
    # GenePT基因过滤
    if filter_genept:
        genept_genes = load_genept_genes()
        if genept_genes is not None:
            # 找到与GenePT共有的基因
            current_genes = set(adata.var_names)
            common_genes = current_genes.intersection(genept_genes)
            
            if len(common_genes) > 0:
                # 过滤到共有基因
                adata = adata[:, list(common_genes)]
                if verbose:
                    print(f"🧬 GenePT基因过滤: {len(common_genes)}/{len(current_genes)} 基因保留")
            else:
                if verbose:
                    print("⚠️ 没有与GenePT共有的基因，跳过基因过滤")
    
    if verbose:
        print(f"最终数据维度: {adata.shape}")
    
    # 标准化到每个细胞总计数为1e4
    sc.pp.normalize_total(adata, target_sum=1)
    
    # log1p变换
    sc.pp.log1p(adata)
    
    # 获取处理后的矩阵
    if hasattr(adata.X, 'toarray'):
        expression_matrix = adata.X.toarray()
    else:
        expression_matrix = adata.X
    
    gene_names = adata.var_names.tolist()
    
    if verbose:
        print(f"✅ 预处理完成: {expression_matrix.shape}")
        print(f"✅ 基因数量: {len(gene_names)}")
    
    return expression_matrix, gene_names


def load_embeddings_and_expression(embedding_file: str, adata_path: str, verbose: bool = False, filter_genept: bool = True):
    """
    加载cell embeddings和预处理后的表达矩阵
    
    Args:
        embedding_file: cell embeddings文件路径
        adata_path: 原始adata路径
        verbose: 是否详细输出
        filter_genept: 是否过滤到GenePT共有基因
        
    Returns:
        cell_embeddings: Cell embeddings矩阵
        expression_matrix: 预处理后的表达矩阵
        gene_names: 基因名称列表
    """
    if verbose:
        print("📥 Loading embeddings and preprocessing expression data")
        print("="*60)
    
    # 加载cell embeddings
    if verbose:
        print(f"📁 Loading cell embeddings: {embedding_file}")
    
    with open(embedding_file, 'rb') as f:
        cell_embeddings = pickle.load(f)
    
    if verbose:
        print(f"✅ Cell embeddings: {cell_embeddings.shape}")
    
    # 预处理adata
    expression_matrix, gene_names = preprocess_adata(adata_path, verbose, filter_genept)
    
    # 确保细胞数量匹配
    n_cells_emb = cell_embeddings.shape[0]
    n_cells_exp = expression_matrix.shape[0]
    
    if n_cells_emb != n_cells_exp:
        min_cells = min(n_cells_emb, n_cells_exp)
        if verbose:
            print(f"⚠️ 细胞数量不匹配 (embedding: {n_cells_emb}, expression: {n_cells_exp})")
            print(f"使用前 {min_cells} 个细胞")
        
        cell_embeddings = cell_embeddings[:min_cells]
        expression_matrix = expression_matrix[:min_cells]
    
    return cell_embeddings, expression_matrix, gene_names



def train_fastopic_model(
    cell_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    gene_names: List[str],
    config: FastopicConfig,
    verbose: bool = False,
):
    """
    训练scFASTopic模型
    
    Args:
        cell_embeddings: Cell embeddings矩阵
        expression_matrix: 预处理后的表达矩阵
        gene_names: 基因名称列表
        config: 配置参数
        verbose: 是否详细输出
        
    Returns:
        results: 训练结果字典
        training_time: 训练时间
    """
    if verbose:
        print("\n🤖 Training scFASTopic model")
        print("="*60)
    
    # 使用真正的FASTopic
    from fastopic import FASTopic
    
    model = FASTopic(
        num_topics=config.n_topics,
        device="cuda" if torch.cuda.is_available() else "cpu",
        DT_alpha=config.DT_alpha,
        TW_alpha=config.TW_alpha,
        theta_temp=config.theta_temp,
        verbose=verbose,
        log_interval=10,
        low_memory=False,
        low_memory_batch_size=8000
    )
    
    # 训练模型
    start_time = time.time()
    if verbose:
        print(f"🔥 Training with {config.n_topics} topics for {config.epochs} epochs...")
    
    # 将表达矩阵转换为稀疏矩阵作为BOW输入
    expression_bow = sp.csr_matrix(expression_matrix)
    
    # 标准训练
    top_words, train_theta = model.fit_transform_sc(
        cell_embeddings=cell_embeddings,
        gene_names=gene_names,
        expression_bow=expression_bow,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
        min_delta=config.min_delta
    )

    training_time = time.time() - start_time

    # 获取结果矩阵
    beta = model.get_beta()  # topic-gene matrix
    theta = train_theta      # cell-topic matrix
    
    # 计算评估指标
    from scipy.stats import entropy
    
    # Shannon熵（衡量topic分布的均匀性）
    topic_weights = theta.mean(axis=0)
    shannon_entropy = entropy(topic_weights + 1e-12, base=2)
    
    # 有效topic数量
    effective_topics = 2**shannon_entropy
    
    # 主导topic占比
    max_topic_weight = topic_weights.max()
    dominant_topic_ratio = max_topic_weight * 100
    
    results = {
        'beta': beta,
        'theta': theta,
        'top_words': top_words,
        'shannon_entropy': shannon_entropy,
        'effective_topics': effective_topics,
        'dominant_topic_ratio': dominant_topic_ratio,
    }

    if verbose:
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📊 Shannon Entropy: {shannon_entropy:.3f}")
        print(f"🎯 Effective Topics: {effective_topics:.1f}")
        print(f"👑 Dominant Topic: {dominant_topic_ratio:.1f}%")

    return model, results, training_time


def save_all_matrices(
    model,
    results: dict,
    config: FastopicConfig,
    verbose: bool = False,
):
    """保存所有矩阵"""
    if verbose:
        print("\n💾 Saving matrices")
        print("="*60)
    
    # 准备需要保存的矩阵（仅保存用户需要的4种）
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    matrices = {
        'cell_topic_matrix': results['theta'],
        'topic_gene_matrix': results['beta'],
        'gene_embeddings': _to_numpy(model.word_embeddings),
        'topic_embeddings': _to_numpy(model.topic_embeddings),
    }
    
    # 验证矩阵
    if not validate_matrices(matrices):
        raise ValueError("Matrix validation failed")
    
    # 保存矩阵
    saved_files = save_matrices(
        matrices=matrices,
        dataset_name=config.dataset,
        n_topics=config.n_topics,
        output_dir=config.output_dir
    )

    return saved_files




def main():
    """主函数"""
    print("🚀 scFASTopic Training Pipeline")
    print("="*80)
    
    # 解析参数
    args = parse_args()
    
    config = config_from_args(args)

    # 设置随机种子
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if config.verbose:
        print(f"📊 Configuration:")
        print(f"  Dataset: {config.dataset}")
        print(f"  Topics: {config.n_topics}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Early stopping patience: {config.patience}")
        print(f"  GenePT gene filtering: {config.filter_genept}")
        print(f"  Embedding file: {config.embedding_file}")
        print(f"  Adata file: {config.adata_path}")
    
    try:
        # Step 1: 加载embeddings和预处理表达数据
        cell_embeddings, expression_matrix, gene_names = load_embeddings_and_expression(
            config.embedding_file, config.adata_path, config.verbose, config.filter_genept
        )
        
        # Step 2: 训练模型
        model, results, training_time = train_fastopic_model(
            cell_embeddings, expression_matrix, gene_names, config, config.verbose
        )

        # Step 3: 保存矩阵
        saved_files = save_all_matrices(
            model, results, config, config.verbose
        )

        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}/")
        
        print(f"\n🎯 Final Results:")
        print(f"  Shannon Entropy: {results['shannon_entropy']:.3f}")
        print(f"  Effective Topics: {results['effective_topics']:.1f}")
        print(f"  Training Time: {training_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
