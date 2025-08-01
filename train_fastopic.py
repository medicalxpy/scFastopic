#!/usr/bin/env python3
"""
FASTopic训练脚本
专门用于加载已有的cell embeddings训练scFASTopic模型
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
import pickle
import json
import time
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 可视化相关导入
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import umap


# 使用自定义的utils函数
def save_matrices(matrices, dataset_name, n_topics, output_dir):
    """保存矩阵到指定目录"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    saved_files = []
    for matrix_name, matrix in matrices.items():
        filename = f"{dataset_name}_{matrix_name}_{n_topics}.pkl"
        filepath = output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(matrix, f)
        
        saved_files.append(str(filepath))
        print(f"💾 Saved {matrix_name}: {filepath}")
    
    return saved_files

def calculate_topic_metrics(cell_topic_matrix):
    """计算topic质量指标"""
    from scipy.stats import entropy
    
    # 处理NaN值
    if np.isnan(cell_topic_matrix).any():
        print("⚠️ Warning: Found NaN values in cell-topic matrix for metrics calculation")
        cell_topic_matrix = np.nan_to_num(cell_topic_matrix, nan=0.0)
    
    topic_weights = cell_topic_matrix.mean(axis=0)
    shannon_entropy = entropy(topic_weights + 1e-12, base=2)
    effective_topics = 2**shannon_entropy
    dominant_topic_ratio = topic_weights.max() * 100
    
    return {
        'shannon_entropy': shannon_entropy,
        'effective_topics': effective_topics,
        'dominant_topic_ratio': dominant_topic_ratio,
        'topic_weights': topic_weights
    }

def create_summary_report(matrices, metrics, dataset_name, n_topics):
    """创建摘要报告"""
    report = f"""# scFASTopic Training Report

## Dataset Information
- Dataset: {dataset_name}
- Number of topics: {n_topics}
- Number of cells: {matrices['cell_embeddings'].shape[0]:,}
- Number of genes: {matrices['topic_gene_matrix'].shape[1]:,}

## Model Performance
- Shannon Entropy: {metrics['shannon_entropy']:.3f}
- Effective Topics: {metrics['effective_topics']:.1f}
- Dominant Topic Ratio: {metrics['dominant_topic_ratio']:.1f}%

## Matrix Shapes
- Cell embeddings: {matrices['cell_embeddings'].shape}
- Cell-topic matrix: {matrices['cell_topic_matrix'].shape}
- Topic-gene matrix: {matrices['topic_gene_matrix'].shape}
"""
    return report

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


class FastopicConfig:
    """FASTopic训练配置类"""
    
    def __init__(self):
        # 输入输出
        self.embedding_file = None
        self.genes_file = None
        self.dataset = 'PBMC'
        self.output_dir = 'results'
        self.visualization_dir = 'visualization'
        
        # 模型参数
        self.n_topics = 20
        self.epochs = 100
        self.learning_rate = 0.01
        
        # FASTopic超参数
        self.DT_alpha = 1.0
        self.TW_alpha = 1.0
        self.theta_temp = 2.0
        
        # 其他参数
        self.verbose = True
        self.seed = 42


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train scFASTopic with pre-extracted cell embeddings')
    
    # 输入文件参数
    parser.add_argument('--embedding_file', type=str, required=True,
                       help='Path to cell embeddings pkl file')
    parser.add_argument('--genes_file', type=str, required=True,
                       help='Path to selected genes pkl file')
    parser.add_argument('--dataset', type=str, default='PBMC',
                       help='Dataset name')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--viz_dir', type=str, default='visualization',
                       help='Visualization directory')
    
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
    
    return parser.parse_args()


def load_embeddings_and_genes(embedding_file: str, genes_file: str, bow_file: str = None, verbose: bool = False):
    """
    加载cell embeddings、基因列表和BOW矩阵
    
    Args:
        embedding_file: cell embeddings文件路径
        genes_file: 基因列表文件路径
        bow_file: 基因表达BOW矩阵文件路径
        verbose: 是否详细输出
        
    Returns:
        cell_embeddings: Cell embeddings矩阵
        selected_genes: 基因列表
        expression_bow: 基因表达BOW矩阵
    """
    if verbose:
        print("📥 Loading pre-extracted embeddings and genes")
        print("="*60)
    
    # 加载cell embeddings
    if verbose:
        print(f"📁 Loading cell embeddings: {embedding_file}")
    
    with open(embedding_file, 'rb') as f:
        cell_embeddings = pickle.load(f)
    
    # 加载基因列表
    if verbose:
        print(f"📁 Loading genes: {genes_file}")
    
    with open(genes_file, 'rb') as f:
        selected_genes = pickle.load(f)
    
    # 加载基因表达BOW矩阵（如果提供）
    expression_bow = None
    if bow_file and os.path.exists(bow_file):
        if verbose:
            print(f"📁 Loading expression BOW: {bow_file}")
        with open(bow_file, 'rb') as f:
            expression_bow = pickle.load(f)
        if verbose:
            print(f"✅ Expression BOW: {expression_bow.shape}, 稀疏度: {1 - expression_bow.nnz / (expression_bow.shape[0] * expression_bow.shape[1]):.3f}")
    
    if verbose:
        print(f"✅ Cell embeddings: {cell_embeddings.shape}")
        print(f"✅ Selected genes: {len(selected_genes)}")
    
    return cell_embeddings, selected_genes, expression_bow


def train_fastopic_model(cell_embeddings: np.ndarray, 
                        selected_genes: List[str],
                        expression_bow,
                        config: FastopicConfig,
                        verbose: bool = False):
    """
    训练scFASTopic模型
    
    Args:
        cell_embeddings: Cell embeddings矩阵
        selected_genes: 基因列表
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
        low_memory=True,                    # 开启低内存模式
        low_memory_batch_size=4000          # 设置批次大小为1000
    )
    
    # 训练模型
    start_time = time.time()
    if verbose:
        print(f"🔥 Training with {config.n_topics} topics for {config.epochs} epochs...")
    
    top_words, train_theta = model.fit_transform_sc(
        cell_embeddings=cell_embeddings,
        gene_names=selected_genes,
        expression_bow=expression_bow,
        epochs=config.epochs,
        learning_rate=config.learning_rate
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
        'model': model
    }
    
    if verbose:
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📊 Shannon Entropy: {shannon_entropy:.3f}")
        print(f"🎯 Effective Topics: {effective_topics:.1f}")
        print(f"👑 Dominant Topic: {dominant_topic_ratio:.1f}%")
    
    return results, training_time


def save_all_matrices(results: dict,
                     cell_embeddings: np.ndarray,
                     config: FastopicConfig,
                     verbose: bool = False):
    """保存所有矩阵"""
    if verbose:
        print("\n💾 Saving matrices")
        print("="*60)
    
    # 准备所有矩阵
    matrices = {
        'cell_embeddings': cell_embeddings,
        'cell_topic_matrix': results['theta'],  # FASTopic返回的cell-topic矩阵
        'topic_gene_matrix': results['beta'],   # FASTopic返回的topic-gene矩阵
        'gene_embeddings': results['model'].word_embeddings,  # 从模型获取基因嵌入
        'topic_embeddings': results['model'].topic_embeddings  # 从模型获取主题嵌入
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
    
    return saved_files, matrices


def create_visualizations(results: dict, config: FastopicConfig, verbose: bool = False):
    """创建可视化"""
    if verbose:
        print("\n🎨 Creating visualizations")
        print("="*60)
    
    try:
        # 创建输出目录
        viz_dir = Path(config.visualization_dir)
        viz_dir.mkdir(exist_ok=True)
        
        # 获取cell-topic矩阵
        if verbose:
            print("📐 Creating cell-topic visualizations with cell type annotation...")
        cell_topic_matrix = results['theta']
        
        # 检查并处理NaN值
        if np.isnan(cell_topic_matrix).any():
            if verbose:
                print("⚠️ Warning: Found NaN values in cell-topic matrix, replacing with zeros")
            cell_topic_matrix = np.nan_to_num(cell_topic_matrix, nan=0.0)
        
        # 加载细胞类型信息
        try:
            adata = sc.read_h5ad('/autodl-fs/data/dataset/PBMC.h5ad')
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            cell_types = adata.obs['scType_celltype'].values[:cell_topic_matrix.shape[0]]
            cell_type_simple = adata.obs['cell_type'].values[:cell_topic_matrix.shape[0]]
            
            if verbose:
                print(f"✅ Loaded cell types for {len(cell_types)} cells")
        except:
            cell_types = None
            cell_type_simple = None
            if verbose:
                print("⚠️ Could not load cell type information, using default colors")
        
        # UMAP降维cell-topic矩阵
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(cell_topic_matrix)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Cell-Topic Analysis ({config.dataset}-{config.n_topics} topics)', 
                    fontsize=16, fontweight='bold')
        
        # 图1: Cell-Topic UMAP (按简化cell type染色)
        ax1 = axes[0]
        if cell_type_simple is not None:
            unique_simple_types = pd.Series(cell_type_simple).unique()
            colors_simple = plt.cm.tab10(np.linspace(0, 1, len(unique_simple_types)))
            color_map_simple = dict(zip(unique_simple_types, colors_simple))
            
            for cell_type in unique_simple_types:
                mask = cell_type_simple == cell_type
                ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[color_map_simple[cell_type]], label=cell_type, alpha=0.6, s=1)
            
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.set_title('Cell-Topic UMAP (by Cell Type)', fontsize=14, fontweight='bold')
        else:
            ax1.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1, c='steelblue')
            ax1.set_title('Cell-Topic UMAP', fontsize=14, fontweight='bold')
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # 图2: 每个细胞类型的平均topic分布热图
        ax2 = axes[1]
        if cell_type_simple is not None:
            # 计算每个细胞类型的平均topic分布
            cell_type_topic_means = []
            cell_type_labels = []
            
            for cell_type in unique_simple_types[:8]:  # 显示前8个最常见的类型
                mask = cell_type_simple == cell_type
                if np.sum(mask) > 100:  # 只包含足够细胞数的类型
                    mean_topics = cell_topic_matrix[mask].mean(axis=0)
                    cell_type_topic_means.append(mean_topics)
                    cell_type_labels.append(cell_type.replace(', ', '\n'))
            
            if cell_type_topic_means:
                cell_type_topic_means = np.array(cell_type_topic_means)
                
                # 创建热图
                sns.heatmap(cell_type_topic_means, 
                           xticklabels=[f'T{i}' for i in range(cell_topic_matrix.shape[1])],
                           yticklabels=cell_type_labels,
                           cmap='YlOrRd', ax=ax2, 
                           cbar_kws={'label': 'Average Topic Weight'})
                ax2.set_title('Average Topic Distribution by Cell Type', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Topics')
                ax2.set_ylabel('Cell Types')
        else:
            # 如果没有细胞类型信息，显示总体topic分布
            topic_weights = cell_topic_matrix.mean(axis=0)
            ax2.bar(range(len(topic_weights)), topic_weights, color='steelblue')
            ax2.set_title('Average Topic Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Topics')
            ax2.set_ylabel('Average Weight')
        
        plt.tight_layout()
        
        # 保存图像
        umap_filename = viz_dir / f"{config.dataset}_cell_topic_celltype_{config.n_topics}.png"
        plt.savefig(umap_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"✅ Cell-topic visualization saved: {umap_filename}")
        
        return [str(umap_filename)]
        
    except ImportError:
        if verbose:
            print("⚠️ UMAP not available, skipping visualization")
        return []


def main():
    """主函数"""
    print("🚀 scFASTopic Training Pipeline")
    print("="*80)
    
    # 解析参数
    args = parse_args()
    
    # 创建配置
    config = FastopicConfig()
    
    # 更新配置
    config.embedding_file = args.embedding_file
    config.genes_file = args.genes_file
    config.dataset = args.dataset
    config.output_dir = args.output_dir
    config.visualization_dir = args.viz_dir
    config.n_topics = args.n_topics
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.DT_alpha = args.DT_alpha
    config.TW_alpha = args.TW_alpha
    config.theta_temp = args.theta_temp
    config.verbose = not args.quiet
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if config.verbose:
        print(f"📊 Configuration:")
        print(f"  Dataset: {config.dataset}")
        print(f"  Topics: {config.n_topics}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Embedding file: {config.embedding_file}")
        print(f"  Genes file: {config.genes_file}")
    
    try:
        # Step 1: 加载embeddings、基因和BOW矩阵
        # 自动推断BOW文件路径
        bow_file = config.embedding_file.replace('_cell_embeddings.pkl', '_expression_bow.pkl')
        cell_embeddings, selected_genes, expression_bow = load_embeddings_and_genes(
            config.embedding_file, config.genes_file, bow_file, config.verbose
        )
        
        # Step 2: 训练模型
        results, training_time = train_fastopic_model(
            cell_embeddings, selected_genes, expression_bow, config, config.verbose
        )
        
        # Step 3: 保存矩阵
        saved_files, matrices = save_all_matrices(
            results, cell_embeddings, config, config.verbose
        )
        
        # Step 4: 创建可视化
        viz_files = create_visualizations(results, config, config.verbose)
        
        # Step 5: 计算指标和生成报告
        metrics = calculate_topic_metrics(results['theta'])
        
        # 创建报告
        report = create_summary_report(
            matrices=matrices,
            metrics=metrics,
            dataset_name=config.dataset,
            n_topics=config.n_topics
        )
        
        # 保存报告
        report_file = Path(config.output_dir) / f"{config.dataset}_fastopic_report_{config.n_topics}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}/")
        print(f"🖼️ Visualizations saved to: {config.visualization_dir}/")
        print(f"📄 Report saved to: {report_file}")
        
        print(f"\n🎯 Final Results:")
        print(f"  Shannon Entropy: {results['shannon_entropy']:.3f}")
        print(f"  Effective Topics: {results['effective_topics']:.1f}")
        print(f"  Training Time: {training_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())