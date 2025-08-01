#!/usr/bin/env python3
"""
scFASTopic Cell Embedding Extraction Script

使用原始Geneformer库提取cell embeddings的独立脚本
直接调用Geneformer的TranscriptomeTokenizer和EmbExtractor
"""

import os
import sys
import argparse
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import scanpy as sc
import anndata as ad

# 添加Geneformer路径到系统路径
sys.path.insert(0, '/root/autodl-tmp/Geneformer')

# 导入Geneformer原始组件
from geneformer import TranscriptomeTokenizer, EmbExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置类
class EmbeddingConfig:
    def __init__(self, 
                 input_data: str,
                 dataset_name: str = "dataset",
                 output_dir: str = "results",
                 max_cells: Optional[int] = None,
                 n_top_genes: Optional[int] = None,
                 use_geneformer: bool = True,
                 model_path: Optional[str] = None,
                 emb_layer: int = -1,
                 forward_batch_size: int = 100,
                 nproc: int = 4,
                 verbose: bool = False):
        
        self.input_data = input_data
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.max_cells = max_cells
        self.n_top_genes = n_top_genes
        self.use_geneformer = use_geneformer
        self.model_path = model_path
        self.emb_layer = emb_layer
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.verbose = verbose
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

def preprocess_single_cell_data(adata: ad.AnnData, 
                               n_top_genes: Optional[int] = 2000, 
                               max_cells: Optional[int] = None,
                               verbose: bool = False) -> Tuple[ad.AnnData, List[str]]:
    """预处理单细胞数据"""
    if verbose:
        logger.info(f"原始数据: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # 复制数据避免修改原始数据
    adata = adata.copy()
    
    # 基本过滤
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    if verbose:
        logger.info(f"过滤后: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # 保存原始counts
    adata.raw = adata
    
    # 归一化和标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 寻找高变基因或使用全部基因
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
        highly_variable_genes = adata.var_names[adata.var.highly_variable].tolist()
        adata = adata[:, highly_variable_genes].copy()
        if verbose:
            logger.info(f"HVG模式: 选择了 {len(highly_variable_genes)} 个高变基因")
    else:
        # 使用全部基因
        highly_variable_genes = adata.var_names.tolist()
        if verbose:
            logger.info(f"完整数据模式: 使用全部 {len(highly_variable_genes)} 个基因")
    
    # 限制细胞数量
    if max_cells and adata.n_obs > max_cells:
        adata = adata[:max_cells, :].copy()
        if verbose:
            logger.info(f"随机采样到 {max_cells} 个细胞")
    
    
    
    # 还原到raw counts用于Geneformer
    raw_counts = adata.raw.X[:, adata.raw.var_names.isin(adata.var_names)]
    adata.X = raw_counts.astype(int)  # Geneformer需要整数counts
    
    # 添加Geneformer需要的n_counts列
    adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    
    # 保存真实的基因表达矩阵用于FASTopic训练
    # 使用log1p标准化的表达数据作为BOW矩阵
    import scipy.sparse as sp
    if hasattr(raw_counts, 'toarray'):
        expression_matrix = raw_counts.toarray().astype(np.float32)
    else:
        expression_matrix = raw_counts.astype(np.float32)
    
    # 标准化：log1p + min-max scaling to [0,1]
    expression_matrix = np.log1p(expression_matrix)
    row_max = expression_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1  # 避免除零
    expression_matrix = expression_matrix / row_max
    
    # 转换为稀疏矩阵
    expression_bow = sp.csr_matrix(expression_matrix)
    
    if verbose:
        logger.info(f"最终数据: {adata.n_obs} cells × {adata.n_vars} genes")
        logger.info(f"基因表达BOW矩阵: {expression_bow.shape}, 稀疏度: {1 - expression_bow.nnz / (expression_bow.shape[0] * expression_bow.shape[1]):.3f}")
    
    return adata, highly_variable_genes, expression_bow

def extract_geneformer_embeddings(adata: ad.AnnData, 
                                 config: EmbeddingConfig,
                                 verbose: bool = False) -> np.ndarray:
    """使用真实Geneformer提取cell embeddings"""
    
    # 创建临时目录
    temp_dir = Path(config.output_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. 使用TranscriptomeTokenizer进行tokenization
        if verbose:
            logger.info("开始tokenization...")
        
        # 保存h5ad文件
        temp_h5ad_path = temp_dir / f"{config.dataset_name}_temp.h5ad"
        adata.write_h5ad(temp_h5ad_path)
        
        # 初始化tokenizer
        tokenizer = TranscriptomeTokenizer(
            nproc=config.nproc,
            model_version="V2",  # 使用V2版本
            special_token=True
        )
        
        # Tokenize数据
        tokenized_dataset_path = temp_dir / f"{config.dataset_name}_tokenized.dataset"
        tokenizer.tokenize_data(
            data_directory=temp_dir,
            output_directory=temp_dir,
            output_prefix=f"{config.dataset_name}_tokenized",
            file_format="h5ad",
            input_identifier=f"{config.dataset_name}_temp"
        )
        
        if verbose:
            logger.info(f"Tokenization完成，输出: {tokenized_dataset_path}")
        
        # 2. 使用EmbExtractor提取embeddings
        if verbose:
            logger.info("开始提取embeddings...")
        
        # 初始化embedding extractor
        embex = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell",  # 提取cell embeddings
            max_ncells=config.max_cells,
            emb_layer=config.emb_layer,
            forward_batch_size=config.forward_batch_size,
            nproc=config.nproc,
            model_version="V2"
        )
        
        # 提取embeddings
        model_path = config.model_path or "/root/autodl-tmp/Geneformer/models/geneformer-v2"
        
        embs_df = embex.extract_embs(
            model_directory=model_path,
            input_data_file=str(tokenized_dataset_path),
            output_directory=str(temp_dir),
            output_prefix=f"{config.dataset_name}_embeddings"
        )
        
        # 转换为numpy数组
        cell_embeddings = embs_df.iloc[:, :-1].values.astype(np.float32)  # 排除最后的label列
        
        if verbose:
            logger.info(f"成功提取embeddings: {cell_embeddings.shape}")
        
        return cell_embeddings
        
    except Exception as e:
        logger.error(f"Geneformer embedding提取失败: {e}")
        logger.warning("回退到PCA方法")
        return extract_pca_embeddings(adata, verbose=verbose)
    
    finally:
        # 清理临时文件
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def extract_pca_embeddings(adata: ad.AnnData, 
                          embed_dim: int = 768,
                          verbose: bool = False) -> np.ndarray:
    """PCA备选方案"""
    from sklearn.decomposition import PCA
    
    if verbose:
        logger.info("使用PCA提取embeddings")
    
    # 确保数据是dense格式
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # 动态调整PCA维度
    n_samples, n_features = X.shape
    embed_dim = min(embed_dim, n_samples - 1, n_features)
    
    pca = PCA(n_components=embed_dim, random_state=42)
    embeddings = pca.fit_transform(X)
    
    if verbose:
        logger.info(f"PCA embeddings: {embeddings.shape}, 解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
    
    return embeddings.astype(np.float32)

def extract_embeddings(adata: ad.AnnData, config: EmbeddingConfig, verbose: bool = False) -> np.ndarray:
    """主要的embedding提取函数"""
    
    if config.use_geneformer:
        try:
            return extract_geneformer_embeddings(adata, config, verbose=verbose)
        except Exception as e:
            logger.error(f"Geneformer提取失败: {e}")
            logger.warning("回退到PCA方法")
            return extract_pca_embeddings(adata, verbose=verbose)
    else:
        return extract_pca_embeddings(adata, verbose=verbose)

def save_results(cell_embeddings: np.ndarray, 
                selected_genes: List[str],
                expression_bow,  # scipy.sparse.csr_matrix
                config: EmbeddingConfig,
                verbose: bool = False) -> Dict[str, str]:
    """保存结果文件"""
    
    output_files = {}
    
    # 保存cell embeddings
    embedding_file = os.path.join(config.output_dir, f"{config.dataset_name}_cell_embeddings.pkl")
    with open(embedding_file, 'wb') as f:
        pickle.dump(cell_embeddings, f)
    output_files['embeddings'] = embedding_file
    
    # 保存基因列表
    genes_file = os.path.join(config.output_dir, f"{config.dataset_name}_selected_genes.pkl")
    with open(genes_file, 'wb') as f:
        pickle.dump(selected_genes, f)
    output_files['genes'] = genes_file
    
    # 保存真实的基因表达BOW矩阵
    bow_file = os.path.join(config.output_dir, f"{config.dataset_name}_expression_bow.pkl")
    with open(bow_file, 'wb') as f:
        pickle.dump(expression_bow, f)
    output_files['expression_bow'] = bow_file
    
    # 保存配置信息
    config_file = os.path.join(config.output_dir, f"{config.dataset_name}_embedding_config.json")
    config_dict = {
        'dataset_name': config.dataset_name,
        'max_cells': config.max_cells,
        'n_top_genes': config.n_top_genes,
        'use_geneformer': config.use_geneformer,
        'emb_layer': config.emb_layer,
        'forward_batch_size': config.forward_batch_size,
        'embedding_shape': cell_embeddings.shape,
        'n_genes': len(selected_genes)
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    output_files['config'] = config_file
    
    if verbose:
        logger.info(f"结果已保存:")
        logger.info(f"  - Embeddings: {embedding_file}")
        logger.info(f"  - Genes: {genes_file}")
        logger.info(f"  - Expression BOW: {bow_file}")
        logger.info(f"  - Config: {config_file}")
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description="scFASTopic Cell Embedding Extraction")
    parser.add_argument("--input_data", required=True, help="输入数据路径(.h5ad)")
    parser.add_argument("--dataset_name", default="dataset", help="数据集名称")
    parser.add_argument("--output_dir", default="results", help="输出目录")
    parser.add_argument("--max_cells", type=int, help="最大细胞数量")
    parser.add_argument("--n_top_genes", type=int, help="高变基因数量，不指定则使用全部基因")
    parser.add_argument("--use_geneformer", action="store_true", help="使用真实Geneformer模型")
    parser.add_argument("--model_path", help="Geneformer模型路径")
    parser.add_argument("--emb_layer", type=int, default=-1, help="提取embedding的层")
    parser.add_argument("--forward_batch_size", type=int, default=100, help="前向传播批次大小")
    parser.add_argument("--nproc", type=int, default=4, help="进程数")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建配置
    config = EmbeddingConfig(
        input_data=args.input_data,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_cells=args.max_cells,
        n_top_genes=args.n_top_genes,
        use_geneformer=args.use_geneformer,
        model_path=args.model_path,
        emb_layer=args.emb_layer,
        forward_batch_size=args.forward_batch_size,
        nproc=args.nproc,
        verbose=args.verbose
    )
    
    try:
        # 加载数据
        if config.verbose:
            logger.info(f"加载数据: {config.input_data}")
        
        adata = sc.read_h5ad(config.input_data)
        
        # 预处理数据
        if config.verbose:
            logger.info("预处理数据...")
        
        adata_processed, selected_genes, expression_bow = preprocess_single_cell_data(
            adata, 
            n_top_genes=config.n_top_genes,
            max_cells=config.max_cells,
            verbose=config.verbose
        )
        
        # 提取embeddings
        if config.verbose:
            logger.info("提取cell embeddings...")
        
        cell_embeddings = extract_embeddings(adata_processed, config, verbose=config.verbose)
        
        # 保存结果
        if config.verbose:
            logger.info("保存结果...")
        
        saved_files = save_results(cell_embeddings, selected_genes, expression_bow, config, verbose=config.verbose)
        
        # 输出总结
        logger.info("=== Cell Embedding提取完成 ===")
        logger.info(f"数据集: {config.dataset_name}")
        logger.info(f"细胞数: {cell_embeddings.shape[0]}")
        logger.info(f"基因数: {len(selected_genes)}")
        logger.info(f"Embedding维度: {cell_embeddings.shape[1]}")
        logger.info(f"方法: {'Geneformer' if config.use_geneformer else 'PCA'}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"提取过程失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()