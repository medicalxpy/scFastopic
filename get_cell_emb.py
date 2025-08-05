#!/usr/bin/env python3
"""
scFASTopic Cell Embedding Extraction Script

支持多种embedding方法：
1. Geneformer - 使用预训练的Geneformer模型
2. GenePT - 使用GenePT预训练基因embedding (策略1: 加权平均)
3. GenePT - 使用GenePT预训练基因embedding (策略2: 句子embedding)
4. PCA - 备选方案
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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加Geneformer路径到系统路径
sys.path.insert(0, '/root/autodl-tmp/scFastopic/Geneformer')

# 导入Geneformer原始组件 (可选)
try:
    from geneformer import TranscriptomeTokenizer, EmbExtractor
    GENEFORMER_AVAILABLE = True
except ImportError:
    GENEFORMER_AVAILABLE = False
    print("⚠️ Geneformer not available, only GenePT and PCA methods will work")

# 导入Sentence Transformers (可选)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence Transformers not available, GenePT strategy2 will use simplified version")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置类
class EmbeddingConfig:
    def __init__(self, 
                 input_data: str,
                 dataset_name: str = "dataset",
                 output_dir: str = "results/cell_embedding",
                 embedding_type: str = "geneformer",
                 genept_strategy: str = "strategy1",
                 max_cells: Optional[int] = None,
                 n_top_genes: Optional[int] = None,
                 model_path: Optional[str] = None,
                 genept_path: Optional[str] = None,
                 emb_layer: int = -1,
                 forward_batch_size: int = 100,
                 nproc: int = 4,
                 verbose: bool = False):
        
        self.input_data = input_data
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.embedding_type = embedding_type  # "geneformer", "genept", "pca"
        self.genept_strategy = genept_strategy  # "strategy1", "strategy2"
        self.max_cells = max_cells
        self.n_top_genes = n_top_genes
        self.model_path = model_path
        self.genept_path = genept_path or '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
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
        model_path = config.model_path or "/root/autodl-tmp/scFastopic/Geneformer/Geneformer-V2-316M"
        
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

def load_genept_embeddings(genept_path: str, verbose: bool = False) -> Dict[str, np.ndarray]:
    """加载GenePT基因embedding"""
    if verbose:
        logger.info(f"加载GenePT embedding: {genept_path}")
    
    with open(genept_path, 'rb') as f:
        genept_data = pickle.load(f)
    
    if verbose:
        logger.info(f"成功加载 {len(genept_data)} 个基因的embedding")
    
    return genept_data

def extract_genept_strategy1(adata: ad.AnnData, 
                            genept_embeddings: Dict[str, np.ndarray], 
                            verbose: bool = False) -> np.ndarray:
    """
    GenePT策略1: 基于表达水平加权平均基因embedding
    """
    if verbose:
        logger.info("执行GenePT策略1: 基于表达水平加权平均基因embedding")
    
    # 获取基因symbols，优先使用gene_symbols列，否则使用var_names
    if 'gene_symbols' in adata.var.columns:
        gene_symbols = adata.var['gene_symbols'].tolist()
        if verbose:
            logger.info("使用adata.var['gene_symbols']进行GenePT匹配")
    else:
        gene_symbols = adata.var_names.tolist()
        if verbose:
            logger.info("使用adata.var_names进行GenePT匹配")
    
    # 找到在GenePT中存在的基因
    available_genes = [gene for gene in gene_symbols if gene in genept_embeddings]
    
    if verbose:
        logger.info(f"总基因数: {len(gene_symbols)}")
        logger.info(f"在GenePT中找到的基因: {len(available_genes)}")
    
    if len(available_genes) == 0:
        raise ValueError("没有找到任何在GenePT中存在的基因")
    
    # 获取表达矩阵 (cells x genes)
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # 构建可用基因的索引 - 基于gene_symbols匹配
    gene_indices = [i for i, gene in enumerate(gene_symbols) if gene in genept_embeddings]
    available_gene_symbols = [gene_symbols[i] for i in gene_indices]
    
    # 提取对应的表达数据
    X_available = X[:, gene_indices]  # (n_cells, n_available_genes)
    
    # 获取GenePT embeddings
    first_gene = available_gene_symbols[0]
    embedding_dim = len(genept_embeddings[first_gene])
    gene_embeddings = np.array([genept_embeddings[gene] for gene in available_gene_symbols])
    
    if verbose:
        logger.info(f"基因embedding维度: {embedding_dim}")
        logger.info(f"使用的基因数量: {len(available_gene_symbols)}")
    
    # 计算加权平均cell embedding
    cell_embeddings = []
    
    for i in range(X_available.shape[0]):
        cell_expression = X_available[i, :]
        
        # 避免除零错误
        if np.sum(cell_expression) == 0:
            cell_embedding = np.mean(gene_embeddings, axis=0)
        else:
            weights = cell_expression / np.sum(cell_expression)
            cell_embedding = np.sum(gene_embeddings * weights[:, np.newaxis], axis=0)
        
        cell_embeddings.append(cell_embedding)
    
    cell_embeddings = np.array(cell_embeddings)
    
    if verbose:
        logger.info(f"生成的cell embedding形状: {cell_embeddings.shape}")
    
    return cell_embeddings

def extract_genept_strategy2(adata: ad.AnnData, 
                            genept_embeddings: Dict[str, np.ndarray], 
                            verbose: bool = False) -> np.ndarray:
    """
    GenePT策略2: 基于表达水平排序的基因名称创建句子embedding
    """
    if verbose:
        logger.info("执行GenePT策略2: 基于表达水平排序创建句子embedding")
    
    # 获取基因symbols，优先使用gene_symbols列，否则使用var_names
    if 'gene_symbols' in adata.var.columns:
        gene_symbols = adata.var['gene_symbols'].tolist()
        if verbose:
            logger.info("使用adata.var['gene_symbols']进行GenePT匹配")
    else:
        gene_symbols = adata.var_names.tolist()
        if verbose:
            logger.info("使用adata.var_names进行GenePT匹配")
    
    available_genes = [gene for gene in gene_symbols if gene in genept_embeddings]
    
    if len(available_genes) == 0:
        raise ValueError("没有找到任何在GenePT中存在的基因")
    
    # 获取表达矩阵
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # 构建可用基因的索引 - 基于gene_symbols匹配
    gene_indices = [i for i, gene in enumerate(gene_symbols) if gene in genept_embeddings]
    available_gene_symbols = [gene_symbols[i] for i in gene_indices]
    X_available = X[:, gene_indices]
    
    # 尝试使用sentence transformer
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            if verbose:
                logger.info("加载Sentence Transformer模型...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 为每个细胞创建基因句子
            cell_sentences = []
            top_k = min(50, len(available_gene_symbols))
            
            for i in range(X_available.shape[0]):
                cell_expression = X_available[i, :]
                sorted_indices = np.argsort(cell_expression)[::-1]
                top_genes = [available_gene_symbols[idx] for idx in sorted_indices[:top_k] if cell_expression[idx] > 0]
                
                if len(top_genes) == 0:
                    top_genes = available_gene_symbols[:10]
                
                gene_sentence = " ".join(top_genes)
                cell_sentences.append(gene_sentence)
            
            if verbose:
                logger.info("生成句子embedding...")
            cell_embeddings = model.encode(cell_sentences, show_progress_bar=verbose)
            
            return cell_embeddings
            
        except Exception as e:
            if verbose:
                logger.warning(f"Sentence Transformer失败: {e}, 使用简化版本")
    
    # 简化版本：基于表达排序直接组合GenePT embedding
    if verbose:
        logger.info("使用简化版策略2: 基于表达排序组合GenePT embedding")
    
    first_gene = available_gene_symbols[0]
    embedding_dim = len(genept_embeddings[first_gene])
    
    cell_embeddings = []
    top_k = min(20, len(available_gene_symbols))
    
    for i in range(X_available.shape[0]):
        cell_expression = X_available[i, :]
        sorted_indices = np.argsort(cell_expression)[::-1]
        top_gene_indices = sorted_indices[:top_k]
        
        top_embeddings = []
        top_weights = []
        
        for idx in top_gene_indices:
            if cell_expression[idx] > 0:
                gene_symbol = available_gene_symbols[idx]
                top_embeddings.append(genept_embeddings[gene_symbol])
                top_weights.append(cell_expression[idx])
        
        if len(top_embeddings) == 0:
            top_embeddings = [genept_embeddings[available_gene_symbols[i]] for i in range(min(10, len(available_gene_symbols)))]
            top_weights = [1.0] * len(top_embeddings)
        
        top_embeddings = np.array(top_embeddings)
        top_weights = np.array(top_weights)
        top_weights = top_weights / np.sum(top_weights)
        
        cell_embedding = np.sum(top_embeddings * top_weights[:, np.newaxis], axis=0)
        cell_embeddings.append(cell_embedding)
    
    cell_embeddings = np.array(cell_embeddings)
    
    if verbose:
        logger.info(f"生成的cell embedding形状: {cell_embeddings.shape}")
    
    return cell_embeddings

def extract_embeddings(adata: ad.AnnData, config: EmbeddingConfig, verbose: bool = False) -> np.ndarray:
    """主要的embedding提取函数"""
    
    if config.embedding_type == "geneformer":
        if not GENEFORMER_AVAILABLE:
            logger.warning("Geneformer不可用，回退到PCA方法")
            return extract_pca_embeddings(adata, verbose=verbose)
        
        try:
            return extract_geneformer_embeddings(adata, config, verbose=verbose)
        except Exception as e:
            logger.error(f"Geneformer提取失败: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("回退到PCA方法")
            return extract_pca_embeddings(adata, verbose=verbose)
    
    elif config.embedding_type == "genept":
        try:
            # 加载GenePT embeddings
            genept_embeddings = load_genept_embeddings(config.genept_path, verbose=verbose)
            
            if config.genept_strategy == "strategy1":
                return extract_genept_strategy1(adata, genept_embeddings, verbose=verbose)
            elif config.genept_strategy == "strategy2":
                return extract_genept_strategy2(adata, genept_embeddings, verbose=verbose)
            else:
                raise ValueError(f"未知的GenePT策略: {config.genept_strategy}")
                
        except Exception as e:
            logger.error(f"GenePT提取失败: {e}")
            logger.warning("回退到PCA方法")
            return extract_pca_embeddings(adata, verbose=verbose)
    
    elif config.embedding_type == "pca":
        return extract_pca_embeddings(adata, verbose=verbose)
    
    else:
        raise ValueError(f"未知的embedding类型: {config.embedding_type}")

def save_results(cell_embeddings: np.ndarray, 
                config: EmbeddingConfig,
                verbose: bool = False) -> str:
    """保存cell embedding结果"""
    
    # 根据embedding类型生成文件名
    if config.embedding_type == "genept":
        filename = f"{config.dataset_name}_{config.embedding_type}_{config.genept_strategy}.pkl"
    else:
        filename = f"{config.dataset_name}_{config.embedding_type}.pkl"
    
    # 保存cell embeddings
    embedding_file = os.path.join(config.output_dir, filename)
    with open(embedding_file, 'wb') as f:
        pickle.dump(cell_embeddings, f)
    
    if verbose:
        logger.info(f"Cell embeddings已保存到: {embedding_file}")
    
    return embedding_file

def main():
    parser = argparse.ArgumentParser(description="scFASTopic Cell Embedding Extraction")
    parser.add_argument("--input_data", required=True, help="输入数据路径(.h5ad)")
    parser.add_argument("--dataset_name", default="dataset", help="数据集名称")
    parser.add_argument("--output_dir", default="results/cell_embedding", help="输出目录")
    parser.add_argument("--embedding_type", default="geneformer", 
                       choices=["geneformer", "genept", "pca"], 
                       help="Embedding方法 (geneformer/genept/pca)")
    parser.add_argument("--genept_strategy", default="strategy1",
                       choices=["strategy1", "strategy2"],
                       help="GenePT策略 (strategy1: 加权平均, strategy2: 句子embedding)")
    parser.add_argument("--max_cells", type=int, help="最大细胞数量")
    parser.add_argument("--n_top_genes", type=int, help="高变基因数量，不指定则使用全部基因")
    parser.add_argument("--model_path", help="Geneformer模型路径")
    parser.add_argument("--genept_path", help="GenePT embedding文件路径")
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
        embedding_type=args.embedding_type,
        genept_strategy=args.genept_strategy,
        max_cells=args.max_cells,
        n_top_genes=args.n_top_genes,
        model_path=args.model_path,
        genept_path=args.genept_path,
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
            logger.info(f"提取cell embeddings (方法: {config.embedding_type})...")
        
        cell_embeddings = extract_embeddings(adata_processed, config, verbose=config.verbose)
        
        # 保存结果
        if config.verbose:
            logger.info("保存结果...")
        
        saved_file = save_results(cell_embeddings, config, verbose=config.verbose)
        
        # 输出总结
        logger.info("=== Cell Embedding提取完成 ===")
        logger.info(f"数据集: {config.dataset_name}")
        logger.info(f"细胞数: {cell_embeddings.shape[0]}")
        logger.info(f"Embedding维度: {cell_embeddings.shape[1]}")
        logger.info(f"方法: {config.embedding_type}")
        if config.embedding_type == "genept":
            logger.info(f"策略: {config.genept_strategy}")
        logger.info(f"保存位置: {saved_file}")
        
        return saved_file
        
    except Exception as e:
        logger.error(f"提取过程失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()