#!/usr/bin/env python3
"""
scFASTopic Results Visualization

根据文件路径自动识别结果类型并进行相应的可视化
支持的结果类型：
- cell embeddings
- cell topic matrix  
- gene embeddings
- topic embeddings
- topic gene matrix
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import umap
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import warnings
warnings.filterwarnings('ignore')

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: str = "visualization", adata_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.adata_path = adata_path
        self.adata = None
        
        # 支持的结果类型（主要通过目录识别）
        self.supported_types = {
            'cell_embedding',
            'cell_topic', 
            'gene_embedding',
            'topic_embedding',
            'topic_gene'
        }
        
        # 加载adata（如果提供了路径）
        if self.adata_path and os.path.exists(self.adata_path):
            self.load_adata()
    
    def identify_result_type(self, file_path: str) -> str:
        """根据文件目录路径识别结果类型"""
        path_obj = Path(file_path)
        
        # 检查文件所在的目录名
        parent_dirs = [p.name for p in path_obj.parents] + [path_obj.parent.name]
        
        # 目录名到结果类型的映射
        dir_type_mapping = {
            'cell_embedding': 'cell_embedding',
            'cell_topic': 'cell_topic', 
            'topic_gene': 'topic_gene',
            'gene_embedding': 'gene_embedding',
            'topic_embedding': 'topic_embedding'
        }
        
        # 检查目录名
        for dir_name in parent_dirs:
            if dir_name in dir_type_mapping:
                return dir_type_mapping[dir_name]
        
        # 如果目录识别失败，回退到文件名识别
        file_name = path_obj.stem.lower()
        
        # 按照优先级顺序检查文件名中的关键词
        priority_order = [
            ('cell_topic', ['cell_topic']),
            ('topic_gene', ['topic_gene']),
            ('gene_embedding', ['gene_embedding', 'gene_emb']),
            ('topic_embedding', ['topic_embedding', 'topic_emb']),
            ('cell_embedding', ['cell_embedding'])
        ]
        
        for result_type, keywords in priority_order:
            for keyword in keywords:
                if keyword in file_name:
                    return result_type
        
        return 'unknown'
    
    def preprocess_adata(self, adata_path: str, verbose: bool = True):
        """
        加载并预处理adata数据（与train_fastopic.py保持一致）
        
        Args:
            adata_path: 单细胞数据路径
            verbose: 是否详细输出
            
        Returns:
            adata: 预处理后的adata对象
        """
        if verbose:
            print(f"📁 加载adata: {adata_path}")
        
        # 加载数据
        adata = sc.read_h5ad(adata_path)
        
        if verbose:
            print(f"原始数据维度: {adata.shape}")
        
        # 保存cell type信息（在预处理前）
        cell_type_backup = None
        if 'cell_type' in adata.obs.columns:
            cell_type_backup = adata.obs['cell_type'].copy()
            if verbose:
                print(f"✅ 发现cell_type信息: {len(cell_type_backup.unique())} 个类型")
                print(f"   类型: {list(cell_type_backup.unique())}")
        
        # 简单过滤（与train_fastopic.py保持一致）
        # 过滤低质量细胞 (表达基因数 < 200)
        sc.pp.filter_cells(adata, min_genes=200)
        
        # 过滤低表达基因 (在 < 3个细胞中表达)
        sc.pp.filter_genes(adata, min_cells=3)
        
        if verbose:
            print(f"过滤后数据维度: {adata.shape}")
        
        # 恢复cell type信息（确保与过滤后的细胞对应）
        if cell_type_backup is not None:
            # 获取过滤后保留的细胞索引
            remaining_cells = adata.obs.index
            adata.obs['cell_type'] = cell_type_backup.loc[remaining_cells]
            if verbose:
                print(f"✅ 恢复cell_type信息: {len(adata.obs['cell_type'].unique())} 个类型")
        
        # 标准化到每个细胞总计数为1
        sc.pp.normalize_total(adata, target_sum=1)
        
        # log1p变换
        sc.pp.log1p(adata)
        
        if verbose:
            print(f"✅ 预处理完成: {adata.shape[0]} 个细胞, {adata.shape[1]} 个基因")
        
        return adata
    
    def load_adata(self):
        """加载并预处理adata数据"""
        try:
            self.adata = self.preprocess_adata(self.adata_path, verbose=True)
            
            # 检查是否有cell type信息
            if 'cell_type' in self.adata.obs.columns:
                print(f"✅ Cell type信息可用于染色")
            else:
                print("⚠️ 未发现cell_type列，请检查adata.obs中的列名")
                print(f"   可用的obs列: {list(self.adata.obs.columns)}")
                
        except Exception as e:
            print(f"❌ 加载adata失败: {e}")
            self.adata = None
    
    def load_data(self, file_path: str) -> Any:
        """加载pickle数据"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ 成功加载: {file_path}")
            print(f"   数据类型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"   数据形状: {data.shape}")
            return data
        except Exception as e:
            print(f"❌ 加载失败: {file_path}, 错误: {e}")
            return None
    
    def load_result(self, file_path: str) -> Dict[str, Any]:
        """加载结果数据"""
        result_type = self.identify_result_type(file_path)
        data = self.load_data(file_path)
        
        if data is None:
            return None
        
        return {
            'type': result_type,
            'data': data,
            'file_path': file_path
        }
    
    def load_results(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """批量加载多个文件"""
        results = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"📁 加载文件: {file_path}")
                result = self.load_result(file_path)
                if result:
                    print(f"   类型: {result['type']}")
                    results.append(result)
            else:
                print(f"❌ 文件不存在: {file_path}")
        
        return results
    
    def visualize_results(self, results: List[Dict[str, Any]]):
        """可视化结果（占位符，待实现具体绘图逻辑）"""
        print(f"\n🎨 可视化 {len(results)} 个结果文件")
        
        for result in results:
            result_type = result['type']
            data = result['data']
            file_path = result['file_path']
            
            print(f"\n📊 {result_type}: {file_path}")
            print(f"   数据形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            # 实现具体的绘图逻辑
            if result_type == 'cell_embedding':
                self.plot_cell_embedding_umap(data, file_path)
            elif result_type == 'cell_topic':
                self.plot_cell_topic_umap(data, file_path)
            elif result_type == 'gene_embedding':
                # self.plot_gene_embeddings(data, file_path)
                pass
            elif result_type == 'topic_embedding':
                self.plot_topic_embedding_umap(data, file_path)
            elif result_type == 'topic_gene':
                # self.plot_topic_gene_matrix(data, file_path)
                pass
    
    def plot_cell_topic_umap(self, cell_topic_matrix: np.ndarray, file_path: str):
        """绘制cell topic的UMAP降维图，根据cell type染色"""
        print(f"\n🎨 绘制Cell Topic UMAP图: {file_path}")
        
        # 检查数据维度
        n_cells, n_topics = cell_topic_matrix.shape
        print(f"   细胞数量: {n_cells}, 主题数量: {n_topics}")
        
        # 检查是否有adata和cell type信息
        if self.adata is None:
            print("⚠️ 未提供adata路径，将使用默认颜色")
            cell_types = None
        elif 'cell_type' not in self.adata.obs.columns:
            print("⚠️ adata中无cell_type信息，将使用默认颜色")
            cell_types = None
        else:
            cell_types = self.adata.obs['cell_type'].values
            # 检查细胞数量是否匹配
            if len(cell_types) != n_cells:
                print(f"⚠️ 细胞数量不匹配: cell_topic({n_cells}) vs adata({len(cell_types)})")
                cell_types = None
        
        # 使用UMAP进行降维
        print("🔄 执行UMAP降维...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(cell_topic_matrix)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        if cell_types is not None:
            # 根据cell type染色
            unique_types = np.unique(cell_types)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
            
            for i, cell_type in enumerate(unique_types):
                mask = cell_types == cell_type
                plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                           c=[colors[i]], label=cell_type, alpha=0.7, s=20)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            title = f"Cell Topic UMAP (colored by cell type)\n{n_cells} cells, {n_topics} topics"
        else:
            # 使用默认颜色
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                       c='skyblue', alpha=0.7, s=20)
            title = f"Cell Topic UMAP\n{n_cells} cells, {n_topics} topics"
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        file_stem = Path(file_path).stem
        output_file = self.output_dir / f"{file_stem}_umap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ UMAP图已保存: {output_file}")
        
        return umap_coords
    
    def plot_cell_embedding_umap(self, cell_embeddings: np.ndarray, file_path: str):
        """绘制cell embedding的UMAP降维图，根据cell type染色"""
        print(f"\n🎨 绘制Cell Embedding UMAP图: {file_path}")
        
        # 检查数据维度
        n_cells, embedding_dim = cell_embeddings.shape
        print(f"   细胞数量: {n_cells}, Embedding维度: {embedding_dim}")
        
        # 检查是否有adata和cell type信息
        if self.adata is None:
            print("⚠️ 未提供adata路径，将使用默认颜色")
            cell_types = None
        elif 'cell_type' not in self.adata.obs.columns:
            print("⚠️ adata中无cell_type信息，将使用默认颜色")
            cell_types = None
        else:
            cell_types = self.adata.obs['cell_type'].values
            # 检查细胞数量是否匹配
            if len(cell_types) != n_cells:
                print(f"⚠️ 细胞数量不匹配: cell_embedding({n_cells}) vs adata({len(cell_types)})")
                cell_types = None
        
        # 使用UMAP进行降维
        print("🔄 执行UMAP降维...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(cell_embeddings)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        if cell_types is not None:
            # 根据cell type染色
            unique_types = np.unique(cell_types)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
            
            for i, cell_type in enumerate(unique_types):
                mask = cell_types == cell_type
                plt.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                           c=[colors[i]], label=cell_type, alpha=0.7, s=20)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            title = f"Cell Embedding UMAP (colored by cell type)\n{n_cells} cells, {embedding_dim}D embeddings"
        else:
            # 使用默认颜色
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                       c='skyblue', alpha=0.7, s=20)
            title = f"Cell Embedding UMAP\n{n_cells} cells, {embedding_dim}D embeddings"
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        file_stem = Path(file_path).stem
        output_file = self.output_dir / f"{file_stem}_umap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ UMAP图已保存: {output_file}")
        
        return umap_coords
    
    def plot_topic_embedding_umap(self, topic_embeddings: np.ndarray, file_path: str):
        """绘制topic embedding的UMAP降维图，根据topic ID染色"""
        print(f"\n🎨 绘制Topic Embedding UMAP图: {file_path}")
        
        # 检查数据维度
        n_topics, embedding_dim = topic_embeddings.shape
        print(f"   主题数量: {n_topics}, Embedding维度: {embedding_dim}")
        
        # 使用UMAP进行降维
        print("🔄 执行UMAP降维...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_topics-1), min_dist=0.1)
        umap_coords = reducer.fit_transform(topic_embeddings)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 根据topic ID染色
        topic_ids = np.arange(n_topics)
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_topics)))
        
        # 如果主题数量超过20个，使用连续颜色映射
        if n_topics > 20:
            colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
            scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                c=topic_ids, cmap='viridis', alpha=0.8, s=80)
            plt.colorbar(scatter, label='Topic ID')
        else:
            # 对于20个以下的主题，使用离散颜色并添加图例
            for i in range(n_topics):
                plt.scatter(umap_coords[i, 0], umap_coords[i, 1], 
                           c=[colors[i]], label=f'Topic {i}', alpha=0.8, s=80)
            
            # 只有在主题数量不太多时才显示图例
            if n_topics <= 12:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 添加topic ID标注
        for i in range(n_topics):
            plt.annotate(f'T{i}', (umap_coords[i, 0], umap_coords[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.7)
        
        title = f"Topic Embedding UMAP\n{n_topics} topics, {embedding_dim}D embeddings"
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        file_stem = Path(file_path).stem
        output_file = self.output_dir / f"{file_stem}_umap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ UMAP图已保存: {output_file}")
        
        return umap_coords

def main():
    parser = argparse.ArgumentParser(description="scFASTopic结果可视化")
    parser.add_argument("files", nargs="+", help="结果文件路径")
    parser.add_argument("--output_dir", default="visualization", help="输出目录")
    parser.add_argument("--adata_path", help="adata文件路径，用于获取cell type信息")
    parser.add_argument("--no_plot", action="store_true", help="只加载数据不绘图")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = ResultVisualizer(output_dir=args.output_dir, adata_path=args.adata_path)
    
    # 批量加载结果
    results = visualizer.load_results(args.files)
    
    if not results:
        print("❌ 没有成功加载的文件")
        return
    
    # 输出加载总结
    print(f"\n{'='*60}")
    print("加载总结")
    print(f"{'='*60}")
    
    type_counts = {}
    for result in results:
        result_type = result['type']
        type_counts[result_type] = type_counts.get(result_type, 0) + 1
    
    for result_type, count in type_counts.items():
        print(f"📊 {result_type}: {count} 个文件")
    
    # 可视化（如果需要）
    if not args.no_plot:
        visualizer.visualize_results(results)

if __name__ == "__main__":
    main()