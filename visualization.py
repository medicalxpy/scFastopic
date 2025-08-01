#!/usr/bin/env python3
"""
scFASTopic 可视化模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from typing import Dict, Optional, List
import pickle

class ScFastopicVisualizer:
    """scFASTopic可视化器"""
    
    def __init__(self, output_dir: str = "visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_matrices(self, 
                     dataset_name: str, 
                     n_topics: int,
                     results_dir: str = "results") -> Dict[str, np.ndarray]:
        """
        加载保存的矩阵文件
        
        Args:
            dataset_name: 数据集名称
            n_topics: 主题数量
            results_dir: 结果目录
            
        Returns:
            matrices: 矩阵字典
        """
        results_path = Path(results_dir)
        matrices = {}
        
        # 需要加载的矩阵类型
        matrix_types = ['cell_embedding', 'cell_topic', 'topic_gene', 
                       'gene_embedding', 'topic_embedding']
        
        for matrix_type in matrix_types:
            file_name = f"{dataset_name}_{matrix_type}_{n_topics}.pkl"
            file_path = results_path / file_name
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    matrices[matrix_type.replace('_', '_')] = data['matrix']
                print(f"✅ Loaded {matrix_type}: {data['shape']}")
            else:
                print(f"⚠️ Matrix file not found: {file_path}")
        
        return matrices
    
    def create_umap_plot(self,
                        cell_topic_matrix: np.ndarray,
                        cell_info: Optional[pd.DataFrame] = None,
                        dataset_name: str = "Dataset",
                        n_topics: int = 20,
                        save: bool = True) -> str:
        """
        创建UMAP可视化
        
        Args:
            cell_topic_matrix: Cell-topic矩阵
            cell_info: 细胞信息
            dataset_name: 数据集名称
            n_topics: 主题数量
            save: 是否保存图片
            
        Returns:
            output_path: 输出文件路径
        """
        print("🎨 Creating UMAP visualization...")
        
        # UMAP降维
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            verbose=False
        )
        
        embedding = reducer.fit_transform(cell_topic_matrix)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 如果没有提供细胞信息，创建基于主题的信息
        if cell_info is None:
            n_cells = cell_topic_matrix.shape[0]
            main_topics = np.argmax(cell_topic_matrix, axis=1)
            
            cell_info = pd.DataFrame({
                'cell_type': [f'Topic_{t%8}' for t in main_topics],  # 简化为8种类型
                'batch': ['0'] * n_cells
            })
        
        # 左图：按细胞类型着色
        ax1 = axes[0]
        unique_types = cell_info['cell_type'].unique()
        
        for cell_type in unique_types:
            mask = cell_info['cell_type'] == cell_type
            ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                       label=cell_type, alpha=0.6, s=10)
        
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_title(f'UMAP - Cell Types ({dataset_name}-{n_topics})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 右图：按批次着色
        ax2 = axes[1]
        unique_batches = cell_info['batch'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, batch in enumerate(sorted(unique_batches)):
            mask = cell_info['batch'] == batch
            color = colors[i % len(colors)]
            ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                       label=f'Batch {batch}', alpha=0.6, s=10, color=color)
        
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        ax2.set_title(f'UMAP - Batches ({dataset_name}-{n_topics})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save:
            output_path = self.output_dir / f"{dataset_name}_umap_{n_topics}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ UMAP saved: {output_path}")
        else:
            output_path = None
        
        plt.show()
        return str(output_path) if output_path else ""
    
    def create_topic_analysis(self,
                             cell_topic_matrix: np.ndarray,
                             topic_gene_matrix: Optional[np.ndarray] = None,
                             dataset_name: str = "Dataset",
                             n_topics: int = 20,
                             save: bool = True) -> str:
        """
        创建主题分析图表
        
        Args:
            cell_topic_matrix: Cell-topic矩阵
            topic_gene_matrix: Topic-gene矩阵
            dataset_name: 数据集名称
            n_topics: 主题数量
            save: 是否保存
            
        Returns:
            output_path: 输出路径
        """
        print("📊 Creating topic analysis...")
        
        # 计算topic统计
        topic_sums = cell_topic_matrix.sum(axis=0)
        topic_percentages = (topic_sums / topic_sums.sum()) * 100
        
        # Shannon entropy
        probs = topic_percentages / 100
        shannon_entropy = -np.sum(probs * np.log(probs + 1e-10))
        effective_topics = np.exp(shannon_entropy)
        
        # 创建图表
        if topic_gene_matrix is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes = [axes]
        
        # Topic分布柱状图
        ax1 = axes[0] if len(axes) > 1 else axes[0][0]
        bars = ax1.bar(range(n_topics), topic_percentages, alpha=0.7, 
                      color=plt.cm.Set3(np.linspace(0, 1, n_topics)))
        ax1.set_title(f'Topic Distribution ({dataset_name}-{n_topics})', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Topic ID')
        ax1.set_ylabel('Percentage (%)')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, pct) in enumerate(zip(bars, topic_percentages)):
            if pct > 2:  # 只标注大于2%的
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 质量指标文本
        ax2 = axes[1] if len(axes) > 1 else axes[0][1]
        ax2.axis('off')
        
        metrics_text = f"""Topic Quality Metrics

Shannon Entropy: {shannon_entropy:.3f}
Effective Topics: {effective_topics:.1f} / {n_topics}
Max Topic %: {np.max(topic_percentages):.1f}%
Min Topic %: {np.min(topic_percentages):.1f}%

Quality Assessment:
{'✅ High Diversity' if shannon_entropy > 2.5 else '⚠️ Low Diversity'}
{'✅ Balanced Distribution' if np.max(topic_percentages) < 20 else '⚠️ Imbalanced'}
{'✅ All Topics Active' if np.min(topic_percentages) > 1 else '⚠️ Inactive Topics'}

Dataset: {dataset_name}
Topics: {n_topics}
Cells: {cell_topic_matrix.shape[0]:,}
"""
        
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                fontfamily='monospace', fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 如果有topic-gene矩阵，添加更多分析
        if topic_gene_matrix is not None and len(axes) > 1:
            # Cell-topic热力图（采样）
            ax3 = axes[1][0]
            sample_size = min(1000, cell_topic_matrix.shape[0])
            sample_indices = np.random.choice(cell_topic_matrix.shape[0], 
                                            sample_size, replace=False)
            sample_matrix = cell_topic_matrix[sample_indices]
            
            im = ax3.imshow(sample_matrix.T, aspect='auto', cmap='Blues', 
                           vmin=0, vmax=1)
            ax3.set_title(f'Cell-Topic Weights ({sample_size} cells)', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Cells (sampled)')
            ax3.set_ylabel('Topics')
            plt.colorbar(im, ax=ax3, shrink=0.8)
            
            # Topic相似性
            ax4 = axes[1][1]
            from sklearn.metrics.pairwise import cosine_similarity
            topic_similarity = cosine_similarity(topic_gene_matrix)
            
            im2 = ax4.imshow(topic_similarity, cmap='RdBu_r', vmin=-1, vmax=1)
            ax4.set_title('Topic Similarity (Gene-based)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Topics')
            ax4.set_ylabel('Topics')
            plt.colorbar(im2, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        
        # 保存图片
        if save:
            output_path = self.output_dir / f"{dataset_name}_topic_analysis_{n_topics}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Topic analysis saved: {output_path}")
        else:
            output_path = None
        
        plt.show()
        return str(output_path) if output_path else ""
    
    def create_comparison_plot(self,
                              results_dict: Dict[str, Dict],
                              dataset_name: str = "Dataset",
                              save: bool = True) -> str:
        """
        创建多个配置的对比图
        
        Args:
            results_dict: {config_name: {metrics...}} 格式的结果
            dataset_name: 数据集名称
            save: 是否保存
            
        Returns:
            output_path: 输出路径
        """
        print("⚖️ Creating comparison plot...")
        
        config_names = list(results_dict.keys())
        
        # 提取指标
        shannon_entropies = [results_dict[name]['shannon_entropy'] for name in config_names]
        effective_topics = [results_dict[name]['effective_topics'] for name in config_names]
        max_topic_pcts = [np.max(results_dict[name]['topic_percentages']) for name in config_names]
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Shannon Entropy对比
        axes[0].bar(config_names, shannon_entropies, alpha=0.7, color='skyblue')
        axes[0].set_title('Shannon Entropy Comparison', fontweight='bold')
        axes[0].set_ylabel('Shannon Entropy')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Effective Topics对比
        axes[1].bar(config_names, effective_topics, alpha=0.7, color='lightgreen')
        axes[1].set_title('Effective Topics Comparison', fontweight='bold')
        axes[1].set_ylabel('Effective Topics')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Max Topic %对比
        axes[2].bar(config_names, max_topic_pcts, alpha=0.7, color='salmon')
        axes[2].set_title('Max Topic % Comparison', fontweight='bold')
        axes[2].set_ylabel('Max Topic %')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"{dataset_name}_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison saved: {output_path}")
        else:
            output_path = None
        
        plt.show()
        return str(output_path) if output_path else ""
    
    def visualize_from_files(self,
                           dataset_name: str,
                           n_topics: int,
                           cell_info: Optional[pd.DataFrame] = None,
                           results_dir: str = "results") -> List[str]:
        """
        从保存的文件创建所有可视化
        
        Args:
            dataset_name: 数据集名称
            n_topics: 主题数量
            cell_info: 细胞信息
            results_dir: 结果目录
            
        Returns:
            output_files: 生成的文件列表
        """
        print(f"🎨 Creating visualizations for {dataset_name} ({n_topics} topics)")
        
        # 加载矩阵
        matrices = self.load_matrices(dataset_name, n_topics, results_dir)
        
        if 'cell_topic' not in matrices:
            print("❌ Cell-topic matrix not found!")
            return []
        
        output_files = []
        
        # UMAP可视化
        umap_file = self.create_umap_plot(
            matrices['cell_topic'],
            cell_info,
            dataset_name,
            n_topics
        )
        if umap_file:
            output_files.append(umap_file)
        
        # 主题分析
        topic_gene_matrix = matrices.get('topic_gene', None)
        analysis_file = self.create_topic_analysis(
            matrices['cell_topic'],
            topic_gene_matrix,
            dataset_name,
            n_topics
        )
        if analysis_file:
            output_files.append(analysis_file)
        
        print(f"✅ Generated {len(output_files)} visualization files")
        return output_files

def main():
    """测试函数"""
    # 示例：为20-topic结果创建可视化
    visualizer = ScFastopicVisualizer("visualization")
    
    # 从文件创建可视化
    output_files = visualizer.visualize_from_files(
        dataset_name="PBMC",
        n_topics=20,
        results_dir="results"
    )
    
    print(f"Generated files: {output_files}")

if __name__ == "__main__":
    main()