#!/usr/bin/env python3
"""
scFASTopic 流水线测试脚本
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """运行命令并检查结果"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ Success ({elapsed:.1f}s)")
        if result.stdout:
            print("Output:", result.stdout[-200:])  # 显示最后200字符
    else:
        print(f"❌ Failed ({elapsed:.1f}s)")
        print("Error:", result.stderr)
        return False
    
    return True

def check_files(file_patterns, description):
    """检查文件是否存在"""
    print(f"\n📁 Checking {description}")
    
    all_exist = True
    for pattern in file_patterns:
        files = list(Path('.').glob(pattern))
        if files:
            print(f"✅ Found: {files[0]}")
        else:
            print(f"❌ Missing: {pattern}")
            all_exist = False
    
    return all_exist

def main():
    """主测试流程"""
    print("🚀 scFASTopic Pipeline Test")
    print("="*60)
    
    # 测试1: 快速训练（20个topics）
    success = run_command(
        "python train.py --n_topics 20 --epochs 50 --max_cells 5000 --quiet",
        "Testing basic training (20 topics, 5000 cells)"
    )
    
    if not success:
        print("❌ Basic training failed")
        return 1
    
    # 检查生成的矩阵文件
    matrix_files = [
        "results/PBMC_cell_embedding_20.pkl",
        "results/PBMC_cell_topic_20.pkl", 
        "results/PBMC_topic_gene_20.pkl"
    ]
    
    if not check_files(matrix_files, "matrix files"):
        print("❌ Matrix files missing")
        return 1
    
    # 检查可视化文件
    viz_files = [
        "visualization/PBMC_umap_20.png",
        "visualization/PBMC_topic_analysis_20.png"
    ]
    
    if not check_files(viz_files, "visualization files"):
        print("❌ Visualization files missing")
        return 1
    
    # 检查报告文件
    report_files = ["results/PBMC_report_20.md"]
    
    if not check_files(report_files, "report files"):
        print("❌ Report files missing")  
        return 1
    
    # 测试2: 不同topic数量
    success = run_command(
        "python train.py --n_topics 10 --epochs 30 --max_cells 3000 --quiet",
        "Testing different topic number (10 topics)"
    )
    
    if not success:
        print("⚠️ Different topic test failed (non-critical)")
    
    # 测试3: 可视化模块单独测试
    success = run_command(
        "python -c \"from visualization import ScFastopicVisualizer; v = ScFastopicVisualizer(); print('✅ Visualization module OK')\"",
        "Testing visualization module import"
    )
    
    if not success:
        print("❌ Visualization module test failed")
        return 1
    
    # 测试4: Geneformer embedding模块
    success = run_command(
        "python -c \"from geneformer_embedding import GeneformerEmbedding; g = GeneformerEmbedding(); print('✅ Geneformer module OK')\"",
        "Testing Geneformer embedding module"
    )
    
    if not success:
        print("❌ Geneformer module test failed")
        return 1
    
    # 测试5: FASTopic模块
    success = run_command(
        "python -c \"from fastopic import scFASTopic, FastopicTrainer; print('✅ FASTopic modules OK')\"",
        "Testing FASTopic modules import"
    )
    
    if not success:
        print("❌ FASTopic modules test failed")
        return 1
    
    print("\n🎉 All tests passed!")
    print("\n📊 Generated files summary:")
    
    # 统计生成的文件
    result_files = list(Path('results').glob('*.pkl'))
    viz_files = list(Path('visualization').glob('*.png'))
    report_files = list(Path('results').glob('*.md'))
    
    print(f"  Matrix files: {len(result_files)}")
    print(f"  Visualization files: {len(viz_files)}")
    print(f"  Report files: {len(report_files)}")
    
    print(f"\n✅ scFASTopic pipeline is working correctly!")
    return 0

if __name__ == "__main__":
    exit(main())