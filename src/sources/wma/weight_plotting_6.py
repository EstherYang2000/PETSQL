import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Rectangle
import argparse

# 配置实验阶段和对应的目录路径
EXPERIMENT_STAGES = {
    1: {
        'name': 'Baseline PETSQL\n(No SL, No SR)',
        'dirs': {
            'WMA': 'data/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_wma',
            'RWMA': 'data/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_rwma',
            'Naive': 'data/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_naive'
        }
    },
    2: {
        'name': 'Baseline + Schema Linking\n(SL, No SR)',
        'dirs': {
            'WMA': 'data/vote/202504/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_base_wma',
            'RWMA': 'data/vote/202504/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_base_rwma',
            'Naive': 'data/vote/202504/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_base_naive'
        }
    },
    3: {
        'name': 'Baseline + Self-Refinement\n(No SL, SR)',
        'dirs': {
            'WMA': 'data/vote/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_rf_wma',
            'RWMA': 'data/vote/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_rf_rwma',
            'Naive': 'data/vote/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_rf_naive'
        }
    },
    4: {
        'name': 'Baseline + SR + SL\n(SL, SR)',
        'dirs': {
            'WMA': 'data/vote/202504/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_wma',
            'RWMA': 'data/vote/202504/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_rwma',
            'Naive': 'data/vote/202504/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_naive'
        }
    },
    5: {
        'name': 'SR w/o SL in Voting\n(SR, No SL in Vote)',
        'dirs': {
            'WMA': 'data/vote/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_rf_wma',
            'RWMA': 'data/vote/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_rf_rwma',
            'Naive': 'data/vote/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_rf_naive'
        }
    },
    6: {
        'name': 'SR w/ SL in Voting\n(SR, SL in Vote)',
        'dirs': {
            'WMA': 'data/vote/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_wma_6',
            'RWMA': 'data/vote/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_rwma_6',
            'Naive': 'data/vote/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_naive_6'
        }
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare 6 experimental stages with 3 voting strategies each.")
    parser.add_argument("--dataset_type", type=str, default="Spider 1.0", help="Type of dataset used.")
    parser.add_argument("--data_type", type=str, default="Dev", help="Type of data (Train/Test/Dev).")
    parser.add_argument("--output_dir", type=str, default="data/pic", help="Output directory for plots.")
    parser.add_argument("--figure_number", type=str, default="6x3_comparison", help="Figure number for file naming.")
    return parser.parse_args()

def load_strategy_data(result_dir):
    """加载单个策略的数据"""
    file_path = os.path.join(result_dir, "results_1.json")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        for entry in data:
            rnd = entry["index"]
            weights = entry["current_weights"]
            for model, w in weights.items():
                records.append({
                    "Round": rnd, 
                    "Model": model, 
                    "Weight": w
                })
        
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def rename_models(df):
    """重命名模型"""
    rename_dict = {
        "gemini": "Gemini-2.5-Pro",
        "llamaapi_3.3": "Llama-3.3-70B",
        "qwen_api_2_5_72b": "Qwen2.5-72B",
        "qwen_api_32b-instruct-fp16": "Qwen2.5-coder-32B",
        "gpt-4o": "GPT-4o",
        "o3-mini": "o3-mini",
    }
    df['Model'] = df['Model'].map(rename_dict).fillna(df['Model'])
    return df

def plot_6x3_comparison(stages_data, dataset_type, data_type, output_dir, figure_number):
    """创建6x3的比较图"""
    
    # 设置matplotlib参数
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
    })
    
    # 创建6x3子图
    fig, axes = plt.subplots(6, 3, figsize=(15, 20), dpi=300)
    
    # 线条样式和标记
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd', 'v', '>', '<', 'p']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    strategies = ['WMA', 'RWMA', 'Naive']
    
    # 用于存储所有模型名称，以便统一图例
    all_models = set()
    
    for stage_idx in range(6):
        stage_num = stage_idx + 1
        stage_info = EXPERIMENT_STAGES[stage_num]
        
        for strategy_idx, strategy in enumerate(strategies):
            ax = axes[stage_idx, strategy_idx]
            
            # 获取数据
            result_dir = stage_info['dirs'][strategy]
            df = load_strategy_data(result_dir)
            
            if df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{strategy}', fontsize=10, fontweight='bold')
                continue
            
            df = rename_models(df)
            df_pivot = df.pivot(index="Round", columns="Model", values="Weight").fillna(0)
            
            if df_pivot.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{strategy}', fontsize=10, fontweight='bold')
                continue
            
            rounds = df_pivot.index.values
            models = df_pivot.columns.tolist()
            all_models.update(models)
            
            # 绘制每个模型的折线
            for i, model in enumerate(models):
                style = linestyles[i % len(linestyles)]
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                ax.plot(rounds, df_pivot[model],
                       linestyle=style,
                       marker=marker,
                       markevery=max(1, len(rounds)//10),
                       markersize=3,
                       linewidth=1,
                       color=color,
                       label=model,
                       alpha=0.8)
            
            # 设置标题
            if stage_idx == 0:  # 只在第一行显示策略名
                ax.set_title(f'{strategy}', fontsize=11, fontweight='bold', pad=8)
            
            # 设置坐标轴标签
            if stage_idx == 5:  # 只在最后一行显示x轴标签
                ax.set_xlabel('Rounds', fontsize=9)
            if strategy_idx == 0:  # 只在第一列显示y轴标签
                ax.set_ylabel('Weight', fontsize=9)
            
            # 设置网格
            ax.grid(which='major', linestyle='-', linewidth=0.4, color='grey', alpha=0.3)
            ax.minorticks_on()
            ax.grid(which='minor', linestyle=':', linewidth=0.2, color='grey', alpha=0.2)
            
            # 设置坐标轴范围和刻度
            if len(rounds) > 1:
                ax.set_xlim(rounds.min(), rounds.max())
                n_ticks = min(5, len(rounds))
                ax.set_xticks(np.linspace(rounds.min(), rounds.max(), n_ticks, dtype=int))
            
            max_weight = df_pivot.values.max()
            ax.set_ylim(0, max_weight * 1.05)
            ax.set_yticks(np.linspace(0, max_weight, 5))
            
            # 去掉上、右框线
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 在每行图的正中间下方添加阶段标签
        fig.text(0.5, 0.82 - stage_idx * 0.143, f'Stage {stage_num}: {stage_info["name"].replace(chr(10), " ")}', 
                fontsize=11, fontweight='bold', ha='center', va='top', color='blue')
    
    # 设置总标题
    fig.suptitle(f'Voting Strategy Comparison Across 6 Experimental Stages\nDataset: {dataset_type} {data_type}', 
                fontsize=16, fontweight='bold', y=0.965, x=0.5)
    
    # 创建统一图例
    if all_models:
        # 从第一个有数据的子图获取图例
        handles, labels = [], []
        for stage_idx in range(6):
            for strategy_idx in range(3):
                ax = axes[stage_idx, strategy_idx]
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break
        
        if handles:
            fig.legend(handles, labels,
                      loc='center left',
                      bbox_to_anchor=(0.92, 0.5),
                      fontsize=9,
                      handlelength=1.5,
                      labelspacing=0.4,
                      title='Models',
                      title_fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.05, hspace=0.4, wspace=0.25)
    
    # 保存图片
    filename = os.path.join(output_dir, f'voting_comparison_6x3_{figure_number}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved 6x3 comparison chart: {filename}")
    plt.show()

def main():
    args = parse_arguments()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载所有阶段的数据
    stages_data = {}
    for stage_num, stage_info in EXPERIMENT_STAGES.items():
        stages_data[stage_num] = {}
        for strategy, result_dir in stage_info['dirs'].items():
            df = load_strategy_data(result_dir)
            if not df.empty:
                df = rename_models(df)
            stages_data[stage_num][strategy] = df
            print(f"Loaded Stage {stage_num} {strategy}: {len(df)} records")
    
    # 创建6x3比较图
    plot_6x3_comparison(stages_data, args.dataset_type, args.data_type, 
                       args.output_dir, args.figure_number)
    
    print(f"\nChart saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
"""

python src/sources/wma/weight_plotting_6.py \
  --dataset_type "Spider 1.0" \
  --data_type "Dev" \
  --output_dir "data/pic" \
  --figure_number "6x3_comparison"
"""