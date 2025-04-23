import json
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"  # 在浏览器中打开
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
parser.add_argument("--result_dir", type=str, required=True, help="Directory containing the result files.")
parser.add_argument("--strategy", type=str, required=True, help="Voting strategy used.")
parser.add_argument("--title", type=str, required=True, help="Subtitle for the plot.")
parser.add_argument("--number", type=str, required=True, help="Number for the plot.")
parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset used.")
parser.add_argument("--data_type", type=str, required=True, help="Type of data (Train/Test).")
args = parser.parse_args()
# === 1) 配置路径与参数 ===
result_dir = args.result_dir
strategy   = args.strategy
title     = args.title
subtitle   = f"{title} with {strategy}"
number    = args.number
pic_dir = "data/pic"
dataset_type = args.dataset_type
data_type = args.data_type
N = 6
# ============================================
final_path = os.path.join(result_dir, "final_result_1.json")
error_path = os.path.join(result_dir, "current_error.json")

# === 2) 读入 JSON ===
with open(final_path, "r", encoding="utf-8") as f:
    final_data = json.load(f)
with open(error_path, "r", encoding="utf-8") as f:
    current_errors = json.load(f)

# === 3) 准备 DataFrame ===
numeric_errors = [list(d.values())[0] for d in current_errors]
records = []
for idx, rec in enumerate(final_data):
    rnd    = idx + 1
    M_star = rec["best_expert_mistakes"]
    err    = numeric_errors[idx]
    # 平均遗憾 = (M_t - M*_t) / t
    records.append({"Round": rnd, "Metric": "Final Error Rate",       "Value": err / rnd})
    records.append({"Round": rnd, "Metric": "Best Expert Error Rate", "Value": M_star / rnd})
    records.append({"Round": rnd, "Metric": "Average Regret",         "Value": (err - M_star) / rnd})

df = pd.DataFrame(records)

# === 4) 绘图 ===
fig = px.line(
    df,
    x="Round",
    y="Value",
    color="Metric",
    markers=False,
    hover_data={"Metric": True, "Value":":.4f"},
    template="plotly_white"
)

# === 5) 布局美化 ===
fig.update_layout(
    # 主标题
    title={
        "text": f"Voting Error Bound Rates of PETSQL Pipeline Variants on Spider 1.0 {data_type}",
        "x": 0.5, "xanchor": "center", "yanchor": "top",
        "font": {"size": 20, "family": "Arial", "color": "black"}
    },
    # 副标题注释
    annotations=[{
        "text": subtitle,
        "x": 0, "xref": "paper",
        "y": 1.03, "yref": "paper",
        "showarrow": False,
        "align": "left",
        "font": {"size": 14, "family": "Arial", "color": "gray"}
    }],
    # 轴标签
    xaxis_title="Rounds",
    yaxis_title="Error Rate",
    # 图例瘦身并外置
    legend={
        "title": "", "orientation": "v",
        "x": 1.02, "xanchor": "left", "y": 1,
        "font": {"size": 12, "family": "Arial"},
        "borderwidth": 0
    },
    # 留白给图例和标题
    margin={"l": 60, "r": 200, "t": 140, "b": 60},
)

# === 6) 网格细节 ===
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#EEEEEE", zeroline=False)
fig.show()

# 转成纯数字列表
numeric_errors = [list(d.values())[0] for d in current_errors]

# 生成回合索引 1..T
T = len(final_data)
rounds = np.arange(1, T + 1)

# Final Error Rate = M_t / t
final_err = [numeric_errors[i] / rounds[i] for i in range(T)]
# Best Expert Error Rate = M*_t / t
best_err  = [final_data[i]["best_expert_mistakes"] / rounds[i] for i in range(T)]
# Average Regret = (M_t - M*_t) / t
regret    = [(numeric_errors[i] - final_data[i]["best_expert_mistakes"]) / rounds[i]
             for i in range(T)]

# === 4) Matplotlib 论文级风格设置 ===
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
})

# === 5) 创建画布 ===
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 绘制三条曲线
ax.plot(rounds, final_err,
        color='tab:blue', lw=1.8,
        label='Final Error Rate')
ax.plot(rounds, best_err,
        color='tab:orange', lw=1.8, linestyle='--',
        label='Best Expert Error Rate')
ax.plot(rounds, regret,
        color='tab:green', lw=1.8, linestyle=':',
        label='Average Regret')

# === 6) 主标题 & 副标题 ===
fig.suptitle(
    f"Voting Error Bound Rates of PETSQL Pipeline Variants on Spider 1.0 {data_type}",
    fontsize=16, fontweight='bold',
    x=0.5, y=1.02, ha='center'
)
ax.set_title(subtitle, loc='left', fontsize=12, pad=8)

# === 7) 刻度与网格 ===
ax.set_xticks(np.arange(0, T+1, 200))
ax.set_yticks(np.linspace(0, max(final_err + best_err + regret) * 1.05, 6))
ax.grid(which='major', linestyle='-',  linewidth=0.8, color='grey', alpha=0.4)
ax.grid(which='minor', linestyle=':',  linewidth=0.4, color='grey', alpha=0.2)
ax.minorticks_on()

# === 8) 坐标轴标签 & 去除多余边框 ===
ax.set_xlabel('Rounds', fontsize=12, labelpad=6)
ax.set_ylabel('Error Rate', fontsize=12, labelpad=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === 9) 图例外置 & 瘦身 ===
leg = ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    fontsize=9,
    handlelength=2.0,
    labelspacing=0.3,
    markerscale=0  # 隐藏 marker
)

# === 10) 布局调整：主区占 90%，右侧 10% 留给图例，上方留白给主标题 ===
fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.10)

# === 11) 保存为高分辨率图像 ===
output_path = f"{pic_dir}/error_bound_{dataset_type}_{number}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_path}")

# === 12) 展示图像 ===
plt.show()