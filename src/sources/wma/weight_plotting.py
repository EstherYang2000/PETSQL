import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"  # Opens the plot in a web browser
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import argparse

parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
parser.add_argument("--result_dir", type=str, required=True, help="Directory containing the result files.")
parser.add_argument("--strategy", type=str, required=True, help="Voting strategy used.")
parser.add_argument("--title", type=str, required=True, help="Subtitle for the plot.")
parser.add_argument("--number", type=str, required=True, help="Number for the plot.")
parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset used.")
parser.add_argument("--data_type", type=str, required=True, help="Type of data (Train/Test).")
args = parser.parse_args()
# === 1) 配置路径与参数，请替换为你的实际值 ===
result_dir = args.result_dir
strategy   = args.strategy
title     = args.title
subtitle   = f"{title} with {strategy}"
number     = args.number
dataset_type =args.dataset_type
data_type = args.data_type
file_path  = os.path.join(result_dir, "results_1.json")
# ============================================

# === 2) 读取 JSON 并构造 DataFrame ===
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for entry in data:
    rnd     = entry["index"]
    weights = entry["current_weights"]
    for model, w in weights.items():
        records.append({"Round": rnd, "Model": model, "Weight": w})

df = pd.DataFrame(records)
rename_dict = {
    "gemini": "Gemini-2.5-Pro Experimental",
    "llamaapi_3.3": "Llama-3.3-70B",
    "qwen_api_2_5_72b": "Qwen2.5-72B",
    "qwen_api_32b-instruct-fp16": "Qwen2.5-coder-32B-instruct-fp16",
    "gpt-4o": "GPT-4o",
    "o3-mini": "o3-mini",
}
# === 3) Pivot & 排序，为 hover 排名准备 ===
df_pivot = df.pivot(index="Round", columns="Model", values="Weight").fillna(0)
df_pivot.rename(columns=rename_dict, inplace=True)

df_melt  = (
    df_pivot
    .reset_index()
    .melt(id_vars="Round", var_name="Model", value_name="Weight")
)
df_melt["Rank"] = df_melt.groupby("Round")["Weight"].rank(ascending=False, method="first")
df_melt.sort_values(["Round","Weight"], ascending=[True, False], inplace=True)

# === 4) 绘图 ===
fig = px.line(
    df_melt,
    x="Round",
    y="Weight",
    color="Model",
    markers=False,
    hover_data={"Model": True, "Weight": ":.4f", "Rank": True},
    template="plotly_white"
)

# 主标题
fig.update_layout(
    title={
        "text": f"Voting Strategy Model Weights on Spider 1.0 {data_type}",
        "x": 0.5, "xanchor": "center", "yanchor": "top",
        "font": {"size": 20, "family": "Arial", "color": "black"}
    },
    # 副标题为 annotation
    annotations=[{
        "text": subtitle,
        "x": 0, "xref": "paper",
        "y": 1.02, "yref": "paper",
        "showarrow": False,
        "align": "left",
        "font": {"size": 14, "family": "Arial", "color": "gray"}
    }],
    # 图例放右侧且瘦身
    legend={
        "title": "",
        "orientation": "v",
        "x": 1.02, "xanchor": "left", "y": 1,
        "font": {"size": 10},
        "traceorder": "normal"
    },
    # 留白
    margin={"l": 60, "r": 200, "t": 140, "b": 60}
)

# 轴标签与网格
fig.update_xaxes(title_text="Rounds",
                 showgrid=True, gridwidth=1, gridcolor="#EEEEEE")
fig.update_yaxes(title_text="Model Weight",
                 showgrid=True, gridwidth=1, gridcolor="#EEEEEE")

fig.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# --- 2) 构建 DataFrame ---
records = []
for entry in data:
    idx = entry["index"]
    for model, w in entry["current_weights"].items():
        records.append({"round": idx, "model": model, "weight": w})
df = pd.DataFrame(records)

# Pivot 成每列一个模型
df_pivot = df.pivot(index="round", columns="model", values="weight").fillna(0)
df_pivot.rename(columns=rename_dict, inplace=True)
rounds = df_pivot.index.values
models = df_pivot.columns.tolist()

# === 3) Matplotlib 论文风格设置 ===
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

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# 为每个模型画一条折线，使用不同线型/marker
linestyles = ['-', '--', '-.', ':']
markers     = ['o', 's', '^', 'd', 'v', '>', '<', 'p', 'h']
for i, model in enumerate(models):
    style = linestyles[i % len(linestyles)]
    m     = markers[i % len(markers)]
    ax.plot(rounds, df_pivot[model],
            linestyle=style,
            marker=m,
            markevery= max(1, len(rounds)//20),  # 稀疏 marker
            markersize=4,
            linewidth=1.2,
            label=model)

# 坐标轴与网格
ax.set_xlabel('Rounds', labelpad=6)
ax.set_ylabel('Model Weight', labelpad=6)
# 1) 主標題（Figure 層級），居中，略往上
fig.suptitle(
    f"Voting Strategy Model Weights on Spider 1.0 {data_type}",
    fontsize=16,
    fontweight='bold',
    y=1.02,      # 往上移一點
    x=0.5,
    ha='center'
)

# 2) 副標題（Axis 層級），偏左，字體小一點
ax.set_title(
    subtitle,  # 你原本的 subtitle 變量
    fontsize=12,
    loc='left',
    pad=8       # 距離上方主區留白
)# 主次刻度
ax.set_xticks(np.linspace(rounds.min(), rounds.max(), 6, dtype=int))
ax.set_yticks(np.linspace(0, df_pivot.values.max(), 5))
ax.grid(which='major', linestyle='-',  linewidth=0.6, color='grey', alpha=0.4)
ax.grid(which='minor', linestyle=':',  linewidth=0.4, color='grey', alpha=0.2)
ax.minorticks_on()

# 去掉上、右框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 留出右侧 10% 空间给图例
fig.subplots_adjust(right=0.95)

# legend 依然外置
leg = ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.00, 1.00),
    fontsize=8,
    handlelength=1.8,
    labelspacing=0.2
)

plt.savefig(f'data/pic/model_weights_{dataset_type}_{number}.png', dpi=300, bbox_inches='tight')
plt.show()






