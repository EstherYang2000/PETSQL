import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"  # Plotly 在浏览器中打开
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
result_dir   = args.result_dir
number       = args.number
strategy     = args.strategy
title        = args.title
subtitle     = f"{title} with {strategy}"
pic_dir      = "data/pic"
dataset_type = args.dataset_type
data_type = args.data_type
final_path = os.path.join(result_dir, "final_result_1.json")
error_path = os.path.join(result_dir, "current_error.json")

# === 2) 读取 JSON 并构造宽表 DataFrame ===
with open(final_path, "r", encoding="utf-8") as f:
    final_data = json.load(f)
with open(error_path, "r", encoding="utf-8") as f:
    curr_err = json.load(f)

records = []
for entry in final_data:
    rnd = entry["index"] + 1
    for model, m in entry["currenrt_mistakes"].items():
        records.append({"Round": rnd, "Model": model, "Mistakes": m})

df      = pd.DataFrame(records)

df_wide = df.pivot(index="Round", columns="Model", values="Mistakes").fillna(0)
rename_dict = {
    "gemini": "Gemini-2.5-Pro Experimental",
    "llamaapi_3.3": "Llama-3.3-70B",
    "qwen_api_2_5_72b": "Qwen2.5-72B",
    "qwen_api_32b-instruct-fp16": "Qwen2.5-coder-32B-instruct-fp16",
    "gpt-4o": "GPT-4o",
    "o3-mini": "o3-mini",
}
df_wide.rename(columns=rename_dict, inplace=True)
rounds = df_wide.index.values
models = df_wide.columns.tolist()

# === 3) Matplotlib 论文级静态图 ===
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
mpl_colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
for i, model in enumerate(models):
    ax.plot(rounds, df_wide[model],
            color=mpl_colors[i % len(mpl_colors)],
            lw=1.5, label=model)

# 每100回合标记错误最少的前三名
for rnd in rounds:
    if rnd % 100 == 0:
        bottom3 = df_wide.loc[rnd].nsmallest(3)
        for model, m in bottom3.items():
            ax.scatter(rnd, m,
                       color=mpl_colors[models.index(model) % len(mpl_colors)],
                       edgecolor='k', s=80, zorder=5)
            # ax.text(rnd, m, f" {model}", va='bottom', ha='left', fontsize=8)

fig.suptitle(f"Model Mistakes per Round on Spider 1.0 {data_type}",
     fontsize=16, fontweight='bold',
    x=0.5, y=1.02, ha='center'
)
ax.set_title(subtitle, loc='left', fontsize=12, pad=8)
ax.set_xlabel('Rounds', fontsize=12, labelpad=6)
ax.set_ylabel('Number of Mistakes', fontsize=12, labelpad=6)
ax.set_xticks(np.arange(0, rounds.max()+1, 200))
ax.set_yticks(np.linspace(0, df_wide.values.max()*1.05, 6))
ax.grid(which='major', linestyle='-', linewidth=0.8, color='grey', alpha=0.4)
ax.grid(which='minor', linestyle=':', linewidth=0.4, color='grey', alpha=0.2)
ax.minorticks_on()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1),
          fontsize=9, handlelength=2.0, labelspacing=0.3, markerscale=0)
fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.10)

static_path = os.path.join(pic_dir, f"mistakes_model_{dataset_type}_{number}.jpg")
fig.savefig(static_path, dpi=300, bbox_inches='tight', format='jpg')
print(f"Saved static figure to {static_path}")
plt.show()

# 将 wide 转长格式，方便 px.line
df_melt = (
    df_wide
    .reset_index()
    .melt(id_vars="Round", var_name="Model", value_name="Mistakes")
)

# === 绘制交互式折线图 ===
fig = px.line(
    df_melt,
    x="Round",
    y="Mistakes",
    color="Model",
    labels={"Mistakes":"Number of Mistakes"},
    title=f"Model Mistakes per Round on Spider 1.0 {data_type}",
    template="plotly_white",
    hover_data={"Mistakes":":.0f"}
)

# 每100回合高亮错误最少的前三名
for rnd in rounds:
    if rnd % 100 == 0:
        bottom3 = df_wide.loc[rnd].nsmallest(3)
        for model, m in bottom3.items():
            fig.add_trace(go.Scatter(
                x=[rnd],
                y=[m],
                mode="markers+text",
                marker=dict(symbol="x", size=12, color="black"),
                text=[model],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip"
            ))

# 布局美化
fig.update_layout(
    title={
        "text":f"Model Mistakes per Round on Spider 1.0 {data_type}",
        "x":0.5,"xanchor":"center","font":{"size":20}
    },
    annotations=[{
        "text": subtitle,
        "x": 0, "xref":"paper", "y":1.08, "yref":"paper",
        "showarrow":False, "align":"left",
        "font":{"size":14, "color":"gray"}
    }],
    xaxis=dict(title="Rounds", gridcolor="#EEEEEE"),
    yaxis=dict(title="Number of Mistakes", gridcolor="#EEEEEE"),
    legend=dict(title="", x=1.02, xanchor="left", y=1, font={"size":12}),
    margin=dict(l=60, r=200, t=120, b=60),
)

fig.show()