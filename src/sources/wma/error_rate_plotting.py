import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse

# Academic paper style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.shadow": False,
    "legend.edgecolor": "black",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.axisbelow": True
})

def load_and_process_data(result_dir):
    """Load and process data from result directory"""
    final_path = os.path.join(result_dir, "final_result_1.json")
    error_path = os.path.join(result_dir, "current_error.json")
    
    try:
        with open(final_path, "r", encoding="utf-8") as f:
            final_data = json.load(f)
        with open(error_path, "r", encoding="utf-8") as f:
            current_errors = json.load(f)
        
        # Convert to numeric errors
        numeric_errors = [list(d.values())[0] for d in current_errors]
        
        # Generate rounds
        T = len(final_data)
        rounds = np.arange(1, T + 1)
        
        # Calculate metrics
        final_err = [numeric_errors[i] / rounds[i] for i in range(T)]
        best_err = [final_data[i]["best_expert_mistakes"] / rounds[i] for i in range(T)]
        regret = [(numeric_errors[i] - final_data[i]["best_expert_mistakes"]) / rounds[i] 
                  for i in range(T)]
        
        return rounds, final_err, best_err, regret
    
    except FileNotFoundError as e:
        print(f"Error: Could not find required files - {e}")
        return None, None, None, None

def create_academic_plot(methods_data, dataset_type="Spider 1.0", data_type="Test", 
                        output_dir="data/pic", figure_number="1"):
    """
    Create academic paper style plot comparing multiple methods
    
    Args:
        methods_data: dict with method names as keys and (rounds, final_err, best_err, regret) as values
        dataset_type: Dataset name for title
        data_type: Data type (Train/Test) for title
        output_dir: Directory to save the figure
        figure_number: Figure number for filename
    """
    
    # Color scheme for academic papers (colorblind-friendly)
    colors = {
        'wma': '#1f77b4',      # Blue
        'rwma': '#ff7f0e',     # Orange  
        'naive': '#2ca02c'     # Green
    }
    
    line_styles = {
        'wma': '-',       # Solid
        'rwma': '--',     # Dashed
        'naive': ':'      # Dotted
    }
    
    markers = {
        'wma': 'o', 
        'rwma': 's', 
        'naive': '^'
    }
    
    # Create figure with specific size for academic papers
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    # Plot metrics for each method
    for method_name, (rounds, final_err, best_err, regret) in methods_data.items():
        if rounds is None:
            continue
            
        color = colors.get(method_name.lower(), '#333333')
        line_style = line_styles.get(method_name.lower(), '-')
        marker = markers.get(method_name.lower(), 'o')
        
        # Subsample data points for cleaner visualization
        step = max(1, len(rounds) // 20)  # Show ~20 points maximum
        idx = np.arange(0, len(rounds), step)
        
        # Plot 1: Final Error Rate
        ax1.plot(rounds[idx], np.array(final_err)[idx], 
                color=color, linestyle=line_style, marker=marker,
                markersize=4, linewidth=2, markevery=2,
                label=method_name.upper(), alpha=0.8)
        
        # Plot 2: Best Expert Error Rate  
        ax2.plot(rounds[idx], np.array(best_err)[idx],
                color=color, linestyle=line_style, marker=marker,
                markersize=4, linewidth=2, markevery=2,
                label=method_name.upper(), alpha=0.8)
        
        # Plot 3: Average Regret
        ax3.plot(rounds[idx], np.array(regret)[idx],
                color=color, linestyle=line_style, marker=marker,
                markersize=4, linewidth=2, markevery=2,
                label=method_name.upper(), alpha=0.8)
    
    # Configure subplots
    subplot_configs = [
        (ax1, "Final Error Rate", "Final Error Rate"),
        (ax2, "Best Expert Error Rate", "Best Expert Error Rate"), 
        (ax3, "Average Regret", "Average Regret")
    ]
    
    for ax, title, ylabel in subplot_configs:
        # Grid
        ax.grid(True, linestyle='-', linewidth=0.3, color='gray', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.2, color='gray', alpha=0.4)
        ax.minorticks_on()
        
        # Labels and title
        ax.set_xlabel('Rounds', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(ord("a") + subplot_configs.index((ax, title, ylabel)))}) {title}', 
                    fontsize=12, fontweight='bold', pad=10)
        
        # Tick formatting
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # Set reasonable axis limits
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Legend (only on first subplot)
        if ax == ax1:
            legend = ax.legend(loc='upper right', fontsize=9, 
                             framealpha=0.9, fancybox=False)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(0.8)
        if ax == ax2:
            legend = ax.legend(loc='upper right', fontsize=9, 
                             framealpha=0.9, fancybox=False)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(0.8)
        if ax == ax3:
            legend = ax.legend(loc='upper right', fontsize=9, 
                             framealpha=0.9, fancybox=False)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(0.8)
    
    # Main title
    fig.suptitle(f'Voting Error Bound Analysis on {dataset_type} {data_type} Dataset', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95, 
                       wspace=0.25, hspace=0.3)
    
    # # Save figure
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"voting_comparison_{figure_number}.pdf")
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', 
    #            facecolor='white', edgecolor='none')
    
    # Also save as PNG for preview
    png_path = os.path.join(output_dir, f"voting_comparison_{figure_number}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"Saved academic figure to:")
    # print(f"  PDF: {output_path}")
    print(f"  PNG: {png_path}")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create academic comparison plot for voting methods.")
    parser.add_argument("--wma_dir", type=str, help="Directory containing WMA results.")
    parser.add_argument("--rwma_dir", type=str, help="Directory containing RWMA results.")  
    parser.add_argument("--naive_dir", type=str, help="Directory containing Naive results.")
    parser.add_argument("--dataset_type", type=str, default="Spider 1.0", help="Dataset name.")
    parser.add_argument("--data_type", type=str, default="Test", help="Data type (Train/Test).")
    parser.add_argument("--output_dir", type=str, default="data/pic", help="Output directory.")
    parser.add_argument("--figure_number", type=str, default="1", help="Figure number.")
    
    args = parser.parse_args()
    
    # Load data for each method
    methods_data = {}
    
    if args.wma_dir:
        methods_data['WMA'] = load_and_process_data(args.wma_dir)
    if args.rwma_dir:
        methods_data['RWMA'] = load_and_process_data(args.rwma_dir)
    if args.naive_dir:
        methods_data['Naive'] = load_and_process_data(args.naive_dir)
    
    # Create the plot
    if methods_data:
        create_academic_plot(methods_data, args.dataset_type, args.data_type,
                           args.output_dir, args.figure_number)
    else:
        print("No valid data directories provided. Please specify at least one of --wma_dir, --rwma_dir, or --naive_dir")
        
        # Demo with synthetic data
        print("\nGenerating demo plot with synthetic data...")
        demo_rounds = np.arange(1, 1001)
        demo_methods = {
            'WMA': (demo_rounds, 
                   0.5 * np.exp(-demo_rounds/500) + 0.1 + 0.05*np.random.randn(1000)*np.exp(-demo_rounds/300),
                   0.3 * np.exp(-demo_rounds/400) + 0.05 + 0.03*np.random.randn(1000)*np.exp(-demo_rounds/200), 
                   0.2 * np.exp(-demo_rounds/600) + 0.05 + 0.02*np.random.randn(1000)*np.exp(-demo_rounds/400)),
            'RWMA': (demo_rounds,
                    0.6 * np.exp(-demo_rounds/600) + 0.12 + 0.06*np.random.randn(1000)*np.exp(-demo_rounds/350),
                    0.35 * np.exp(-demo_rounds/450) + 0.06 + 0.04*np.random.randn(1000)*np.exp(-demo_rounds/250),
                    0.25 * np.exp(-demo_rounds/700) + 0.06 + 0.03*np.random.randn(1000)*np.exp(-demo_rounds/450)),
            'Naive': (demo_rounds,
                     0.8 * np.exp(-demo_rounds/400) + 0.15 + 0.08*np.random.randn(1000)*np.exp(-demo_rounds/250),
                     0.4 * np.exp(-demo_rounds/350) + 0.08 + 0.05*np.random.randn(1000)*np.exp(-demo_rounds/180),
                     0.4 * np.exp(-demo_rounds/500) + 0.07 + 0.04*np.random.randn(1000)*np.exp(-demo_rounds/350))
        }
        
        create_academic_plot(demo_methods, "Spider 1.0", "Test", "data/pic", "demo")

    """
    
python src/sources/wma/error_rate_plotting.py \
                --wma_dir bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base_wma_6 \
                --rwma_dir bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base_rwma_6 \
                --naive_dir bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base_naive_6 \
                 --dataset_type "BIRD" \
                --data_type "Dev" \
                --output_dir "data/pic/bird_dev/6/1_baseline" \
                --figure_number "1"
    """