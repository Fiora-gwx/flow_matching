#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def visualize_results(csv_path, output_dir):
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 过滤掉失败的实验
    if 'status' in df.columns:
        df = df[df['status'] == 'completed']

    # 如果有多次重复实验，取相同条件下的最小 FID (最好结果)
    df = df.groupby(['alpha', 'epoch', 'nfe'], as_index=False)['fid'].min()

    # 只看 epoch 500 (或 499，根据你的实际保存逻辑) 
    # 为了鲁棒性，这里取数据中存在的最大学习轮数
    max_epoch = df['epoch'].max()
    print(f"Generating plots for Epoch {max_epoch}...")
    df_final = df[df['epoch'] == max_epoch].copy()

    alphas = sorted(df_final['alpha'].unique())
    # 移除 0.5 以便为其单独分配颜色，其余的用 colormap
    other_alphas = [a for a in alphas if a != 0.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(other_alphas)))
    color_map = {a: c for a, c in zip(other_alphas, colors)}
    color_map[0.5] = 'red' # 强制 Base 为红色

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============ 图1: FID vs NFE 曲线 ============
    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        data_alpha = df_final[df_final['alpha'] == alpha].sort_values('nfe')
        if data_alpha.empty: continue
        
        label = f'α={alpha:.2f}'
        if alpha == 0.5:
            label += ' (Base FM)'
            plt.plot(data_alpha['nfe'], data_alpha['fid'], 
                    'o-', linewidth=3, markersize=10, 
                    color=color_map[alpha], label=label, zorder=10)
        else:
            plt.plot(data_alpha['nfe'], data_alpha['fid'], 
                    'o-', linewidth=2, markersize=8,
                    color=color_map[alpha], label=label, alpha=0.8)

    plt.xlabel('Number of Function Evaluations (NFE)', fontsize=14)
    plt.ylabel('FID ↓', fontsize=14)
    plt.title(f'FID vs NFE for Different α (Epoch {max_epoch})', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    nfe_ticks = sorted(df_final['nfe'].unique())
    plt.xticks(nfe_ticks, [str(int(x)) for x in nfe_ticks])
    plt.tight_layout()
    plt.savefig(output_dir / 'fid_vs_nfe_all_alpha.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 图2: 热力图 ============
    try:
        pivot_data = df_final.pivot(index='alpha', columns='nfe', values='fid')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                    cbar_kws={'label': 'FID ↓'})
        plt.xlabel('NFE', fontsize=14)
        plt.ylabel('α', fontsize=14)
        plt.title(f'FID Heatmap: α vs NFE (Epoch {max_epoch})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'fid_heatmap_alpha_nfe.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not generate heatmap: {e}")

    # ============ 图3: 相对改善 ============
    base_data = df_final[df_final['alpha'] == 0.5]
    if not base_data.empty:
        base_fid = base_data.set_index('nfe')['fid']
        plt.figure(figsize=(10, 6))

        for alpha in alphas:
            if alpha == 0.5: continue
            
            data_alpha = df_final[df_final['alpha'] == alpha].set_index('nfe')['fid']
            
            # 只计算 shared NFE 的相对提升
            common_nfes = base_fid.index.intersection(data_alpha.index)
            if common_nfes.empty: continue
            
            improvement = (base_fid[common_nfes] - data_alpha[common_nfes]) / base_fid[common_nfes] * 100 
            
            plt.plot(common_nfes, improvement.values, 'o-', linewidth=2, markersize=8,
                    color=color_map[alpha], label=f'α={alpha:.2f}', alpha=0.8)

        plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Base FM', zorder=10)
        plt.xlabel('NFE', fontsize=14)
        plt.ylabel('Relative Improvement vs Base (%)', fontsize=14)
        plt.title(f'FID Improvement Relative to Base FM (Epoch {max_epoch})', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.xticks(nfe_ticks, [str(int(x)) for x in nfe_ticks])
        plt.axhspan(-5, 5, alpha=0.1, color='gray', label='±5% (noise)')
        plt.tight_layout()
        plt.savefig(output_dir / 'relative_improvement_vs_base.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ============ 图4: 4步 vs 100步 Pareto ============
    nfe_low = 4
    nfe_high = 100
    
    # 检查是否存在这两个 NFE 的数据
    if nfe_low in df_final['nfe'].values and nfe_high in df_final['nfe'].values:
        nfe_4_data = df_final[df_final['nfe'] == nfe_low].set_index('alpha')['fid']
        nfe_100_data = df_final[df_final['nfe'] == nfe_high].set_index('alpha')['fid']

        plt.figure(figsize=(8, 6))
        for alpha in alphas:
            if alpha in nfe_100_data.index and alpha in nfe_4_data.index:
                x = nfe_100_data.loc[alpha]
                y = nfe_4_data.loc[alpha]
                
                if alpha == 0.5:
                    plt.scatter(x, y, s=300, c='red', marker='*', 
                                edgecolors='black', linewidths=1.5, zorder=10,
                                label=f'α={alpha:.2f} (Base)')
                else:
                    plt.scatter(x, y, s=150, alpha=0.8, color=color_map[alpha])
                
                plt.annotate(f'{alpha:.2f}', (x, y), 
                            xytext=(8, 8), textcoords='offset points',
                            fontsize=10, fontweight='bold')

        plt.xlabel(f'FID @ {nfe_high} steps ↓', fontsize=14)
        plt.ylabel(f'FID @ {nfe_low} steps ↓', fontsize=14)
        plt.title('Trade-off: Few-step vs Many-step', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'pareto_4step_vs_100step.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ============ 统计打印 ============
    print("\n" + "="*60)
    print(f"📊 STATISTICAL ANALYSIS (Epoch {max_epoch})")
    print("="*60)

    if not base_data.empty:
        for nfe in nfe_ticks:
            print(f"\n--- NFE = {nfe} ---")
            if nfe not in base_fid.index:
                print("Base data missing for this NFE.")
                continue
                
            b_val = base_fid.loc[nfe]
            print(f"Base FM: {b_val:.2f}")
            
            best_alpha, best_fid, best_imp = None, float('inf'), -float('inf')
            better_count = 0
            
            for alpha in alphas:
                if alpha == 0.5: continue
                subset = df_final[(df_final['alpha']==alpha) & (df_final['nfe']==nfe)]
                if subset.empty: continue
                
                val = subset['fid'].values[0]
                imp = (b_val - val) / b_val * 100
                
                if val < b_val: better_count += 1
                if val < best_fid:
                    best_fid, best_alpha, best_imp = val, alpha, imp
            
            if best_alpha is not None:
                print(f"Best α={best_alpha:.2f}: FID={best_fid:.2f} (Δ={best_imp:+.1f}%)")
                print(f"Better than base: {better_count}/{len(alphas)-1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv")
    parser.add_argument("--out", type=str, default="./plots", help="Output directory for plots")
    args = parser.parse_args()
    visualize_results(args.csv, args.out)