
# -*- coding: utf-8 -*-
import os, sys, argparse, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
ALGOS={'sheer': ('algs.sheer','SHEER'),'actar': ('algs.actar','ACTAR'),
       'tfsm_dpc': ('algs.tfsm_dpc','TFSM_DPC'),'dst_woa': ('algs.dst_woa','DST_WOA'),
       'trail': ('algs.trail','TRAIL')}
# === 指标配置（名称、是否百分比、单位、派生/换算） ===
METRIC_SPECS = {
    # 可靠性
    "pdr":                      {"label": "PDR",                           "percent": True,  "unit": "%"},
    "drop_rate":                {"label": "Drop Rate",                     "percent": True,  "unit": "%"},
    "timely_transfer_rate":     {"label": "Timely Transfer Rate (Cond.)",  "percent": True,  "unit": "%"},
    "timely_overall":           {"label": "Overall Timely Rate",           "percent": True,  "unit": "%"},  # NEW (⑥)

    # 寿命
    "FND":                      {"label": "FND (Rounds)",                  "percent": False, "unit": "rounds"},
    "HND":                      {"label": "HND (Rounds)",                  "percent": False, "unit": "rounds"},
    "LND":                      {"label": "LND (Rounds)",                  "percent": False, "unit": "rounds"},
    "func_life_pdr85":          {"label": "Functional Lifetime (PDR≥0.85)","percent": False, "unit": "rounds"},

    # 能耗/吞吐（派生列）
    "energy_mJ_per_pkt": {"label": "Energy per Delivered",                 "percent": False, "unit": "mJ/bit"},
    "throughput_kbit_round":    {"label": "Throughput",                    "percent": False, "unit": "kbit/round"},
    "bits_per_joule":           {"label": "Bits per Joule (BPJ)",          "percent": False, "unit": "bit/J"},  # NEW (③)
    "energy_rate":              {"label": "Energy Rate",                   "percent": True,  "unit": "%"},

    # 拓扑/结构
    "avg_hops_to_bs":           {"label": "Average Hops to BS",            "percent": False, "unit": ""},
    "avg_ch_per_round":         {"label": "Avg CH per Round",              "percent": False, "unit": ""},
    "avg_cluster_size":         {"label": "Average Cluster Size",          "percent": False, "unit": ""},

    # 安全/黑名单
    "malicious_drop":           {"label": "Malicious Drops",               "percent": False, "unit": "pkts"},
    "malicious_delay":          {"label": "Malicious Delay",               "percent": False, "unit": "pkts"},
    "blacklisted_malicious":    {"label": "Blacklisted Malicious",         "percent": False, "unit": "nodes"},
    "blacklisted_normal":       {"label": "Blacklisted Normal (FP)",       "percent": False, "unit": "nodes"},
    "false_blacklist_events":   {"label": "False Blacklist Events",        "percent": False, "unit": "events"},

    # 控制开销（派生/论文向）
    "control_overhead_kbit":    {"label": "Control Overhead",              "percent": False, "unit": "kbit"},
    "overhead_ratio":           {"label": "Overhead Ratio",                "percent": False, "unit": ""},      # NEW (④)

    # 相对基线（SHEER）——新增 (⑤)
    "pdr_rel_impr_pct":         {"label": "PDR Rel. Improvement vs Baseline",      "percent": True,  "unit": "%"},
    "bpj_rel_impr_pct":         {"label": "BPJ Rel. Improvement vs Baseline",      "percent": True,  "unit": "%"},
    "overhead_rel_red_pct":     {"label": "Overhead Reduction vs Baseline",        "percent": True,  "unit": "%"},
}



def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """派生/换算一些更易读的列，不改变原 df"""
    d = df.copy()

    # 单位换算
    if "energy_per_delivered" in d.columns and "energy_mJ_per_pkt" not in d.columns:
        d["energy_mJ_per_pkt"] = d["energy_per_delivered"] * 1e3  # J/bit -> mJ/bit
    if "throughput_bits_per_round" in d.columns and "throughput_kbit_round" not in d.columns:
        d["throughput_kbit_round"] = d["throughput_bits_per_round"] / 1000.0
    if "control_overhead_bits" in d.columns and "control_overhead_kbit" not in d.columns:
        d["control_overhead_kbit"] = d["control_overhead_bits"] / 1000.0

    # === 论文向派生列 ===
    # (③) BPJ: bit / J
    if "energy_per_delivered" in d.columns and "bits_per_joule" not in d.columns:
        d["bits_per_joule"] = 1.0 / d["energy_per_delivered"].replace(0, np.nan)

    # (⑥) Overall timely: 端到端按时率 = PDR × 条件及时率
    if {"pdr", "timely_transfer_rate"} <= set(d.columns) and "timely_overall" not in d.columns:
        d["timely_overall"] = d["pdr"] * d["timely_transfer_rate"]

    # (④) Overhead Ratio: 控制比特 / (有效载荷比特)
    # 有效载荷比特近似 = 每轮吞吐 × 轮数
    if {"control_overhead_bits", "throughput_bits_per_round", "rounds_run"} <= set(d.columns) and "overhead_ratio" not in d.columns:
        denom = (d["throughput_bits_per_round"] * d["rounds_run"]).replace(0, np.nan)
        d["overhead_ratio"] = d["control_overhead_bits"] / denom

    return d


def get_algo_ctor(tag:str):
    mod_name, cls = ALGOS[tag]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls)

def run_one(tag:str, n_nodes:int, n_malicious:int, seed:int, rounds:int, until_dead:bool, out_dir:str, suffix:str=""):
    from core.wsn_core import Simulation
    ctor = get_algo_ctor(tag)
    sim = Simulation(ctor, n_nodes=n_nodes, n_malicious=n_malicious, seed=seed)
    if until_dead:
        res, hist = sim.run(rounds=0)
    else:
        res, hist = sim.run(rounds=rounds)
    if suffix:
        hist.to_csv(os.path.join(out_dir, f'hist_{suffix}_{tag}.csv'), index=False)
    else:
        hist.to_csv(os.path.join(out_dir, f'hist_{tag}.csv'), index=False)
    res['seed'] = seed
    # 统一展示名（与 ALGOS 的第二列一致）
    res['algo'] = ALGOS[tag][1]   # e.g., 'SHEER','ACTAR','TFSM_DPC','DST_WOA','TRAIL'
    return res


def plot_bar_with_table_highlight(
    df,
    xcol: str,
    ycol_mean: str,
    ycol_ci95: str,
    ncol: str,
    title: str,
    ylabel: str,
    out_path: str,
    mode: str = "max"   # "max"：取最大为最佳；"closest_to_zero"：离0最近为最佳（用于斜率）
):

    """
    生成“柱状图 + 表格”的合成图，并高亮最佳算法。
    - df: 含统计列的 DataFrame（如 robustness_slope.csv 或 pdr_auc_ratio.csv）
    - xcol: 类别列（通常为 'algo'）
    - ycol_mean: 指标均值列（如 'pdr_slope_mean' 或 'auc_norm_mean'）
    - ycol_ci95: 95%CI 列
    - ncol: 样本量列（如 'n'）
    - mode: 'max' 取最大为最佳；'closest_to_zero' 用 |值| 最小为最佳（适合“斜率越接近0越好”）
    """
    if df is None or len(df) == 0:
        print(f"[WARN] plot_bar_with_table_highlight: empty df, skip {out_path}")
        return

    # 排序 & 找到最佳
    if mode == "closest_to_zero":
        # 表格按“离0最近”从好到差排列
        df_plot = df.copy().assign(_rank=np.abs(df[ycol_mean].to_numpy()))
        df_plot = df_plot.sort_values("_rank", ascending=True).drop(columns="_rank")
        best_mask = (np.abs(df_plot[ycol_mean].to_numpy()) ==
                     np.abs(df_plot[ycol_mean].to_numpy()).min())
    else:
        # 表格按均值从大到小排列
        df_plot = df.sort_values(ycol_mean, ascending=False).copy()
        best_mask = (df_plot[ycol_mean].to_numpy() == df_plot[ycol_mean].max())

    # 准备画布（上：柱状图；下：表格）
    fig = plt.figure(figsize=(9, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 2.0])
    ax_bar = fig.add_subplot(gs[0])
    ax_tab = fig.add_subplot(gs[1])

    # --- 柱状图 ---
    xs = np.arange(len(df_plot))
    ys = df_plot[ycol_mean].to_numpy()
    yerr = df_plot[ycol_ci95].to_numpy() if (ycol_ci95 in df_plot.columns) else None

    bars = ax_bar.bar(xs, ys)
    if yerr is not None and not np.all(np.isnan(yerr)):
        ax_bar.errorbar(xs, ys, yerr=yerr, fmt='none', capsize=3)

    # 高亮最佳：给最佳柱子加星标 & 加粗 x 轴标签
    for i, is_best in enumerate(best_mask):
        if is_best:
            ax_bar.text(xs[i], ys[i], "★", ha='center', va='bottom', fontsize=14)
    labels = [f"{a}" + (" ★" if is_best else "") for a, is_best in zip(df_plot[xcol], best_mask)]
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(labels, rotation=15)
    ax_bar.set_ylabel(ylabel)
    ax_bar.set_title(title)
    # 辅助线（斜率图用0线更直观）
    if mode == "closest_to_zero":
        ax_bar.axhline(0.0, linewidth=1)
    ax_bar.grid(True, axis='y', linewidth=0.5, alpha=0.5)

    # --- 表格 ---
    # 将均值±CI 与 n 生成字符串；数值格式更论文向
    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "–"
        return f"{v:.3f}"

    table_rows = []
    for _, row in df_plot.iterrows():
        mean_s = _fmt(row[ycol_mean])
        ci_s   = _fmt(row[ycol_ci95]) if ycol_ci95 in df_plot.columns else "–"
        n_s    = str(int(row[ncol])) if ncol in df_plot.columns and not np.isnan(row[ncol]) else "–"
        table_rows.append([row[xcol], mean_s, ci_s, n_s])

    col_labels = [xcol, "mean", "95%CI", "n"]
    ax_tab.axis('off')
    tbl = ax_tab.table(cellText=table_rows, colLabels=col_labels, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)

    # 高亮最佳行（背景浅黄 & 加粗）
    best_row_idx = np.where(best_mask)[0][0]  # 若有并列，会取第一个
    # +1 因为第0行是表头
    for j in range(len(col_labels)):
        cell = tbl[(best_row_idx + 1, j)]
        cell.set_facecolor('#fff7cc')
        cell.set_text_props(fontweight='bold')

    fig.tight_layout()
    # 保存 PNG
    plt.savefig(out_path, dpi=240, bbox_inches='tight')
    # 额外保存 SVG / PDF（同名不同扩展名）
    base, _ = os.path.splitext(out_path)
    plt.savefig(base + ".svg", dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f"[图] 合成版已生成：{out_path}, {base + '.svg'}, {base + '.pdf'}")
# >>> NEW: 统计辅助——按 (algo, seed) 把各 ratio 取均值，避免比值分布权重不一致
def _mean_by_seed(df_all: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = ['algo', 'seed'] + cols
    sub  = df_all[keep].copy()
    return (sub.groupby(['algo','seed'])[cols]
                .mean()
                .reset_index())

# >>> NEW: ① 控制开销比合成图（越低越好）
def plot_overhead_bar(df_all: pd.DataFrame, out_dir: str):
    by_seed = _mean_by_seed(df_all, ['overhead_ratio'])
    stat = (by_seed.groupby('algo', as_index=False)
                  .agg(overhead_ratio_mean=('overhead_ratio','mean'),
                       overhead_ratio_std =('overhead_ratio','std'),
                       n=('overhead_ratio','count')))
    stat['overhead_ratio_ci95'] = 1.96 * stat['overhead_ratio_std'] / np.sqrt(np.maximum(stat['n'], 1))
    plot_bar_with_table_highlight(
        df=stat, xcol='algo',
        ycol_mean='overhead_ratio_mean', ycol_ci95='overhead_ratio_ci95', ncol='n',
        title='Control Overhead Ratio (lower is better)',
        ylabel='control bits / (payload bits over rounds)',
        out_path=os.path.join(out_dir, 'fig_overhead_ratio_bar_table.png'),
        mode='closest_to_zero'   # 越接近0越好 = 越小越好
    )

# >>> NEW: ② 单位成功传输能耗合成图（越低越好）
def plot_energy_per_delivered_bar(df_all: pd.DataFrame, out_dir: str):
    by_seed = _mean_by_seed(df_all, ['energy_per_delivered'])
    stat = (by_seed.groupby('algo', as_index=False)
                  .agg(ed_mean=('energy_per_delivered','mean'),
                       ed_std =('energy_per_delivered','std'),
                       n=('energy_per_delivered','count')))
    stat['ed_ci95'] = 1.96 * stat['ed_std'] / np.sqrt(np.maximum(stat['n'], 1))
    plot_bar_with_table_highlight(
        df=stat, xcol='algo',
        ycol_mean='ed_mean', ycol_ci95='ed_ci95', ncol='n',
        title='Energy per Delivered (lower is better)',
        ylabel='energy per delivered data unit',
        out_path=os.path.join(out_dir, 'fig_energy_per_delivered_bar_table.png'),
        mode='closest_to_zero'
    )

# >>> NEW: ③ 胜率图（以 PDR 为例：每个 ratio 看谁最高）
def plot_winrate_pdr(df_all: pd.DataFrame, out_dir: str):
    # 对每个 (ratio, algo) 先在 seed 维度取均值，再在 ratio 维度找赢家
    p = (df_all.groupby(['ratio','algo'])['pdr']
               .mean()
               .reset_index())
    winners = p.loc[p.groupby('ratio')['pdr'].idxmax()]
    win = (winners['algo'].value_counts()
                     .rename_axis('algo')
                     .reset_index(name='wins'))
    # 画水平条形图
    plt.figure(figsize=(8, 4.2))
    ys = np.arange(len(win))
    plt.barh(ys, win['wins'].to_numpy())
    plt.yticks(ys, win['algo'].tolist())
    plt.xlabel('Number of ratios won (by highest PDR)')
    plt.title('Win-rate across malicious ratios (metric: PDR)')
    for y, v in zip(ys, win['wins'].tolist()):
        plt.text(v + 0.1, y, str(v), va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_winrate_pdr.png'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'fig_winrate_pdr.svg'), dpi=150)
    plt.close()

# >>> NEW: ④ 可靠性–开销 Pareto（横轴 AUC_R，纵轴 Overhead，左下角更优）
def plot_pareto_auc_overhead(df_all: pd.DataFrame, auc_stat: pd.DataFrame, out_dir: str):
    # 平均 overhead（按 (algo, seed) 先均值，然后到 algo）
    over = _mean_by_seed(df_all, ['overhead_ratio'])
    over_m = (over.groupby('algo', as_index=False)
                   .agg(overhead_ratio_mean=('overhead_ratio','mean')))
    # 合并 AUC_R（已有：algo, auc_norm_mean）
    m = pd.merge(auc_stat[['algo','auc_norm_mean']], over_m, on='algo', how='inner')

    # 计算 Pareto 前沿：AUC 越大越好，Overhead 越小越好
    pts = m[['auc_norm_mean','overhead_ratio_mean']].to_numpy()
    is_dom = np.zeros(len(m), dtype=bool)
    for i,(x,y) in enumerate(pts):
        dominated = False
        for j,(x2,y2) in enumerate(pts):
            if j==i: continue
            if (x2>=x and y2<=y) and ((x2>x) or (y2<y)):
                dominated = True; break
        is_dom[i] = dominated
    front = m.loc[~is_dom].sort_values(['auc_norm_mean','overhead_ratio_mean'], ascending=[True, True])

    # 画散点 + 前沿折线
    plt.figure(figsize=(6.6, 5.2))
    for _, r in m.iterrows():
        plt.scatter(r['auc_norm_mean'], r['overhead_ratio_mean'])
        plt.text(r['auc_norm_mean']+1e-4, r['overhead_ratio_mean']+1e-4, r['algo'], fontsize=9)
    if len(front) >= 2:
        plt.plot(front['auc_norm_mean'], front['overhead_ratio_mean'])
    plt.xlabel('AUC_R of PDR–ratio (higher is better)')
    plt.ylabel('Control Overhead Ratio (lower is better)')
    plt.title('Reliability vs. Control Overhead (Pareto)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_pareto_auc_overhead.png'), dpi=150)
    plt.savefig(os.path.join(out_dir, 'fig_pareto_auc_overhead.svg'), dpi=150)
    plt.close()


def run_all_tags(tags, n_nodes:int, n_malicious:int, seeds:list, rounds:int, until_dead:bool, out_dir:str, suffix:str=""):
    os.makedirs(out_dir, exist_ok=True)
    rows=[]
    for seed in seeds:
        for tag in tags:
            print(f'Running {tag} (seed={seed}) ...')
            res = run_one(tag, n_nodes, n_malicious, seed, rounds, until_dead, out_dir, suffix=suffix)
            rows.append(res)
    df = pd.DataFrame(rows)
    df.insert(0, 'algo_tag', [r['algo'] for r in rows])
    if suffix:
        df.to_csv(os.path.join(out_dir, f'summary_{suffix}.csv'), index=False)
    else:
        df.to_csv(os.path.join(out_dir, f'summary.csv'), index=False)
    return df

# >>> CHANGED: 更稳的聚合方式（均值±标准差），顺序固定，误差线与数值标注
# 覆盖原 plot_bars：按算法聚合画“柱状对比”，自动单位/百分比/CI
def plot_bars(summary: pd.DataFrame, out_dir: str, algo_order=None):
    summary = add_derived_columns(summary)
    if algo_order is None:
        algo_order = [ALGOS[k][1] for k in ALGOS]  # 动态取 'SHEER','ACTAR','TFSM_DPC','DST_WOA','TRAIL'


    # 仅保留出现过的算法顺序
    algos_present = [a for a in algo_order if a in summary["algo"].unique()]
    if not algos_present:
        print("[warn] no known algos present in summary.")
        return

    grouped = summary.groupby('algo')
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(numeric_only=True).fillna(0.0)
    cnts  = grouped.size()

    # 想画的指标（从 METRIC_SPECS 里挑存在于数据里的键）
    keys = [k for k in METRIC_SPECS.keys() if k in means.columns]

    for key in keys:
        spec = METRIC_SPECS[key]
        y = means.loc[algos_present, key].values
        sd = stds.loc[algos_present, key].values if cnts.max() > 1 else None
        n  = cnts.loc[algos_present].values if cnts.max() > 1 else None

        # 百分比转 %
        y_plot = y * 100.0 if spec["percent"] else y
        yerr = None
        if sd is not None and n is not None:
            sem = sd / np.sqrt(np.maximum(n, 1))
            ci95 = 1.96 * sem
            yerr = ci95 * (100.0 if spec["percent"] else 1.0)

        plt.figure(figsize=(8, 4.2))
        xs = np.arange(len(algos_present))
        bars = plt.bar(xs, y_plot, yerr=yerr, capsize=4)
        plt.xticks(xs, algos_present, rotation=20, ha='right')
        ylabel = f'{spec["label"]} ({spec["unit"]})' if spec["unit"] else spec["label"]
        plt.ylabel(ylabel)
        plt.title(spec["label"])
        if spec["percent"]:
            plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # 数值标注
        for i, b in enumerate(bars):
            val = y_plot[i]
            s = f'{val:.1f}' if spec["percent"] else (f'{val:.3g}' if abs(val) < 1000 else f'{val:,.0f}')
            plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, s, ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'metric_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'metric_{key}.svg'), dpi=150)
        plt.close()


# 覆盖原 plot_lines_with_error：随比例（%）的“折线+95%CI 阴影”，自动单位/百分比
def plot_lines_with_error(df_all: pd.DataFrame, out_dir: str, seeds: int):
    df_all = add_derived_columns(df_all)
    algo_order = [ALGOS[k][1] for k in ALGOS]  # 动态取 'SHEER','ACTAR','TFSM_DPC','DST_WOA','TRAIL'


    # 只画在 df 中实际存在的指标
    keys = [k for k in METRIC_SPECS.keys() if k in df_all.columns]

    def agg_mean_ci(df: pd.DataFrame, key: str):
        g = df.groupby(['algo', 'ratio'])[key]
        stat = g.agg(['mean', 'std', 'count']).reset_index()
        stat.rename(columns={'count':'n'}, inplace=True)
        stat['sem'] = stat['std'] / np.sqrt(np.maximum(stat['n'], 1))
        stat['ci95'] = 1.96 * stat['sem']
        return stat.sort_values(['algo', 'ratio'])

    for key in keys:
        spec = METRIC_SPECS[key]
        plt.figure(figsize=(8.8, 4.8))
        any_line = False

        for algo_name in algo_order:
            sub = df_all[df_all['algo'] == algo_name]
            if sub.empty:
                continue
            stat = agg_mean_ci(sub, key)
            if stat.empty:
                continue

            xs = (stat['ratio'] * 100).values  # 横轴：恶意比例(%)
            ys = stat['mean'].values
            ci = stat['ci95'].values if seeds > 1 else np.zeros_like(ys)

            # 百分比转 %
            ys_plot = ys * 100.0 if spec["percent"] else ys
            ci_plot = ci * (100.0 if spec["percent"] else 1.0)

            plt.plot(xs, ys_plot, marker='o', linewidth=1.8, label=algo_name)
            if seeds > 1:
                plt.fill_between(xs, ys_plot - ci_plot, ys_plot + ci_plot, alpha=0.18)

            any_line = True

        if not any_line:
            plt.close()
            continue

        plt.xlabel('Malicious ratio (%)')
        ylabel = f'{spec["label"]} ({spec["unit"]})' if spec["unit"] else spec["label"]
        plt.ylabel(ylabel)
        plt.title(spec["label"])
        # 仅对“概率/率”类正向百分比使用 0–100 的固定轴
        if key in ("pdr", "timely_transfer_rate", "timely_overall", "drop_rate"):
            plt.ylim(0, 100)
            if key in ("pdr", "timely_transfer_rate"):
                plt.axhline(90, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # 对“相对提升/降低(%)”不设 0–100 上限，并加 0 参考线
        if key.endswith("_rel_impr_pct") or key.endswith("_rel_red_pct"):
            plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)

        plt.xticks([10,20,30,40,50])
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'line_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'line_{key}.svg'), dpi=150)
        plt.close()


# >>> CHANGED: 容错、网格、线宽、SVG 导出
def plot_timely_curves(tags, out_dir: str, suffix: str = ""):
    plt.figure(figsize=(8, 4.2))
    for tag in tags:
        fname = f'hist_{suffix}_{tag}.csv' if suffix else f'hist_{tag}.csv'
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            print(f"[warn] history file not found: {fpath}")
            continue
        h = pd.read_csv(fpath)
        label_name = ALGOS.get(tag, (None, tag))[1]
        plt.plot(h['round'], h['cum_timely_rate'], label=label_name, linewidth=1.8)
    plt.xlabel('round')
    plt.ylabel('cumulative timely transfer rate')
    plt.legend()
    plt.title('Timely Transfer Rate (Cumulative)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    name = f'timely_rate_curves_{suffix}.png' if suffix else 'timely_rate_curves.png'
    plt.savefig(os.path.join(out_dir, name), dpi=150)
    # 另存 SVG
    name_svg = f'timely_rate_curves_{suffix}.svg' if suffix else 'timely_rate_curves.svg'
    plt.savefig(os.path.join(out_dir, name_svg), dpi=150)
    plt.close()


def parse_ratios(s: str):
    items = [x.strip() for x in s.split(',') if x.strip()]
    vals = []
    for it in items:
        try:
            v = float(it)
            if 0.0 < v < 1.0:
                vals.append(v)
        except:
            pass
    return sorted(set(vals))

# >>> CHANGED: 每个算法用自己的 xs（防止缺某个 ratio 时长度不匹配），误差带对齐
def agg_mean_std(df_all: pd.DataFrame, key: str):
    g = df_all.groupby(['algo', 'ratio'])[key]
    mu = g.mean().reset_index(name='mean')
    sd = g.std().fillna(0.0).reset_index(name='std')
    cnt = g.count().reset_index(name='n')
    out = mu.merge(sd, on=['algo', 'ratio']).merge(cnt, on=['algo', 'ratio'])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices=list(ALGOS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ratios', type=str, help='例如 0.1,0.2,0.3,0.4,0.5')
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--until-dead', action='store_true', help='忽略 --rounds，直至能量耗尽 (LND)')
    parser.add_argument('--seeds', type=int, default=1, help='重复实验次数（不同随机种子）')
    args = parser.parse_args()



    out_dir= 'outputs'; os.makedirs(out_dir, exist_ok=True)
    tags = [args.algo] if args.algo else list(ALGOS.keys())
    if args.all or (not args.algo): tags=list(ALGOS.keys())

    if args.ratios:
        ratios = parse_ratios(args.ratios)
        if not ratios:
            print("ratios 解析失败，请使用例如 --ratios 0.1,0.2,0.3,0.4,0.5")
            sys.exit(1)
        all_rows=[]
        seeds_list=list(range(42, 42+args.seeds))
        for r in ratios:
            n_nodes=100; n_mal=int(n_nodes*r)
            df = run_all_tags(tags, n_nodes, n_mal, seeds_list, rounds=args.rounds,
                              until_dead=args.until_dead, out_dir=out_dir, suffix=f'ratio_{int(r*100)}')
            # 追加 drop_rate 派生列
            df['ratio'] = r
            # >>> CHANGED: 统一端到端口径
            df['drop_rate'] = 1.0 - df['pdr']
            all_rows.append(df)
            plot_timely_curves(tags, out_dir, suffix=f'ratio_{int(r*100)}')
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, 'summary_all.csv'), index=False)
        # === 派生列（BPJ/总体及时率/开销占比等） ===
        df_all = add_derived_columns(df_all)
        # === (⑤) 相对基线提升（基线取 SHEER）===
        BASELINE = ALGOS['sheer'][1]  # 'SHEER'
        base = (df_all[df_all['algo'] == BASELINE]
                [['ratio', 'seed', 'pdr', 'bits_per_joule', 'overhead_ratio']]
                .rename(columns={'pdr': 'pdr_base', 'bits_per_joule': 'bpj_base', 'overhead_ratio': 'overhead_base'}))

        df_all = df_all.merge(base, on=['ratio', 'seed'], how='left')
        # 提升(%) / 降低(%)
        df_all['pdr_rel_impr_pct'] = (df_all['pdr'] / df_all['pdr_base'] - 1.0) * 100.0
        df_all['bpj_rel_impr_pct'] = (df_all['bits_per_joule'] / df_all['bpj_base'] - 1.0) * 100.0
        df_all['overhead_rel_red_pct'] = (1.0 - df_all['overhead_ratio'] / df_all['overhead_base']) * 100.0

        # === 论文主表（algo×ratio 聚合：均值/标准差/95%CI/CV）===
        def _ci95(x):
            x = x.dropna()
            return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

        def _cv(x):
            x = x.dropna()
            return x.std(ddof=1) / x.mean() if len(x) > 1 and x.mean() != 0 else np.nan

        keys_for_table = [
            'pdr', 'timely_overall', 'bits_per_joule', 'overhead_ratio',
            'pdr_rel_impr_pct', 'bpj_rel_impr_pct', 'overhead_rel_red_pct',
            'FND', 'HND', 'LND'
        ]
        g = df_all.groupby(['algo', 'ratio'])
        paper_agg = []
        for k in keys_for_table:
            s = g[k].agg(['mean', 'std', 'count']).rename(
                columns={'mean': f'{k}_mean', 'std': f'{k}_std', 'count': f'{k}_n'})
            s[f'{k}_ci95'] = g[k].apply(_ci95)
            s[f'{k}_cv'] = g[k].apply(_cv)
            paper_agg.append(s)
        paper_table = pd.concat(paper_agg, axis=1).reset_index()
        paper_table.to_csv(os.path.join(out_dir, 'paper_table_by_ratio.csv'), index=False)
        print("[csv] outputs/paper_table_by_ratio.csv")

        # === ① 稳健性斜率（PDR~ratio）——先 (algo,seed) 级拟合，再按 algo 聚合 ===
        def _slope(x, y):
            x = np.asarray(x);
            y = np.asarray(y)
            if len(np.unique(x)) < 2:
                return np.nan
            return np.polyfit(x, y, 1)[0]

        # 修复 pandas FutureWarning：groupby 后先选子列再 apply
        by_alg_seed_slope = (df_all
                             .groupby(['algo', 'seed'])[['ratio', 'pdr']]
                             .apply(lambda g: pd.Series({'pdr_slope': _slope(g['ratio'].to_numpy(),
                                                                             g['pdr'].to_numpy())}))
                             .reset_index())

        robust = (by_alg_seed_slope.groupby('algo', as_index=False)
                  .agg(pdr_slope_mean=('pdr_slope', 'mean'),
                       pdr_slope_std=('pdr_slope', 'std'),
                       n=('pdr_slope', 'count')))
        robust['pdr_slope_ci95'] = 1.96 * robust['pdr_slope_std'] / np.sqrt(np.maximum(robust['n'], 1))

        robust_out = os.path.join(out_dir, 'robustness_slope.csv')
        robust.to_csv(robust_out, index=False)
        print(f"[csv] {robust_out}")

        # === ② AUC_R（PDR–ratio 曲线面积）——(algo,seed) 级，后按 algo 聚合 ===
        def _auc_per_seed(g: pd.DataFrame) -> pd.Series:
            sub = g[['ratio', 'pdr']].dropna().sort_values('ratio')
            xs, ys = sub['ratio'].to_numpy(), sub['pdr'].to_numpy()
            if len(xs) < 2 or np.isclose(xs.max(), xs.min()):
                return pd.Series({'auc_raw': np.nan, 'auc_norm': np.nan})
            # 修复 numpy DeprecationWarning: trapz -> trapezoid
            try:
                auc_raw = np.trapezoid(ys, xs)
            except AttributeError:
                auc_raw = np.trapz(ys, xs)
            auc_norm = auc_raw / (xs.max() - xs.min())  # 归一化成“区间平均PDR”
            return pd.Series({'auc_raw': auc_raw, 'auc_norm': auc_norm})

        by_alg_seed_auc = (df_all
                           .groupby(['algo', 'seed'])[['ratio', 'pdr']]
                           .apply(_auc_per_seed)
                           .reset_index())

        auc_stat = (by_alg_seed_auc.groupby('algo', as_index=False)
                    .agg(auc_norm_mean=('auc_norm', 'mean'),
                         auc_norm_std=('auc_norm', 'std'),
                         n=('auc_norm', 'count')))
        auc_stat['auc_norm_ci95'] = 1.96 * auc_stat['auc_norm_std'] / np.sqrt(np.maximum(auc_stat['n'], 1))

        auc_detail_path = os.path.join(out_dir, 'pdr_auc_ratio_detail.csv')
        auc_out = os.path.join(out_dir, 'pdr_auc_ratio.csv')
        by_alg_seed_auc.to_csv(auc_detail_path, index=False)
        auc_stat.to_csv(auc_out, index=False)
        print(f"[csv] {auc_out}")
        print(f"[csv] {auc_detail_path}")
        # === (⑦) 稳定性导出（不分 ratio 的总览） ===
        stab = (df_all.groupby('algo')
                .agg(pdr_mean=('pdr', 'mean'), pdr_std=('pdr', 'std'),
                     bpj_mean=('bits_per_joule', 'mean'), bpj_std=('bits_per_joule', 'std'),
                     overhead_mean=('overhead_ratio', 'mean'), overhead_std=('overhead_ratio', 'std'),
                     timely_overall_mean=('timely_overall', 'mean'), timely_overall_std=('timely_overall', 'std'),
                     n=('pdr', 'count'))
                .reset_index())
        stab['pdr_ci95'] = 1.96 * stab['pdr_std'] / np.sqrt(np.maximum(stab['n'], 1))
        stab['bpj_ci95'] = 1.96 * stab['bpj_std'] / np.sqrt(np.maximum(stab['n'], 1))
        stab['overhead_ci95'] = 1.96 * stab['overhead_std'] / np.sqrt(np.maximum(stab['n'], 1))
        stab['timely_overall_ci95'] = 1.96 * stab['timely_overall_std'] / np.sqrt(np.maximum(stab['n'], 1))
        stab.to_csv(os.path.join(out_dir, 'stability_by_algo.csv'), index=False)
        print("[csv] outputs/stability_by_algo.csv")

        # === (⑦) 与基线的配对 t 检验（同 ratio、同 seed 对齐） ===
        try:
            from scipy import stats as _stats
        except Exception:
            _stats = None

        def _paired_p(metric):
            if _stats is None:
                return np.nan
            a = df_all[df_all['algo'] != BASELINE][['algo', 'ratio', 'seed', metric]]
            b = df_all[df_all['algo'] == BASELINE][['ratio', 'seed', metric]].rename(columns={metric: f'{metric}_base'})
            m = a.merge(b, on=['ratio', 'seed'], how='inner')
            res = []
            for algo, sub in m.groupby('algo'):
                if len(sub) < 2:
                    res.append((algo, np.nan))
                    continue
                t, p = _stats.ttest_rel(sub[metric], sub[f'{metric}_base'])
                res.append((algo, p))
            return pd.DataFrame(res, columns=['algo', f'p_{metric}_vs_{BASELINE}'])

        p_pdr = _paired_p('pdr')
        p_bpj = _paired_p('bits_per_joule')
        p_ovh = _paired_p('overhead_ratio')
        pval_df = (((p_pdr.merge(p_bpj, on='algo', how='outer')).merge(p_ovh, on='algo', how='outer'))
                   .sort_values('algo'))
        pval_df.to_csv(os.path.join(out_dir, 'paired_t_pvalues_vs_baseline.csv'), index=False)
        print("[csv] outputs/paired_t_pvalues_vs_baseline.csv")

        # === ③ 合成图：斜率（离0越近越好） & AUC_R（越大越好），并高亮最佳算法 ===
        plot_bar_with_table_highlight(
            df=robust,
            xcol='algo',
            ycol_mean='pdr_slope_mean',
            ycol_ci95='pdr_slope_ci95',
            ncol='n',
            title='Robustness (PDR–ratio slope, closer to 0 is better)',
            ylabel='Slope of PDR vs. ratio',
            out_path=os.path.join(out_dir, 'fig_pdr_slope_bar_table.png'),
            mode='closest_to_zero'
        )

        plot_bar_with_table_highlight(
            df=auc_stat,
            xcol='algo',
            ycol_mean='auc_norm_mean',
            ycol_ci95='auc_norm_ci95',
            ncol='n',
            title='Robustness (AUC_R over ratio range)',
            ylabel='Normalized AUC of PDR–ratio (≈ mean PDR)',
            out_path=os.path.join(out_dir, 'fig_auc_ratio_bar_table.png'),
            mode='max'
        )

        # === 仍保留你原有的折线图输出（PDR/寿命等） ===
        # >>> NEW: 追加 4 张“论文友好型”合成图
        plot_overhead_bar(df_all, out_dir)
        plot_energy_per_delivered_bar(df_all, out_dir)
        plot_winrate_pdr(df_all, out_dir)
        plot_pareto_auc_overhead(df_all, auc_stat, out_dir)

        plot_lines_with_error(df_all, out_dir, seeds=args.seeds)
        print(
            "批量完成：summary_all.csv、robustness_slope.csv、pdr_auc_ratio*.csv、以及合成图 fig_*_bar_table.png 与 line_*.png。")

    else:
        seeds_list=list(range(42, 42+args.seeds))
        df = run_all_tags(tags, 100, 30, seeds_list, rounds=args.rounds,
                          until_dead=args.until_dead, out_dir=out_dir, suffix="")
        # >>> CHANGED: 统一端到端口径
        df['drop_rate'] = 1.0 - df['pdr']
        plot_bars(df, out_dir)
        plot_timely_curves(tags, out_dir)
        print("单次运行完成：summary.csv / metric_*.png / timely_rate_curves.png")

if __name__ == '__main__':
    main()
