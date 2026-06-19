# -*- coding: utf-8 -*-
"""
============================================================
 黄土高原退耕还林 EVI–VOD 时空差异性分析
 第二部分：统计分析与出图（Python 端）
============================================================

 输入数据（由 01_preprocessing.js 从 GEE 导出）：
   - evi_vod_monthly_ts.csv         区域平均月度时间序列（按植被类型）
   - evi_vod_di_igbp_mean.tif       多年均值空间图（EVI, VOD, DI, IGBP）
   - evi_vod_annual_YYYY.tif        逐年均值（用于 Sen slope）

 产出图表（对应 proposal 编号）：
   Fig.2  季节动态曲线（双 Y 轴，3 panel）
   Fig.3  互相关 Lag Analysis 柱状图 ⭐
   Fig.4  Sen Slope + Mann-Kendall 趋势图（需像元级 GeoTIFF）
   Fig.5  z-score 异常时序图
   Fig.6  VOD-EVI 饱和效应散点密度图
   Fig.7  Almon DLM 分布滞后权重曲线
   Fig.8  EVI-VOD 迟滞回环（相空间轨迹）

 依赖：
   pip install numpy pandas matplotlib seaborn scipy
   pip install pymannkendall rasterio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.interpolate import make_interp_spline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================

# ---- 文件路径（占位符，替换为实际路径） ----
CSV_PATH       = 'PLACEHOLDER_PATH/evi_vod_monthly_ts.csv'
SPATIAL_TIF    = 'PLACEHOLDER_PATH/evi_vod_di_igbp_mean.tif'
ANNUAL_TIF_DIR = 'PLACEHOLDER_PATH/'  # 内含 evi_vod_annual_2010.tif ~ 2021.tif
OUTPUT_DIR     = 'PLACEHOLDER_PATH/figures/'

# ---- 出图全局样式 ----
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'legend.fontsize':    10,
    'figure.dpi':         200,
    'savefig.dpi':        300,
    'savefig.bbox_inches': 'tight',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# 植被类型配色方案
VEG_COLORS = {
    'Grassland':  '#66c2a5',
    'Shrubland':  '#fc8d62',
    'Forest':     '#8da0cb',
    'Cropland':   '#e78ac3',
    'All_Vegetation': '#a6d854',
}

VEG_LABELS_ZH = {
    'Grassland':  '草地',
    'Shrubland':  '灌木',
    'Forest':     '森林',
    'Cropland':   '农田',
    'All_Vegetation': '全部植被',
}

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

# IGBP 土地覆盖映射
IGBP_NAMES = {
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Water Bodies"
}

IGBP_COLORS = {
    1: '#05450a', 2: '#086a10', 3: '#54a708', 4: '#78d203', 5: '#009900',
    6: '#c6b044', 7: '#dcd159', 8: '#dade48', 9: '#fbff13', 10: '#b6ff05',
    11: '#27ff87', 12: '#c24f44', 13: '#a5a5a5', 14: '#ff6d4c', 15: '#69fff8',
    16: '#f9ffa4', 17: '#1c0dff'
}

# ============================================================
# 地图制图要素与空间格局绘制
# ============================================================

def format_geo_ticks(ax, bounds):
    """格式化地理坐标轴刻度为 E/N 经纬度表示"""
    left, bottom, right, top = bounds
    x_range = right - left
    y_range = top - bottom
    x_step = 2.0 if x_range > 5 else 1.0
    y_step = 2.0 if y_range > 5 else 1.0
    
    ax.set_xticks(np.arange(np.ceil(left), right, x_step))
    ax.set_yticks(np.arange(np.ceil(bottom), top, y_step))
    
    import matplotlib.ticker as mticker
    lon_fmt = lambda x, pos: f"{abs(x):.1f}°E" if x >= 0 else f"{abs(x):.1f}°W"
    lat_fmt = lambda y, pos: f"{abs(y):.1f}°N" if y >= 0 else f"{abs(y):.1f}°S"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_fmt))
    ax.tick_params(axis='both', labelsize=8)

def add_scale_bar(ax, bounds, scale_length_km=100, scale_loc=(0.08, 0.08), color='black'):
    """在图层中绘制简洁线段比例尺"""
    left, bottom, right, top = bounds
    lat_center = (bottom + top) / 2.0
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_center))
    length_deg_lon = scale_length_km / km_per_deg_lon
    
    lon_start = left + scale_loc[0] * (right - left)
    lat_start = bottom + scale_loc[1] * (top - bottom)
    lon_end = lon_start + length_deg_lon
    
    ax.plot([lon_start, lon_end], [lat_start, lat_start], color=color, lw=1.5, zorder=5)
    tick_height = 0.02 * (top - bottom)
    ax.plot([lon_start, lon_start], [lat_start, lat_start + tick_height], color=color, lw=1.5, zorder=5)
    ax.plot([lon_end, lon_end], [lat_start, lat_start + tick_height], color=color, lw=1.5, zorder=5)
    ax.text((lon_start + lon_end) / 2.0, lat_start + tick_height * 0.7, 
            f"{scale_length_km} km", ha='center', va='bottom', 
            fontsize=8, fontweight='bold', color=color, zorder=6)

def add_north_arrow(ax, bounds, arrow_loc=(0.9, 0.85), color='black'):
    """在图层中绘制指北针"""
    left, bottom, right, top = bounds
    lon_arrow = left + arrow_loc[0] * (right - left)
    lat_arrow = bottom + arrow_loc[1] * (top - bottom)
    arrow_len = 0.06 * (top - bottom)
    
    arrow = FancyArrowPatch(
        (lon_arrow, lat_arrow), (lon_arrow, lat_arrow + arrow_len),
        arrowstyle='->', mutation_scale=12, color=color, lw=1.5, zorder=5
    )
    ax.add_patch(arrow)
    ax.text(lon_arrow, lat_arrow + arrow_len + 0.01 * (top - bottom), 'N',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=color, zorder=6)

def plot_spatial_patterns(spatial_tif_path, output_path=None):
    """绘制多年平均空间格局对比图 (Fig.1a-1d)"""
    try:
        import rasterio
    except ImportError:
        print("  [WARNING] rasterio 未安装，跳过 Fig.1 空间图绘制")
        return
        
    if not os.path.exists(spatial_tif_path):
        print(f"  [WARNING] 未找到多年平均空间 TIFF 文件: {spatial_tif_path}，跳过 Fig.1 空间图绘制")
        return
        
    with rasterio.open(spatial_tif_path) as src:
        evi = src.read(1)
        vod = src.read(2)
        di = src.read(3)
        igbp = src.read(4)
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        nodata = src.nodata
        
    # 处理 nodata
    for data in [evi, vod, di, igbp]:
        if nodata is not None:
            data[data == nodata] = np.nan
            
    left, bottom, right, top = bounds
    extent = [left, right, bottom, top]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1a. EVI
    im_evi = axes[0].imshow(evi, cmap='YlGn', extent=extent, zorder=1)
    axes[0].set_title("Fig.1a  多年平均 EVI 绿度分布", fontweight='bold')
    plt.colorbar(im_evi, ax=axes[0], label='EVI', shrink=0.7)
    
    # 1b. VOD
    im_vod = axes[1].imshow(vod, cmap='YlOrRd', extent=extent, zorder=1)
    axes[1].set_title("Fig.1b  多年平均 VODCA L-band VOD 分布", fontweight='bold')
    plt.colorbar(im_vod, ax=axes[1], label='VOD (neper)', shrink=0.7)
    
    # 1c. DI
    im_di = axes[2].imshow(di, cmap='RdYlGn', vmin=-0.5, vmax=0.5, extent=extent, zorder=1)
    axes[2].set_title("Fig.1c  植被绿化-结构恢复解耦指数 (DI)", fontweight='bold')
    plt.colorbar(im_di, ax=axes[2], label='DI (VOD_norm - EVI_norm)', shrink=0.7)
    
    # 1d. IGBP
    present_classes = np.unique(igbp[~np.isnan(igbp) & (igbp > 0)]).astype(int)
    cmap_colors = [IGBP_COLORS.get(cl, '#ffffff') for cl in present_classes]
    discrete_cmap = mcolors.ListedColormap(cmap_colors)
    
    class_mapping = {cl: i for i, cl in enumerate(present_classes)}
    mapped_igbp = np.full(igbp.shape, np.nan)
    for cl, i in class_mapping.items():
        mapped_igbp[igbp == cl] = i
        
    axes[3].imshow(mapped_igbp, cmap=discrete_cmap, extent=extent, zorder=1)
    axes[3].set_title("Fig.1d  MCD12C1 IGBP 土地覆盖类型", fontweight='bold')
    
    legend_patches = []
    for cl in present_classes:
        name_zh = IGBP_NAMES.get(cl, f"Class {cl}")
        zh_map = {
            1: "针叶常绿林 (ENF)", 5: "混交林 (MF)", 7: "开阔灌木林 (OSH)", 
            8: "稀树木质草原 (WSA)", 9: "稀树草原 (SAV)", 10: "草地 (Grasslands)", 
            12: "农田 (Croplands)", 14: "农田/自然植被镶嵌", 16: "裸地 (Barren)"
        }
        name_label = zh_map.get(cl, name_zh)
        patch = mpatches.Patch(color=IGBP_COLORS.get(cl, '#ffffff'), label=name_label)
        legend_patches.append(patch)
        
    axes[3].legend(handles=legend_patches, loc='lower left', fontsize=7, framealpha=0.8,
                   bbox_to_anchor=(0.02, 0.02))
                   
    for ax in axes:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        format_geo_ticks(ax, bounds)
        add_scale_bar(ax, bounds, scale_length_km=100, scale_loc=(0.06, 0.06))
        add_north_arrow(ax, bounds, arrow_loc=(0.92, 0.82))
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

# ============================================================
# 1. 数据加载与预处理
# ============================================================

def load_timeseries(csv_path):
    """加载 GEE 导出的月度时间序列 CSV"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )
    df = df.sort_values(['veg_type', 'date']).reset_index(drop=True)

    # 线性插值填补个别缺测月份
    filled = []
    for vtype in df['veg_type'].unique():
        sub = df[df['veg_type'] == vtype].set_index('date')
        # 重建完整月度索引
        full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq='MS')
        sub = sub.reindex(full_idx)
        sub['EVI'] = sub['EVI'].interpolate(method='linear')
        sub['VOD'] = sub['VOD'].interpolate(method='linear')
        sub['veg_type'] = vtype
        sub['year']  = sub.index.year
        sub['month'] = sub.index.month
        sub = sub.reset_index().rename(columns={'index': 'date'})
        filled.append(sub)

    return pd.concat(filled, ignore_index=True)


# 加载数据
print("正在加载数据...")
df = load_timeseries(CSV_PATH)
print(f"  共 {len(df)} 条记录, 植被类型: {df['veg_type'].unique()}")

# 绘制 Fig.1 空间格局图
print("\n绘制 Fig.1: 空间格局与解耦指数 DI 空间分布图...")
plot_spatial_patterns(SPATIAL_TIF, output_path=OUTPUT_DIR + 'fig1_spatial_patterns.png')


# ============================================================
# 2. Fig.2 — 季节动态曲线（双 Y 轴，3 panel）
# ============================================================

def plot_seasonal_dynamics(df, veg_types=None, output_path=None):
    """
    绘制典型地表类型的 EVI-VOD 季节对比图
    3 panel，每个 panel 一种植被类型，双 Y 轴
    """
    if veg_types is None:
        veg_types = ['Grassland', 'Shrubland', 'Forest']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, vtype in zip(axes, veg_types):
        sub = df[df['veg_type'] == vtype]
        # 多年月均值 ± 标准差
        monthly = sub.groupby('month').agg(
            evi_mean=('EVI', 'mean'), evi_std=('EVI', 'std'),
            vod_mean=('VOD', 'mean'), vod_std=('VOD', 'std')
        ).reset_index()

        months = monthly['month'].values
        # EVI（左 Y 轴）
        color_evi = '#2ca02c'
        ax.fill_between(months,
                        monthly['evi_mean'] - monthly['evi_std'],
                        monthly['evi_mean'] + monthly['evi_std'],
                        alpha=0.15, color=color_evi)
        ln1 = ax.plot(months, monthly['evi_mean'], 'o-',
                      color=color_evi, linewidth=2, markersize=5, label='EVI')
        ax.set_ylabel('EVI', color=color_evi)
        ax.tick_params(axis='y', labelcolor=color_evi)

        # VOD（右 Y 轴）
        ax2 = ax.twinx()
        color_vod = '#d62728'
        ax2.fill_between(months,
                         monthly['vod_mean'] - monthly['vod_std'],
                         monthly['vod_mean'] + monthly['vod_std'],
                         alpha=0.15, color=color_vod)
        ln2 = ax2.plot(months, monthly['vod_mean'], 's--',
                       color=color_vod, linewidth=2, markersize=5, label='VOD')
        ax2.set_ylabel('VOD (neper)', color=color_vod)
        ax2.tick_params(axis='y', labelcolor=color_vod)
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(False)

        # 标注峰值月份
        evi_peak = months[np.argmax(monthly['evi_mean'])]
        vod_peak = months[np.argmax(monthly['vod_mean'])]
        lag = vod_peak - evi_peak

        ax.set_title(f'{VEG_LABELS_ZH[vtype]} ({vtype})\n'
                     f'EVI peak: {MONTH_LABELS[evi_peak-1]}, '
                     f'VOD peak: {MONTH_LABELS[vod_peak-1]}, '
                     f'Lag: {lag} mo', fontsize=11)

        ax.set_xlabel('Month')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)

        # 合并图例
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', framealpha=0.8, fontsize=9)

    fig.suptitle('Fig.2  EVI 与 VOD 季节动态对比（多年月均值 ± 1σ）',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

print("\n绘制 Fig.2: 季节动态曲线...")
plot_seasonal_dynamics(df, output_path=OUTPUT_DIR + 'fig2_seasonal_dynamics.png')


# ============================================================
# 3. Fig.3 — 互相关 Lag Analysis（⭐ 核心分析）
# ============================================================

def cross_correlation_analysis(df, max_lag=6, veg_types=None):
    """
    计算不同植被类型的 EVI-VOD 互相关分析
    正 lag 表示 VOD 滞后于 EVI（EVI 领先）
    返回: {veg_type: {'lags': [...], 'corrs': [...], 'pvals': [...], 'best_lag': int}}
    """
    if veg_types is None:
        veg_types = ['Grassland', 'Shrubland', 'Forest']

    results = {}
    for vtype in veg_types:
        sub = df[df['veg_type'] == vtype].sort_values('date')
        evi = sub['EVI'].values
        vod = sub['VOD'].values
        n = len(evi)

        lags = list(range(-max_lag, max_lag + 1))
        corrs = []
        pvals = []

        for lag in lags:
            if lag >= 0:
                # 正 lag: EVI[:-lag] vs VOD[lag:]  (EVI 领先)
                e = evi[:n - lag] if lag > 0 else evi
                v = vod[lag:] if lag > 0 else vod
            else:
                # 负 lag: EVI[-lag:] vs VOD[:lag]  (VOD 领先)
                e = evi[-lag:]
                v = vod[:n + lag]

            r, p = stats.pearsonr(e, v)
            corrs.append(r)
            pvals.append(p)

        best_idx = np.argmax(corrs)
        results[vtype] = {
            'lags':     lags,
            'corrs':    corrs,
            'pvals':    pvals,
            'best_lag': lags[best_idx],
            'best_r':   corrs[best_idx],
            'best_p':   pvals[best_idx],
        }

    return results


def plot_lag_analysis(results, output_path=None):
    """绘制 Fig.3: 互相关系数 vs 滞后月数 柱状图"""
    veg_types = list(results.keys())
    n_types = len(veg_types)

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4.5), sharey=True)
    if n_types == 1:
        axes = [axes]

    for ax, vtype in zip(axes, veg_types):
        res = results[vtype]
        lags  = res['lags']
        corrs = res['corrs']
        pvals = res['pvals']

        # 颜色：最优 lag 高亮
        colors = []
        for i, lag in enumerate(lags):
            if lag == res['best_lag']:
                colors.append('#e74c3c')     # 最优 lag 红色高亮
            elif pvals[i] < 0.01:
                colors.append('#3498db')     # 显著 (p<0.01) 蓝色
            elif pvals[i] < 0.05:
                colors.append('#85c1e9')     # 显著 (p<0.05) 浅蓝
            else:
                colors.append('#bdc3c7')     # 不显著 灰色

        bars = ax.bar(lags, corrs, color=colors, edgecolor='white', linewidth=0.5)

        # 标注最优 lag
        ax.annotate(
            f'Best lag = {res["best_lag"]} mo\nr = {res["best_r"]:.3f}\np < {res["best_p"]:.0e}',
            xy=(res['best_lag'], res['best_r']),
            xytext=(res['best_lag'] + 1.5, res['best_r'] - 0.05),
            fontsize=9, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor='#e74c3c')
        )

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Lag (months)\n← VOD leads | EVI leads →')
        ax.set_title(f'{VEG_LABELS_ZH[vtype]} ({vtype})', fontweight='bold')
        ax.set_xticks(lags)

    axes[0].set_ylabel('Pearson r')

    fig.suptitle('Fig.3  EVI–VOD 互相关分析（Lag Analysis）',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

    # 打印汇总表
    print("\n  ┌─────────────┬──────────┬───────────┬───────────┐")
    print("  │ 植被类型     │ 最优lag  │ Pearson r │ p-value   │")
    print("  ├─────────────┼──────────┼───────────┼───────────┤")
    for vtype, res in results.items():
        print(f"  │ {VEG_LABELS_ZH[vtype]:<10s}  │ {res['best_lag']:>5d} mo │ "
              f"{res['best_r']:>9.4f} │ {res['best_p']:>9.2e} │")
    print("  └─────────────┴──────────┴───────────┴───────────┘")


print("\n绘制 Fig.3: 互相关 Lag Analysis ⭐...")
lag_results = cross_correlation_analysis(df)
plot_lag_analysis(lag_results, output_path=OUTPUT_DIR + 'fig3_lag_analysis.png')


# ============================================================
# 4. Fig.4 — Sen Slope + Mann-Kendall 趋势检验（像元级）
# ============================================================
# 注意：此部分需要像元级逐年 GeoTIFF 数据
# 若仅有区域平均 CSV，可做区域平均趋势分析

def regional_trend_analysis(df, output_path=None):
    """
    区域平均尺度的 Sen slope + Mann-Kendall 趋势分析
    对每种植被类型计算年均值序列的趋势
    """
    try:
        import pymannkendall as mk
    except ImportError:
        print("  [WARNING] pymannkendall 未安装，跳过 MK 检验")
        print("  运行: pip install pymannkendall")
        return None

    veg_types = ['Grassland', 'Shrubland', 'Forest', 'All_Vegetation']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    trend_results = {}

    for ax, vtype in zip(axes, veg_types):
        sub = df[df['veg_type'] == vtype]
        annual = sub.groupby('year').agg(
            EVI_mean=('EVI', 'mean'),
            VOD_mean=('VOD', 'mean'),
        ).reset_index()

        years = annual['year'].values

        # EVI 趋势
        evi_mk = mk.original_test(annual['EVI_mean'].values)
        # VOD 趋势
        vod_mk = mk.original_test(annual['VOD_mean'].values)

        trend_results[vtype] = {
            'evi_slope': evi_mk.slope, 'evi_p': evi_mk.p,
            'evi_trend': evi_mk.trend,
            'vod_slope': vod_mk.slope, 'vod_p': vod_mk.p,
            'vod_trend': vod_mk.trend,
        }

        # 绘图
        color_evi = '#2ca02c'
        color_vod = '#d62728'

        ax.plot(years, annual['EVI_mean'], 'o-', color=color_evi,
                linewidth=2, markersize=5, label='EVI')
        # EVI 趋势线
        evi_trend_line = evi_mk.intercept + evi_mk.slope * (years - years[0])
        ax.plot(years, evi_trend_line, '--', color=color_evi, alpha=0.7)

        ax2 = ax.twinx()
        ax2.plot(years, annual['VOD_mean'], 's-', color=color_vod,
                 linewidth=2, markersize=5, label='VOD')
        # VOD 趋势线
        vod_trend_line = vod_mk.intercept + vod_mk.slope * (years - years[0])
        ax2.plot(years, vod_trend_line, '--', color=color_vod, alpha=0.7)
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(False)

        # 显著性标注
        evi_sig = '***' if evi_mk.p < 0.001 else ('**' if evi_mk.p < 0.01
                   else ('*' if evi_mk.p < 0.05 else 'n.s.'))
        vod_sig = '***' if vod_mk.p < 0.001 else ('**' if vod_mk.p < 0.01
                   else ('*' if vod_mk.p < 0.05 else 'n.s.'))

        ax.set_title(
            f'{VEG_LABELS_ZH[vtype]}\n'
            f'EVI: Sen={evi_mk.slope:.5f}/yr {evi_sig}  |  '
            f'VOD: Sen={vod_mk.slope:.5f}/yr {vod_sig}',
            fontsize=10
        )
        ax.set_ylabel('EVI', color=color_evi)
        ax2.set_ylabel('VOD', color=color_vod)

        if ax in axes[-2:]:
            ax.set_xlabel('Year')

    fig.suptitle('Fig.4  年际趋势：Sen Slope + Mann-Kendall 检验（区域平均）',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

    return trend_results


print("\n绘制 Fig.4: Sen Slope + Mann-Kendall 趋势...")
trend_results = regional_trend_analysis(
    df, output_path=OUTPUT_DIR + 'fig4_trend_analysis.png')


# ============================================================
# 5. Fig.5 — z-score 异常时序图（数据驱动）
# ============================================================

def zscore_anomaly_analysis(df, veg_type='All_Vegetation', output_path=None):
    """
    计算标准化月度异常值（z-score），识别极端气候异常时段
    z_t = (x_t - x_bar_month) / sigma_month
    """
    sub = df[df['veg_type'] == veg_type].copy().sort_values('date')

    # 计算每月的气候态（长期均值和标准差）
    climatology = sub.groupby('month').agg(
        evi_clim_mean=('EVI', 'mean'), evi_clim_std=('EVI', 'std'),
        vod_clim_mean=('VOD', 'mean'), vod_clim_std=('VOD', 'std'),
    )

    # 计算 z-score
    sub = sub.merge(climatology, on='month', how='left')
    sub['EVI_z'] = (sub['EVI'] - sub['evi_clim_mean']) / sub['evi_clim_std']
    sub['VOD_z'] = (sub['VOD'] - sub['vod_clim_mean']) / sub['vod_clim_std']

    # 识别异常时段：EVI_z 和 VOD_z 同时 < -1 的连续月段
    sub['both_negative'] = (sub['EVI_z'] < -1) & (sub['VOD_z'] < -1)
    sub['anomaly_score'] = np.where(sub['both_negative'],
                                     sub['EVI_z'] + sub['VOD_z'], 0)

    # 找到最强异常时段
    worst_idx = sub['anomaly_score'].idxmin()
    worst_date = sub.loc[worst_idx, 'date']

    # ---- 绘图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1]})

    dates = sub['date'].values

    # EVI z-score
    ax1.fill_between(dates, -1, 1, alpha=0.1, color='green',
                     label='±1σ 正常范围')
    ax1.bar(dates, sub['EVI_z'], width=25, color=np.where(sub['EVI_z'] < 0, '#e74c3c', '#27ae60'),
            alpha=0.7, edgecolor='none')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axhline(-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.axhline(1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.set_ylabel('EVI z-score')
    ax1.set_title('EVI 标准化异常', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # VOD z-score
    ax2.fill_between(dates, -1, 1, alpha=0.1, color='orange',
                     label='±1σ 正常范围')
    ax2.bar(dates, sub['VOD_z'], width=25, color=np.where(sub['VOD_z'] < 0, '#e74c3c', '#2980b9'),
            alpha=0.7, edgecolor='none')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axhline(-1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.axhline(1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.set_ylabel('VOD z-score')
    ax2.set_title('VOD 标准化异常', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)

    # 标注最强异常时段
    for ax in [ax1, ax2]:
        ax.axvline(worst_date, color='red', linewidth=1.5, linestyle='-', alpha=0.8)
        ax.annotate(f'最强异常\n{worst_date.strftime("%Y-%m")}',
                    xy=(worst_date, 0), fontsize=9, color='red',
                    fontweight='bold', ha='center',
                    xytext=(worst_date + pd.Timedelta(days=120), 2),
                    arrowprops=dict(arrowstyle='->', color='red'))

    # 高亮所有双指标同时 < -1σ 的月份
    anomaly_months = sub[sub['both_negative']]['date']
    for d in anomaly_months:
        for ax in [ax1, ax2]:
            ax.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15),
                       alpha=0.15, color='red', zorder=0)

    fig.suptitle('Fig.5  EVI–VOD 月度标准化异常值（z-score）时序图\n'
                 f'（{VEG_LABELS_ZH[veg_type]}区域平均，2010–2021）',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

    # 打印异常时段统计
    print(f"\n  双指标同时 < -1σ 的月份共 {len(anomaly_months)} 个：")
    for _, row in sub[sub['both_negative']].iterrows():
        print(f"    {row['date'].strftime('%Y-%m')}  "
              f"EVI_z={row['EVI_z']:+.2f}  VOD_z={row['VOD_z']:+.2f}")

    return sub


print("\n绘制 Fig.5: z-score 异常时序图...")
anomaly_df = zscore_anomaly_analysis(
    df, veg_type='All_Vegetation',
    output_path=OUTPUT_DIR + 'fig5_zscore_anomaly.png')


# ============================================================
# 6. Fig.6 — 饱和效应散点密度图
# ============================================================

def plot_saturation_scatter(df, output_path=None):
    """
    VOD vs EVI 散点密度图，按植被类型赋色
    观察 EVI 在高 VOD 区域是否饱和
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    veg_types_plot = ['Forest', 'Shrubland', 'Grassland', 'Cropland']

    for vtype in veg_types_plot:
        sub = df[df['veg_type'] == vtype]
        ax.scatter(sub['VOD'], sub['EVI'],
                   c=VEG_COLORS.get(vtype, '#888'),
                   alpha=0.25, s=8, edgecolors='none',
                   label=f'{VEG_LABELS_ZH[vtype]} (n={len(sub)})')

    ax.set_xlabel('VOD (neper)', fontsize=13)
    ax.set_ylabel('EVI', fontsize=13)
    ax.set_title('Fig.6  VOD–EVI 散点图：饱和效应探究', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', markerscale=4, framealpha=0.9, fontsize=10)

    # 添加注释箭头标注饱和区
    ax.annotate('EVI 饱和区\n（渐近线形态）',
                xy=(0.45, 0.48), xytext=(0.35, 0.55),
                fontsize=10, fontweight='bold', color='#7f8c8d',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

print("\n绘制 Fig.6: 饱和效应散点图...")
plot_saturation_scatter(df, output_path=OUTPUT_DIR + 'fig6_saturation.png')


# ============================================================
# 7. Fig.7 — Almon DLM 分布滞后权重（Discussion）
# ============================================================

def almon_dlm_analysis(df, K=6, poly_degree=2, veg_types=None, output_path=None):
    """
    Almon 多项式约束的分布滞后模型
    VOD_t = α + Σ β_k * EVI_{t-k} + ε
    约束 β_k = γ_0 + γ_1*k + γ_2*k^2 (2阶多项式)

    通过构造 Almon 变换矩阵 H，将原始滞后变量 X 转换为 Z = X @ H，
    使得 OLS 估计 γ 后可反推 β = H @ γ，从根本上消除共线性。
    """
    if veg_types is None:
        veg_types = ['Grassland', 'Shrubland', 'Forest']

    fig, axes = plt.subplots(1, len(veg_types), figsize=(5 * len(veg_types), 4.5),
                             sharey=True)
    if len(veg_types) == 1:
        axes = [axes]

    all_results = {}

    for ax, vtype in zip(axes, veg_types):
        sub = df[df['veg_type'] == vtype].sort_values('date')
        evi_full = sub['EVI'].values
        vod_full = sub['VOD'].values
        n = len(evi_full)

        # 构造滞后矩阵 X: (n-K) x (K+1)
        X_lag = np.column_stack([
            evi_full[K - k: n - k] for k in range(K + 1)
        ])
        y = vod_full[K:]

        # Almon 变换矩阵 H: (K+1) x (poly_degree+1)
        k_vals = np.arange(K + 1, dtype=float)
        H = np.column_stack([k_vals**d for d in range(poly_degree + 1)])

        # 变换后的设计矩阵 Z = X @ H
        Z = X_lag @ H

        # 加截距项
        Z_with_const = np.column_stack([np.ones(len(Z)), Z])

        # OLS 估计 γ
        try:
            gamma, residuals, rank, sv = np.linalg.lstsq(Z_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            print(f"  [WARNING] {vtype} 的 OLS 求解失败，跳过")
            continue

        # 反推 β = H @ γ[1:]  (去掉截距)
        gamma_poly = gamma[1:]
        beta = H @ gamma_poly

        # 拟合值 & R²
        y_hat = Z_with_const @ gamma
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        all_results[vtype] = {
            'beta': beta,
            'gamma': gamma,
            'r_squared': r_squared,
            'peak_lag': np.argmax(beta),
        }

        # 绘制权重曲线
        ax.bar(k_vals, beta, color=VEG_COLORS.get(vtype, '#888'),
               alpha=0.6, edgecolor='white', width=0.7)

        # 用平滑曲线连接
        if len(k_vals) > 3:
            k_smooth = np.linspace(0, K, 100)
            beta_smooth_fn = np.poly1d(np.polyfit(k_vals, beta, poly_degree))
            ax.plot(k_smooth, beta_smooth_fn(k_smooth), '-',
                    color=VEG_COLORS.get(vtype, '#888'), linewidth=2.5)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Lag k (months)')
        ax.set_title(f'{VEG_LABELS_ZH[vtype]}\n'
                     f'Peak lag = {np.argmax(beta)} mo, R² = {r_squared:.3f}',
                     fontsize=11)
        ax.set_xticks(range(K + 1))

    axes[0].set_ylabel(r'$\beta_k$ (Almon weight)')

    fig.suptitle('Fig.7  Almon 多项式分布滞后模型：EVI 对 VOD 的"生态记忆"权重',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

    # 打印结果
    print("\n  Almon DLM 结果汇总：")
    for vtype, res in all_results.items():
        print(f"    {VEG_LABELS_ZH[vtype]:6s}: peak lag = {res['peak_lag']} mo, "
              f"R² = {res['r_squared']:.4f}")

    return all_results


print("\n绘制 Fig.7: Almon DLM 分布滞后权重...")
dlm_results = almon_dlm_analysis(
    df, K=6, poly_degree=2,
    output_path=OUTPUT_DIR + 'fig7_almon_dlm.png')


# ============================================================
# 8. Fig.8 — EVI-VOD 迟滞回环（Discussion）
# ============================================================

def plot_hysteresis_loops(df, veg_types=None, output_path=None):
    """
    绘制 EVI-VOD 相空间轨迹（迟滞回环）
    X 轴 = EVI, Y 轴 = VOD, 按月份顺序连接
    用 Shoelace 公式计算回环面积
    """
    if veg_types is None:
        veg_types = ['Grassland', 'Shrubland', 'Forest']

    fig, axes = plt.subplots(1, len(veg_types), figsize=(5.5 * len(veg_types), 5),
                             sharey=True)
    if len(veg_types) == 1:
        axes = [axes]

    loop_areas = {}

    for ax, vtype in zip(axes, veg_types):
        sub = df[df['veg_type'] == vtype]
        monthly = sub.groupby('month').agg(
            evi=('EVI', 'mean'), vod=('VOD', 'mean')
        ).reset_index()

        evi = monthly['evi'].values
        vod = monthly['vod'].values
        months = monthly['month'].values

        # 闭合轨迹（首尾相连）
        evi_closed = np.append(evi, evi[0])
        vod_closed = np.append(vod, vod[0])

        # Shoelace 公式计算回环面积
        n = len(evi)
        area = 0.5 * abs(sum(
            evi[i] * vod[(i + 1) % n] - evi[(i + 1) % n] * vod[i]
            for i in range(n)
        ))
        loop_areas[vtype] = area

        # 绘制轨迹
        color = VEG_COLORS.get(vtype, '#888')
        ax.plot(evi_closed, vod_closed, '-', color=color,
                linewidth=2.5, alpha=0.8, zorder=2)
        ax.fill(evi_closed, vod_closed, color=color, alpha=0.08, zorder=1)

        # 标注月份编号
        for i, m in enumerate(months):
            ax.plot(evi[i], vod[i], 'o', color=color,
                    markersize=10, zorder=3, markeredgecolor='white',
                    markeredgewidth=1.5)
            ax.annotate(MONTH_LABELS[m - 1], (evi[i], vod[i]),
                        fontsize=7, fontweight='bold', ha='center', va='center',
                        color='white', zorder=4)

        # 标注箭头表示方向（从 Jan→Feb）
        mid_x = (evi[0] + evi[1]) / 2
        mid_y = (vod[0] + vod[1]) / 2
        dx = evi[1] - evi[0]
        dy = vod[1] - vod[0]
        ax.annotate('', xy=(mid_x + dx * 0.3, mid_y + dy * 0.3),
                    xytext=(mid_x - dx * 0.1, mid_y - dy * 0.1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

        ax.set_xlabel('EVI', fontsize=12)
        ax.set_title(f'{VEG_LABELS_ZH[vtype]}\n'
                     f'Loop area = {area:.6f}',
                     fontsize=11, fontweight='bold')

    axes[0].set_ylabel('VOD (neper)', fontsize=12)

    fig.suptitle('Fig.8  EVI–VOD 季节迟滞回环（相空间轨迹）\n'
                 '回环面积 = EVI-VOD 去同步化程度',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

    # 打印回环面积对比
    print("\n  迟滞回环面积对比：")
    for vtype, area in sorted(loop_areas.items(), key=lambda x: x[1]):
        print(f"    {VEG_LABELS_ZH[vtype]:6s}: {area:.6f}  "
              f"{'(近同步)' if area < 0.001 else '(显著滞后)' if area > 0.003 else ''}")

    return loop_areas


print("\n绘制 Fig.8: 迟滞回环...")
loop_areas = plot_hysteresis_loops(
    df, output_path=OUTPUT_DIR + 'fig8_hysteresis_loop.png')


# ============================================================
# 9. Fig.4 补充 — 像元级 Sen Slope 空间图（需 GeoTIFF）
# ============================================================

def pixel_level_trend_analysis(annual_tif_dir, output_path=None):
    """
    对逐年 GeoTIFF 进行像元级 Sen Slope + Mann-Kendall 趋势分析
    需要 rasterio 和 pymannkendall 库
    """
    try:
        import rasterio
        import pymannkendall as mk
    except ImportError:
        print("  [WARNING] rasterio 或 pymannkendall 未安装，跳过像元级趋势分析")
        print("  运行: pip install rasterio pymannkendall")
        return

    import os
    import glob

    years = range(2010, 2022)
    tif_files = [os.path.join(annual_tif_dir, f'evi_vod_annual_{yr}.tif')
                 for yr in years]

    # 检查文件是否存在
    existing = [f for f in tif_files if os.path.exists(f)]
    if len(existing) < 3:
        print(f"  [WARNING] 仅找到 {len(existing)} 个年度 GeoTIFF，需要至少 3 个")
        print(f"  搜索路径: {annual_tif_dir}")
        return

    # 读取所有年份数据
    evi_stack = []
    vod_stack = []
    bounds = None
    for f in existing:
        with rasterio.open(f) as src:
            data = src.read()  # (bands, rows, cols)
            evi_stack.append(data[0])  # Band 1 = EVI
            vod_stack.append(data[1])  # Band 2 = VOD
            transform = src.transform
            crs = src.crs
            shape = data[0].shape
            if bounds is None:
                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

    evi_stack = np.array(evi_stack)  # (years, rows, cols)
    vod_stack = np.array(vod_stack)
    n_years, n_rows, n_cols = evi_stack.shape
    print(f"  加载 {n_years} 年数据, 空间: {n_rows}x{n_cols} 像元")

    # 逐像元计算 Sen slope + MK 检验
    evi_slope  = np.full((n_rows, n_cols), np.nan)
    evi_pval   = np.full((n_rows, n_cols), np.nan)
    vod_slope  = np.full((n_rows, n_cols), np.nan)
    vod_pval   = np.full((n_rows, n_cols), np.nan)

    total_pixels = n_rows * n_cols
    processed = 0

    for i in range(n_rows):
        for j in range(n_cols):
            evi_ts = evi_stack[:, i, j]
            vod_ts = vod_stack[:, i, j]

            # 跳过含 NaN 的像元
            if np.any(np.isnan(evi_ts)) or np.any(np.isnan(vod_ts)):
                continue
            if np.all(evi_ts == 0) or np.all(vod_ts == 0):
                continue

            try:
                evi_result = mk.original_test(evi_ts)
                evi_slope[i, j] = evi_result.slope
                evi_pval[i, j]  = evi_result.p

                vod_result = mk.original_test(vod_ts)
                vod_slope[i, j] = vod_result.slope
                vod_pval[i, j]  = vod_result.p
            except Exception:
                pass

        processed += n_cols
        if (i + 1) % 10 == 0:
            print(f"    进度: {processed}/{total_pixels} 像元 "
                  f"({processed/total_pixels*100:.1f}%)")

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    # 显著性掩膜
    evi_sig = evi_pval < 0.05
    vod_sig = vod_pval < 0.05

    left, bottom, right, top = bounds
    extent = [left, right, bottom, top]

    # EVI 趋势图
    vmax = max(np.nanpercentile(abs(evi_slope), 95),
               np.nanpercentile(abs(vod_slope), 95))
    if vmax <= 0: vmax = 0.01
    im1 = axes[0].imshow(evi_slope, cmap='RdYlGn', vmin=-vmax, vmax=vmax, extent=extent, zorder=1)
    
    # 显著性区域转换为经纬度加点标注
    lon_grid = np.linspace(left, right, n_cols)
    lat_grid = np.linspace(top, bottom, n_rows) # top is north, bottom is south
    
    sig_y, sig_x = np.where(evi_sig)
    if len(sig_x) > 0:
        sig_lons = lon_grid[sig_x]
        sig_lats = lat_grid[sig_y]
        axes[0].scatter(sig_lons, sig_lats, s=0.4, c='black', alpha=0.35, marker='.', zorder=2)
    axes[0].set_title('Fig.4a  EVI 像元级趋势变化 (·= p<0.05)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Slope (/yr)', shrink=0.7)

    # VOD 趋势图
    im2 = axes[1].imshow(vod_slope, cmap='RdYlGn', vmin=-vmax, vmax=vmax, extent=extent, zorder=1)
    sig_y_v, sig_x_v = np.where(vod_sig)
    if len(sig_x_v) > 0:
        sig_lons_v = lon_grid[sig_x_v]
        sig_lats_v = lat_grid[sig_y_v]
        axes[1].scatter(sig_lons_v, sig_lats_v, s=0.4, c='black', alpha=0.35, marker='.', zorder=2)
    axes[1].set_title('Fig.4b  VOD 像元级趋势变化 (·= p<0.05)', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Slope (/yr)', shrink=0.7)

    for ax in axes:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        format_geo_ticks(ax, bounds)
        add_scale_bar(ax, bounds, scale_length_km=100, scale_loc=(0.06, 0.06))
        add_north_arrow(ax, bounds, arrow_loc=(0.92, 0.82))

    # 统计显著增加/减少面积占比
    evi_sig_inc = np.sum(evi_sig & (evi_slope > 0))
    evi_sig_dec = np.sum(evi_sig & (evi_slope < 0))
    vod_sig_inc = np.sum(vod_sig & (vod_slope > 0))
    vod_sig_dec = np.sum(vod_sig & (vod_slope < 0))
    valid = np.sum(~np.isnan(evi_slope))

    fig.suptitle(
        f'Fig.4(补)  像元级年际趋势对比（2010–2021）\n'
        f'EVI 显著↑: {evi_sig_inc/valid*100:.1f}%, 显著↓: {evi_sig_dec/valid*100:.1f}%  |  '
        f'VOD 显著↑: {vod_sig_inc/valid*100:.1f}%, 显著↓: {vod_sig_dec/valid*100:.1f}%',
        fontsize=12, fontweight='bold', y=1.04)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


print("\n绘制 Fig.4 补充: 像元级 Sen Slope 空间图...")
pixel_level_trend_analysis(
    ANNUAL_TIF_DIR,
    output_path=OUTPUT_DIR + 'fig4_pixel_trend.png')


# ============================================================
# 10. 汇总输出
# ============================================================

print("\n" + "=" * 60)
print("  全部分析完成！产出图表清单：")
print("=" * 60)
print(f"""
  Fig.1  空间格局及解耦指数 DI 空间分布图
  Fig.2  季节动态曲线（双 Y 轴，3 panel）
  Fig.3  互相关 Lag Analysis 柱状图 ⭐
  Fig.4  Sen Slope + MK 趋势（区域平均 + 像元级）
  Fig.5  z-score 异常时序图
  Fig.6  VOD-EVI 饱和效应散点密度图
  Fig.7  Almon DLM 分布滞后权重曲线
  Fig.8  EVI-VOD 迟滞回环（相空间轨迹）

  输出目录: {OUTPUT_DIR}
""")
print("=" * 60)
