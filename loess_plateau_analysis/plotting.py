# -*- coding: utf-8 -*-
"""
============================================================
 黄土高原退耕还林 EVI–VOD 时空差异性分析
 可视化制图模块 (plotting.py)
============================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats as scipy_stats
from . import config
from . import stats

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def apply_plot_style():
    """应用全局的 matplotlib 绘图样式"""
    plt.rcParams.update(config.PLOT_STYLE)


# ============================================================
# 0. 地图装饰要素工具 (Scale bar, North arrow, Coordinates)
# ============================================================

def format_geo_ticks(ax, bounds):
    """
    格式化地理坐标轴刻度为 E/N 经纬度表示
    """
    left, bottom, right, top = bounds
    
    # 动态确定刻度间隔
    x_range = right - left
    y_range = top - bottom
    
    x_step = 2.0 if x_range > 5 else 1.0
    y_step = 2.0 if y_range > 5 else 1.0
    
    ax.set_xticks(np.arange(np.ceil(left), right, x_step))
    ax.set_yticks(np.arange(np.ceil(bottom), top, y_step))
    
    # 经纬度格式化函数
    def lon_formatter(x, pos):
        return f"{abs(x):.1f}°E" if x >= 0 else f"{abs(x):.1f}°W"
    
    def lat_formatter(y, pos):
        return f"{abs(y):.1f}°N" if y >= 0 else f"{abs(y):.1f}°S"
        
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lat_formatter))
    ax.tick_params(axis='both', labelsize=8)


def add_scale_bar(ax, bounds, scale_length_km=100, scale_loc=(0.08, 0.08), color='black'):
    """
    在图层中绘制简洁线段比例尺 (基于中心纬度的经度公里换算)
    """
    left, bottom, right, top = bounds
    
    # 1. 计算图层中心点的纬度
    lat_center = (bottom + top) / 2.0
    
    # 2. 根据大圆公式计算该纬度下的经纬度与公里换算
    # 纬度 1度 恒约 111.32 km
    # 经度 1度 约 111.32 * cos(lat) km
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_center))
    
    # 3. 确定比例尺线段在数据坐标系(经纬度度数)中的长度
    length_deg_lon = scale_length_km / km_per_deg_lon
    
    # 4. 根据相对 Axes 比例的 scale_loc 确定起点地理坐标
    lon_start = left + scale_loc[0] * (right - left)
    lat_start = bottom + scale_loc[1] * (top - bottom)
    lon_end = lon_start + length_deg_lon
    
    # 5. 绘制比例尺主线与两端刻度线
    ax.plot([lon_start, lon_end], [lat_start, lat_start], color=color, lw=1.5, zorder=5)
    
    tick_height = 0.02 * (top - bottom)
    ax.plot([lon_start, lon_start], [lat_start, lat_start + tick_height], color=color, lw=1.5, zorder=5)
    ax.plot([lon_end, lon_end], [lat_start, lat_start + tick_height], color=color, lw=1.5, zorder=5)
    
    # 6. 标注文本
    ax.text((lon_start + lon_end) / 2.0, lat_start + tick_height * 0.7, 
            f"{scale_length_km} km", ha='center', va='bottom', 
            fontsize=8, fontweight='bold', color=color, zorder=6)


def add_north_arrow(ax, bounds, arrow_loc=(0.9, 0.85), color='black'):
    """
    在图层中绘制精美的指北针
    """
    left, bottom, right, top = bounds
    
    # 计算起点和终点地理坐标
    lon_arrow = left + arrow_loc[0] * (right - left)
    lat_arrow = bottom + arrow_loc[1] * (top - bottom)
    
    arrow_len = 0.06 * (top - bottom)
    
    # 使用 FancyArrowPatch 绘制一个简洁指北针
    arrow = FancyArrowPatch(
        (lon_arrow, lat_arrow), (lon_arrow, lat_arrow + arrow_len),
        arrowstyle='->', mutation_scale=12, color=color, lw=1.5, zorder=5
    )
    ax.add_patch(arrow)
    
    # 在尖端写 'N'
    ax.text(lon_arrow, lat_arrow + arrow_len + 0.01 * (top - bottom), 'N',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=color, zorder=6)


# ============================================================
# 1. Fig.1 — 空间格局对比图 (EVI, VOD, DI, IGBP)
# ============================================================

def plot_fig1_spatial_patterns(data_dict, transform, crs, bounds, output_dir=None, show_title=True):
    """
    绘制 Fig.1: 多年平均 EVI、VOD、DI 与 IGBP 分类空间分布对比图 (4-panel)
    并添加地图装饰要素 (比例尺、指北针、经纬度刻度)
    """
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 确定经纬度边界
    left, bottom, right, top = bounds
    extent = [left, right, bottom, top]
    
    # ------------------ 1a. EVI 空间分布 ------------------
    im_evi = axes[0].imshow(data_dict['EVI'], cmap='YlGn', extent=extent, zorder=1)
    axes[0].set_title("(a) 多年平均 EVI 绿度分布", fontweight='bold')
    plt.colorbar(im_evi, ax=axes[0], label='EVI', shrink=0.7)
    
    # ------------------ 1b. VOD 空间分布 ------------------
    im_vod = axes[0].imshow(data_dict['VOD'], cmap='YlOrRd', extent=extent, zorder=1) # 临时画，下面覆盖，只为了方便共用参数
    axes[1].imshow(data_dict['VOD'], cmap='YlOrRd', extent=extent, zorder=1)
    axes[1].set_title("(b) 多年平均 VODCA L-band VOD 分布", fontweight='bold')
    plt.colorbar(im_vod, ax=axes[1], label='VOD (neper)', shrink=0.7)
    
    # ------------------ 1c. DI 解耦指数空间分布 ------------------
    # DI = VOD_norm - EVI_norm, min-max 归一化后理论在 [-1, 1], 取 [-0.5, 0.5] 高亮背离
    im_di = axes[2].imshow(data_dict['DI'], cmap='RdYlGn', vmin=-0.5, vmax=0.5, extent=extent, zorder=1)
    axes[2].set_title("(c) 植被绿化-结构恢复解耦指数 (DI)", fontweight='bold')
    plt.colorbar(im_di, ax=axes[2], label='DI (VOD_norm - EVI_norm)', shrink=0.7)
    
    # ------------------ 1d. IGBP 土地分类 ------------------
    igbp_data = data_dict['IGBP']
    
    # 提取区域内实际存在的 IGBP 类别并映射颜色
    present_classes = np.unique(igbp_data[~np.isnan(igbp_data) & (igbp_data > 0)]).astype(int)
    
    # 构造离散配色卡
    cmap_colors = [config.IGBP_COLORS.get(cl, '#ffffff') for cl in present_classes]
    discrete_cmap = mcolors.ListedColormap(cmap_colors)
    
    # 转换数值映射
    class_mapping = {cl: i for i, cl in enumerate(present_classes)}
    mapped_igbp = np.full(igbp_data.shape, np.nan)
    for cl, i in class_mapping.items():
        mapped_igbp[igbp_data == cl] = i
        
    axes[3].imshow(mapped_igbp, cmap=discrete_cmap, extent=extent, zorder=1)
    axes[3].set_title("(d) MCD12C1 IGBP 土地覆盖类型", fontweight='bold')
    
    # 绘制 IGBP 图例
    legend_patches = []
    for cl in present_classes:
        name_zh = config.IGBP_NAMES.get(cl, f"Class {cl}")
        # 部分类别翻译为简短中文，保持图例美观
        zh_map = {
            1: "针叶常绿林 (ENF)", 4: "阔叶落叶林 (DBF)", 5: "混交林 (MF)", 
            7: "开阔灌木林 (OSH)", 8: "稀树木质草原 (WSA)", 9: "稀树草原 (SAV)", 
            10: "草地 (Grasslands)", 12: "农田 (Croplands)", 
            13: "城镇与建成区 (Urban)", 14: "农田/自然植被镶嵌", 16: "裸地 (Barren)"
        }
        name_label = zh_map.get(cl, name_zh)
        patch = mpatches.Patch(color=config.IGBP_COLORS.get(cl, '#ffffff'), label=name_label)
        legend_patches.append(patch)
        
    axes[3].legend(handles=legend_patches, loc='upper left', fontsize=7, framealpha=0.8,
                   bbox_to_anchor=(0.02, 0.98))

    # ------------------ 地图装饰要素添加 ------------------
    for ax in axes:
        # 1. 裁剪图面范围
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        # 2. 地理轴格式化
        format_geo_ticks(ax, bounds)
        # 3. 比例尺 (100km)
        add_scale_bar(ax, bounds, scale_length_km=100, scale_loc=(0.06, 0.06))
        # 4. 指北针
        add_north_arrow(ax, bounds, arrow_loc=(0.92, 0.92))
        
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig1_spatial_patterns.png'), dpi=300)
    plt.close('all')


# ============================================================
# 2. Fig.2 — 季节动态曲线图 (双 Y 轴, 3-panel)
# ============================================================

def plot_fig2_seasonal(df, veg_types=None, output_dir=None, show_title=True):
    """
    绘制 Fig.2: 典型地表类型 EVI-VOD 季节均值与标准差对比图
    """
    apply_plot_style()
    if veg_types is None:
        veg_types = ['Grassland', 'Shrubland', 'Forest']
        
    fig, axes = plt.subplots(1, len(veg_types), figsize=(5 * len(veg_types), 4.5), sharey=False)
    
    for i, (ax, vtype) in enumerate(zip(axes, veg_types)):
        monthly = stats.compute_seasonal_climatology(df, vtype)
        months = monthly['month'].values
        
        # 1. EVI (左侧轴)
        color_evi = '#2ca02c'
        ax.fill_between(months,
                        monthly['evi_mean'] - monthly['evi_std'],
                        monthly['evi_mean'] + monthly['evi_std'],
                        alpha=0.12, color=color_evi)
        ln1 = ax.plot(months, monthly['evi_mean'], 'o-',
                      color=color_evi, linewidth=2, markersize=5, label='EVI')
        ax.set_ylabel('EVI', color=color_evi)
        ax.tick_params(axis='y', labelcolor=color_evi)
        
        # 2. VOD (右侧双Y轴)
        ax2 = ax.twinx()
        color_vod = '#d62728'
        ax2.fill_between(months,
                         monthly['vod_mean'] - monthly['vod_std'],
                         monthly['vod_mean'] + monthly['vod_std'],
                         alpha=0.12, color=color_vod)
        ln2 = ax2.plot(months, monthly['vod_mean'], 's--',
                       color=color_vod, linewidth=2, markersize=5, label='VOD')
        ax2.set_ylabel('VOD (neper)', color=color_vod)
        ax2.tick_params(axis='y', labelcolor=color_vod)
        ax2.spines['right'].set_visible(True)
        
        # 3. 统计相位差
        evi_peak = months[np.argmax(monthly['evi_mean'])]
        vod_peak = months[np.argmax(monthly['vod_mean'])]
        lag = vod_peak - evi_peak
        
        ax.set_title(f"({chr(97+i)}) {config.VEG_LABELS_ZH[vtype]} ({vtype})\n"
                     f"EVI 峰值: {config.MONTH_LABELS[evi_peak-1]}月, "
                     f"VOD 峰值: {config.MONTH_LABELS[vod_peak-1]}月\n"
                     f"时滞相位差: {lag} 个月", fontsize=10, fontweight='bold')
                     
        ax.set_xlabel('Month')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(config.MONTH_LABELS, fontsize=8)
        
        # 4. 图例合并
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', framealpha=0.8, fontsize=9)
        
    if show_title:
        fig.suptitle('Fig.2  典型植被覆盖类型的 EVI 与 VOD 季节律动物候对比（2010–2021均值 ± 1σ）',
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig2_seasonal_dynamics.png'))
    plt.close('all')


# ============================================================
# 3. Fig.3 — 互相关滞后分析图 (Lag Analysis)
# ============================================================

def plot_fig3_lag(results, output_dir=None, show_title=True):
    """
    绘制 Fig.3: 互相关系数随滞后月数变化的柱状图。高亮最大相关系数所在的滞后月。
    """
    apply_plot_style()
    veg_types = list(results.keys())
    n_types = len(veg_types)
    
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4.2), sharey=True)
    if n_types == 1:
        axes = [axes]
        
    for i, (ax, vtype) in enumerate(zip(axes, veg_types)):
        res = results[vtype]
        lags = res['lags']
        corrs = res['corrs']
        pvals = res['pvals']
        
        # 配色策略：最优 Lag 高亮红色，其余显著高亮蓝色，不显著显示灰色
        colors = []
        for i, lag in enumerate(lags):
            if lag == res['best_lag']:
                colors.append('#e74c3c')
            elif pvals[i] < 0.01:
                colors.append('#3498db')
            elif pvals[i] < 0.05:
                colors.append('#85c1e9')
            else:
                colors.append('#bdc3c7')
                
        bars = ax.bar(lags, corrs, color=colors, edgecolor='white', linewidth=0.5, zorder=2)
        
        # 添加指引线
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', zorder=1)
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':', zorder=1)
        
        # 标注最优 Lag
        ax.annotate(
            f"最优滞后 = {res['best_lag']}月\nr = {res['best_r']:.3f}\np < {res['best_p']:.1e}",
            xy=(res['best_lag'], res['best_r']),
            xytext=(res['best_lag'] + (1.5 if res['best_lag'] < 3 else -4.0), res['best_r'] - 0.08),
            fontsize=8, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor='#e74c3c', alpha=0.9),
            zorder=6
        )
        
        ax.set_xlabel('滞后阶数 (月)\n← VOD领先 | EVI领先 →', fontsize=9)
        ax.set_title(f"({chr(97+i)}) {config.VEG_LABELS_ZH[vtype]} ({vtype})", fontweight='bold', fontsize=11)
        ax.set_xticks(lags)
        ax.grid(axis='y', alpha=0.3)
        
    axes[0].set_ylabel('Pearson 相关系数 r', fontsize=11)
    if show_title:
        fig.suptitle('Fig.3  光学-微波双指标月度时间序列滞后互相关系数 (Lag Analysis)',
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig3_lag_analysis.png'))
    plt.close('all')


# ============================================================
# 4. Fig.4 — 趋势分析图 (区域平均趋势 & 像元级趋势图)
# ============================================================

def plot_fig4_trends(df, trend_results, output_dir=None, show_title=True):
    """
    绘制 Fig.4 (区域平均趋势图): 4种地表类型 EVI 与 VOD 的长期变化趋势。
    """
    apply_plot_style()
    veg_types = ['Grassland', 'Forest', 'All_Vegetation']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=False)
    
    for i, (ax, vtype) in enumerate(zip(axes, veg_types)):
        sub = df[df['veg_type'] == vtype]
        annual = sub.groupby('year').agg(
            evi=('EVI', 'mean'), vod=('VOD', 'mean')
        ).reset_index()
        
        years = annual['year'].values
        res = trend_results[vtype]
        
        # 1. 绘制 EVI 散点与折线
        color_evi = '#2ca02c'
        ax.plot(years, annual['evi'], 'o-', color=color_evi, lw=1.8, ms=5, label='EVI')
        # EVI 拟合趋势线
        # 我们用 intercept 和 slope 绘制
        # 拟合方程: Y = slope * (X - 2010) + intercept
        # 注意: GEE 年均值 OLS/Sen slope 的 intercept 可能是 2010 年的截距，也可能是常规 Y 轴截距
        # stats.py 算出来的是对真实年份 X 的拟合，此处我们统一计算直线：
        # 如果是 FallbackTrendResult，其 intercept 对应 X=0。如果是 mk，pymannkendall 的 intercept 通常也是对应 X=0 的
        # 为了防错，直接现场重新计算一维回归截距来绘图：
        evi_slope, evi_inter = np.polyfit(years, annual['evi'], 1)
        ax.plot(years, evi_slope * years + evi_inter, '--', color=color_evi, alpha=0.6)
        ax.set_ylabel('EVI', color=color_evi)
        ax.tick_params(axis='y', labelcolor=color_evi)
        
        # 2. 绘制 VOD (右侧轴)
        ax2 = ax.twinx()
        color_vod = '#d62728'
        ax2.plot(years, annual['vod'], 's-', color=color_vod, lw=1.8, ms=5, label='VOD')
        vod_slope, vod_inter = np.polyfit(years, annual['vod'], 1)
        ax2.plot(years, vod_slope * years + vod_inter, '--', color=color_vod, alpha=0.6)
        ax2.set_ylabel('VOD', color=color_vod)
        ax2.tick_params(axis='y', labelcolor=color_vod)
        ax2.spines['right'].set_visible(True)
        
        # 显著性星号标注
        def get_sig_star(p):
            return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            
        evi_sig = get_sig_star(res['evi_p'])
        vod_sig = get_sig_star(res['vod_p'])
        
        ax.set_title(
            f"({chr(97+i)}) {config.VEG_LABELS_ZH[vtype]} ({vtype})\n"
            f"EVI 趋势: {res['evi_slope']:.4f}/yr ({evi_sig}) | "
            f"VOD 趋势: {res['vod_slope']:.4f}/yr ({vod_sig})",
            fontsize=10, fontweight='bold'
        )
        
        ax.set_xlabel('Year')
            
    if show_title:
        fig.suptitle('Fig.4  2010–2021 年黄土高原 EVI 与 VOD 年际演变趋势（区域平均）',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig4_trend_regional.png'))
    plt.close('all')


def plot_fig4_pixel_trends(evi_slope, evi_pval, vod_slope, vod_pval, bounds, output_dir=None, show_title=True):
    """
    绘制 Fig.4(补): 像元级 EVI 与 VOD 的 Sen's Slope 变化趋势及显著性空间分布图。
    显著像元 (p < 0.05) 会以黑点进行叠加覆盖以表示显著性。
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    
    left, bottom, right, top = bounds
    extent = [left, right, bottom, top]
    
    # 确定最大斜率，保证两个子图的 colorbar 刻度对称一致
    vmax = max(np.nanpercentile(abs(evi_slope), 95), np.nanpercentile(abs(vod_slope), 95))
    if vmax <= 0: vmax = 0.01
    
    # 1. EVI 像元级趋势
    im1 = axes[0].imshow(evi_slope, cmap='RdYlGn', vmin=-vmax, vmax=vmax, extent=extent, zorder=1)
    # 叠加显著性黑点
    sig_mask_evi = (evi_pval < 0.05) & (~np.isnan(evi_slope))
    sig_y, sig_x = np.where(sig_mask_evi)
    # 注意，从 matrix index 映射到地理坐标
    # 在 extent 中，我们需要直接获取对应的经纬度值
    n_rows, n_cols = evi_slope.shape
    lon_grid = np.linspace(left, right, n_cols)
    lat_grid = np.linspace(top, bottom, n_rows) # 栅格矩阵行从上往下，对应纬度从北往南
    
    if len(sig_x) > 0:
        sig_lons = lon_grid[sig_x]
        sig_lats = lat_grid[sig_y]
        axes[0].scatter(sig_lons, sig_lats, s=0.4, c='black', alpha=0.35, marker='.', zorder=2)
        
    axes[0].set_title("(a) EVI 像元级趋势变化 (·表示 p<0.05)", fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='EVI Trend (/yr)', shrink=0.7)
    
    # 2. VOD 像元级趋势
    im2 = axes[1].imshow(vod_slope, cmap='RdYlGn', vmin=-vmax, vmax=vmax, extent=extent, zorder=1)
    sig_mask_vod = (vod_pval < 0.05) & (~np.isnan(vod_slope))
    sig_y_v, sig_x_v = np.where(sig_mask_vod)
    if len(sig_x_v) > 0:
        sig_lons_v = lon_grid[sig_x_v]
        sig_lats_v = lat_grid[sig_y_v]
        axes[1].scatter(sig_lons_v, sig_lats_v, s=0.4, c='black', alpha=0.35, marker='.', zorder=2)
        
    axes[1].set_title("(b) VOD 像元级趋势变化 (·表示 p<0.05)", fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='VOD Trend (/yr)', shrink=0.7)
    
    # 地图装饰
    for ax in axes:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        format_geo_ticks(ax, bounds)
        add_scale_bar(ax, bounds, scale_length_km=100, scale_loc=(0.06, 0.06))
        add_north_arrow(ax, bounds, arrow_loc=(0.92, 0.92))
        
    # 计算显著上升/下降的面积占比，在标题中说明
    valid_pixels = np.sum(~np.isnan(evi_slope))
    if valid_pixels > 0:
        evi_inc_pct = np.sum((evi_pval < 0.05) & (evi_slope > 0)) / valid_pixels * 100.0
        evi_dec_pct = np.sum((evi_pval < 0.05) & (evi_slope < 0)) / valid_pixels * 100.0
        vod_inc_pct = np.sum((vod_pval < 0.05) & (vod_slope > 0)) / valid_pixels * 100.0
        vod_dec_pct = np.sum((vod_pval < 0.05) & (vod_slope < 0)) / valid_pixels * 100.0
    else:
        evi_inc_pct = evi_dec_pct = vod_inc_pct = vod_dec_pct = 0.0
        
    if show_title:
        fig.suptitle(
        f"Fig.4(补)  2010–2021 年像元级趋势对比 (Sen Slope + MK 检验)\n"
        f"EVI 显著增加面积: {evi_inc_pct:.1f}%, 显著减少: {evi_dec_pct:.1f}% | "
        f"VOD 显著增加面积: {vod_inc_pct:.1f}%, 显著减少: {vod_dec_pct:.1f}%",
        fontsize=12, fontweight='bold', y=1.04
    )
    
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig4_trend_spatial.png'), dpi=300)
    plt.close('all')


# ============================================================
# 5. Fig.5 — z-score 异常时间序列图 (数据驱动异常识别)
# ============================================================

def plot_fig5_zscore(anomaly_df, veg_label_zh="全部植被", output_dir=None, show_title=True):
    """
    绘制 Fig.5: 12年双指标月度标准化异常 (z-score) 柱状图。
    自动检测并红色阴影高亮双指标同时负异常 (<-1) 的极显著生态胁迫时段。
    """
    apply_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6.8), sharex=True)
    
    dates = anomaly_df['date'].values
    
    # ------------------ 5a. EVI Anomalies ------------------
    ax1.fill_between(dates, -1.0, 1.0, alpha=0.08, color='green', label='±1σ 正常气候波幅')
    
    # 配色：正为绿，负为红
    evi_colors = np.where(anomaly_df['EVI_z'] < 0, '#e74c3c', '#27ae60')
    ax1.bar(dates, anomaly_df['EVI_z'], width=pd.Timedelta(days=20), color=evi_colors, alpha=0.75, edgecolor='none')
    
    ax1.axhline(0, color='gray', lw=0.6)
    ax1.axhline(-1, color='gray', lw=0.5, linestyle='--', alpha=0.6)
    ax1.axhline(1, color='gray', lw=0.5, linestyle='--', alpha=0.6)
    ax1.set_ylabel('EVI z-score', fontsize=11)
    ax1.set_title("(a) EVI 植被绿度月度异常 (Greening Anomaly)", fontweight='bold', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(-3.5, 3.5)
    
    # ------------------ 5b. VOD Anomalies ------------------
    ax2.fill_between(dates, -1.0, 1.0, alpha=0.08, color='orange', label='±1σ 正常气候波幅')
    
    vod_colors = np.where(anomaly_df['VOD_z'] < 0, '#e74c3c', '#2980b9')
    ax2.bar(dates, anomaly_df['VOD_z'], width=pd.Timedelta(days=20), color=vod_colors, alpha=0.75, edgecolor='none')
    
    ax2.axhline(0, color='gray', lw=0.6)
    ax2.axhline(-1, color='gray', lw=0.5, linestyle='--', alpha=0.6)
    ax2.axhline(1, color='gray', lw=0.5, linestyle='--', alpha=0.6)
    ax2.set_ylabel('VOD z-score', fontsize=11)
    ax2.set_title("(b) VOD 植被结构月度异常 (Structural Anomaly)", fontweight='bold', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(-3.5, 3.5)
    
    # ------------------ 高亮最强异常月份与阶段 ------------------
    worst_idx = anomaly_df['anomaly_score'].idxmin()
    worst_date = anomaly_df.loc[worst_idx, 'date']
    worst_score = anomaly_df.loc[worst_idx, 'anomaly_score']
    
    # 高亮所有 double negative (<-1) 像元
    anomaly_dates = anomaly_df[anomaly_df['both_negative']]['date']
    for d in anomaly_dates:
        for ax in [ax1, ax2]:
            ax.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15),
                       color='#f1948a', alpha=0.25, zorder=0)
            
    # 标注最强异常事件
    for ax in [ax1, ax2]:
        ax.axvline(worst_date, color='red', lw=1.2, linestyle='-', zorder=4)
        
    ax1.annotate(
        f"历史最强异常波谷\n({worst_date.strftime('%Y-%m')})",
        xy=(worst_date, anomaly_df.loc[worst_idx, 'EVI_z']),
        xytext=(worst_date - pd.Timedelta(days=365), -2.8),
        fontsize=9, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef9e7', edgecolor='red', alpha=0.9)
    )
    
    ax2.annotate(
        f"历史最强异常波谷\n({worst_date.strftime('%Y-%m')})",
        xy=(worst_date, anomaly_df.loc[worst_idx, 'VOD_z']),
        xytext=(worst_date - pd.Timedelta(days=365), -2.8),
        fontsize=9, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef9e7', edgecolor='red', alpha=0.9)
    )
    
    if show_title:
        fig.suptitle(f'Fig.5  2010–2021 年 EVI 与 VOD 月度标准化异常值 (z-score) 时序分布图\n'
                 f'（基于{veg_label_zh}区域平均计算，红色阴影为双重极端负异常段）',
                 fontsize=13, fontweight='bold', y=1.02)
                 
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig5_zscore_anomalies.png'))
    plt.close('all')


# ============================================================
# 6. Fig.6 — 饱和效应散点密度图
# ============================================================

def plot_fig6_saturation(df, output_dir=None, show_title=True):
    """
    绘制 Fig.6: VOD vs EVI 二维分类散点图。
    观察在高生物量(高 VOD)下，EVI 是否表现出饱和效应的渐近线趋势。
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    
    veg_types = ['Forest', 'Grassland', 'Cropland']
    
    for vtype in veg_types:
        sub = df[df['veg_type'] == vtype]
        ax.scatter(
            sub['VOD'], sub['EVI'],
            c=config.VEG_COLORS[vtype],
            alpha=0.35, s=12, edgecolors='none',
            label=f"{config.VEG_LABELS_ZH[vtype]} (n={len(sub)})"
        )
        
    ax.set_xlabel('植被含水骨架 VOD (neper)', fontsize=12)
    ax.set_ylabel('光学绿度指数 EVI', fontsize=12)
    if show_title:
        ax.set_title("EVI 与 VOD 的二维散点对照：光学饱和效应探究", fontsize=13, fontweight='bold')
    
    # 绘制拟合渐近线（以指数或对数趋势线定性示意饱和趋势）
    # 筛选全部数据拟合
    valid_df = df[df['veg_type'].isin(veg_types)].dropna()
    x = valid_df['VOD'].values
    y = valid_df['EVI'].values
    
    # 用对数曲线定性描绘 EVI 的饱和渐近线: Y = a * ln(X) + b
    log_x = np.log(x)
    slope, intercept, _, _, _ = scipy_stats.linregress(log_x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * np.log(x_fit) + intercept
    
    ax.plot(x_fit, y_fit, '-', color='#7f8c8d', lw=2.0, linestyle='--', label='趋势渐近线')
    
    # 标注饱和饱和区
    ax.annotate(
        "EVI 渐近饱和区\n(高生物量木质部增加，\n而叶片绿度不再显著上升)",
        xy=(0.42, 0.46), xytext=(0.28, 0.54),
        fontsize=9, fontweight='bold', color='#7f8c8d',
        arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5, connectionstyle="arc3,rad=-0.1")
    )
    
    ax.legend(loc='lower right', markerscale=3.0, framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig6_saturation_scatter.png'))
    plt.close('all')


# ============================================================
# 7. Fig.7 — Almon 分布滞后权重图 (Discussion 5.1)
# ============================================================

def plot_fig7_dlm(results, K=6, output_dir=None, show_title=True):
    """
    绘制 Fig.7: Almon 多项式分布滞后模型 (PDL) 得到的各历史月份 EVI 对当前 VOD 的贡献权重。
    """
    apply_plot_style()
    veg_types = list(results.keys())
    n_types = len(veg_types)
    
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4.2), sharey=True)
    if n_types == 1:
        axes = [axes]
        
    for i, (ax, vtype) in enumerate(zip(axes, veg_types)):
        res = results[vtype]
        k_vals = res['k_vals']
        beta = res['beta']
        
        # 绘制权重柱状图
        ax.bar(k_vals, beta, color=config.VEG_COLORS[vtype], alpha=0.5, edgecolor='white', width=0.6, zorder=2)
        
        # 绘制平滑多项式拟合曲线
        k_smooth = np.linspace(0, K, 100)
        # 用 poly1d 表达二次多项式
        poly_coeff = np.polyfit(k_vals, beta, 2)
        poly_fn = np.poly1d(poly_coeff)
        ax.plot(k_smooth, poly_fn(k_smooth), '-', color=config.VEG_COLORS[vtype], lw=2.2, zorder=3)
        
        ax.axhline(0, color='gray', lw=0.6, linestyle='--', zorder=1)
        ax.set_xticks(range(K + 1))
        ax.set_xlabel('滞后时长 k (月)', fontsize=10)
        
        ax.set_title(f"({chr(97+i)}) {config.VEG_LABELS_ZH[vtype]} ({vtype})\n"
                     f"最优贡献时滞 = {res['peak_lag']} 个月\nR² = {res['r_squared']:.3f}",
                     fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
    axes[0].set_ylabel(r'DLM 权重系数 $\beta_k$ (历史绿度对当前骨架贡献)', fontsize=11)
    if show_title:
        fig.suptitle('Fig.7  Almon 二阶多项式约束分布滞后回归权重：生态记忆（Ecological Memory）特征',
                 fontsize=13, fontweight='bold', y=1.04)
                 
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig7_almon_dlm_weights.png'))
    plt.close('all')


# ============================================================
# 8. Fig.8 — EVI-VOD 迟滞回环图 (Discussion 5.2)
# ============================================================

def plot_fig8_hysteresis(df, veg_types=None, output_dir=None, show_title=True):
    """
    绘制 Fig.8: EVI-VOD 季节迟滞回环（相空间轨迹图）。
    计算其封闭多边形面积并标记旋转方向（顺时针/逆时针），解释两种信号物理上的去同步化特征。
    """
    apply_plot_style()
    if veg_types is None:
        veg_types = ['Grassland', 'Shrubland', 'Forest']
        
    fig, axes = plt.subplots(1, len(veg_types), figsize=(5.5 * len(veg_types), 5.0), sharey=True)
    if len(veg_types) == 1:
        axes = [axes]
        
    for i, (ax, vtype) in enumerate(zip(axes, veg_types)):
        # 1. 整理多年月均值
        sub = df[df['veg_type'] == vtype]
        monthly = sub.groupby('month').agg(
            evi=('EVI', 'mean'), vod=('VOD', 'mean')
        ).reset_index()
        
        evi_raw = monthly['evi'].values
        vod_raw = monthly['vod'].values
        months = monthly['month'].values
        
        # 将 EVI 和 VOD 归一化到 [0, 1] 区间，以消除量纲影响，计算无量纲的相对回环面积
        evi = (evi_raw - np.nanmin(evi_raw)) / (np.nanmax(evi_raw) - np.nanmin(evi_raw))
        vod = (vod_raw - np.nanmin(vod_raw)) / (np.nanmax(vod_raw) - np.nanmin(vod_raw))
        
        # 闭合曲线
        evi_closed = np.append(evi, evi[0])
        vod_closed = np.append(vod, vod[0])
        
        # 2. 计算 Shoelace 面积
        area = stats.calculate_loop_area(evi, vod)
        
        # 3. 绘制轨迹
        color = config.VEG_COLORS[vtype]
        ax.plot(evi_closed, vod_closed, '-', color=color, lw=2.5, alpha=0.85, zorder=2)
        ax.fill(evi_closed, vod_closed, color=color, alpha=0.08, zorder=1)
        
        # 4. 绘制各月数据点并标明月份
        for i, m in enumerate(months):
            ax.plot(evi[i], vod[i], 'o', color=color, markersize=10, zorder=3, 
                    markeredgecolor='white', markeredgewidth=1.2)
            ax.text(evi[i], vod[i], str(m), fontsize=7, fontweight='bold',
                    color='white', ha='center', va='center', zorder=4)
            
        # 5. 标明一月份往二月份移动的方向箭头 (定性描绘迟滞闭环顺时针演变)
        # 取 4月和 5月 之间的切向箭头
        mid_x = (evi[3] + evi[4]) / 2.0
        mid_y = (vod[3] + vod[4]) / 2.0
        dx = evi[4] - evi[3]
        dy = vod[4] - vod[3]
        ax.annotate('', xy=(mid_x + dx * 0.25, mid_y + dy * 0.25), 
                    xytext=(mid_x - dx * 0.25, mid_y - dy * 0.25),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0, shrinkA=0, shrinkB=0),
                    zorder=5)
                    
        # 6. 标注落叶期 (9月和 10月) 之间的切向箭头 (落叶期)
        mid_x2 = (evi[8] + evi[9]) / 2.0
        mid_y2 = (vod[8] + vod[9]) / 2.0
        dx2 = evi[9] - evi[8]
        dy2 = vod[9] - vod[8]
        ax.annotate('', xy=(mid_x2 + dx2 * 0.25, mid_y2 + dy2 * 0.25), 
                    xytext=(mid_x2 - dx2 * 0.25, mid_y2 - dy2 * 0.25),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0, shrinkA=0, shrinkB=0),
                    zorder=5)

        ax.set_xlabel('归一化 EVI (绿度)', fontsize=11)
        ax.set_title(f"({chr(97+i)}) {config.VEG_LABELS_ZH[vtype]} ({vtype})\n"
                     f"归一化回环面积 = {area:.3f}",
                     fontsize=10.5, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    axes[0].set_ylabel('归一化 VOD (结构/水分)', fontsize=11)
    
    if show_title:
        fig.suptitle('Fig.8  多年月平均 EVI 与 VOD 的季节相位迟滞回环（相空间封闭轨迹）',
                 fontsize=13, fontweight='bold', y=1.04)
                 
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'fig8_hysteresis_loop.png'))
    plt.close('all')
