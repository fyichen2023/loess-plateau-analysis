# -*- coding: utf-8 -*-
"""
============================================================
 黄土高原退耕还林 EVI–VOD 时空差异性分析
 项目主运行入口 (main.py)
============================================================

 运行该脚本将：
   1. 检查数据输入路径。若未检测到数据，将自动生成高度拟真的黄土高原模拟数据集（测试模式）。
   2. 调用 stats 模块进行季节动态、时滞互相关、Sen+MK趋势、z-score 异常、饱和效应、Almon DLM 及迟滞回环分析。
   3. 调用 plotting 模块渲染 Fig.1 至 Fig.8 出版级图表，并集成地理坐标轴、比例尺、指北针等要素。
   4. 将全部图表保存至 figures 目录中。
"""
import os
import sys

# 将当前目录加入系统路径以支持本地导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loess_plateau_analysis import config
from loess_plateau_analysis import data_loader
from loess_plateau_analysis import stats
from loess_plateau_analysis import plotting
import argparse

def run_pipeline(show_title=True):
    print("=" * 70)
    print("   黄土高原退耕还林 EVI–VOD 遥感时空耦合特性分析主流程启动")
    print("=" * 70)

    # 1. 自动检测并生成模拟测试数据 (防空跑报错设计)
    mock_mode = False
    if not os.path.exists(config.CSV_PATH) or not os.path.exists(config.SPATIAL_TIF):
        print("[System] 提示：在默认路径下未检测到 GEE 导出文件。")
        print(f"         CSV 路径: {config.CSV_PATH}")
        print(f"         TIFF 路径: {config.SPATIAL_TIF}")
        print("         >>> 自动进入【模拟演示模式】，在 data/ 目录下生成黄土高原拟真数据集...")
        try:
            data_loader.generate_mock_datasets(config.DEFAULT_DATA_DIR)
            mock_mode = True
        except Exception as e:
            print(f"[ERROR] 生成模拟数据集失败: {e}")
            sys.exit(1)
            
    # 2. 创建输出图表目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"[System] 图表输出目录已锁定: {config.OUTPUT_DIR}")

    # ============================================================
    # PART 1: 时间序列分析与计算
    # ============================================================
    print("\n--- [PART 1] 正在加载并分析时间序列数据 ---")
    try:
        df = data_loader.load_csv_data(config.CSV_PATH)
        print(f"  成功载入时序数据：共计 {len(df)} 条观测值。")
    except Exception as e:
        print(f"[ERROR] 载入 CSV 数据失败: {e}")
        sys.exit(1)

    # A. 互相关 Lag Analysis
    print("  计算植被分类滞后互相关关系 (Fig.3)...")
    lag_results = {}
    veg_types_to_lag = ['Grassland', 'Shrubland', 'Forest']
    for vtype in veg_types_to_lag:
        # 扩大时滞窗口到 12 个月，以捕获跨越冬季的异相相关
        lag_results[vtype] = stats.compute_cross_correlation(df, vtype, max_lag=12)

    # B. Sen Slope + Mann-Kendall 趋势分析 (区域平均)
    print("  计算区域年际变化趋势及显著性 (Fig.4 区域平均)...")
    trend_results = {}
    veg_types_to_trend = ['Grassland', 'Forest', 'All_Vegetation']
    for vtype in veg_types_to_trend:
        sub = df[df['veg_type'] == vtype]
        annual = sub.groupby('year').agg(
            evi=('EVI', 'mean'), vod=('VOD', 'mean')
        ).reset_index()
        
        # 计算趋势
        evi_trend = stats.run_trend_test(annual['year'].values, annual['evi'].values)
        vod_trend = stats.run_trend_test(annual['year'].values, annual['vod'].values)
        
        trend_results[vtype] = {
            'evi_slope': evi_trend.slope,
            'evi_p': evi_trend.p,
            'vod_slope': vod_trend.slope,
            'vod_p': vod_trend.p
        }

    # C. z-score 异常及事件提取 (Fig.5)
    print("  计算月度标准化气候异常 (z-score) (Fig.5)...")
    anomaly_df = stats.compute_zscore_anomalies(df, 'All_Vegetation')

    # D. Almon 分布滞后多项式拟合 (Fig.7)
    print("  拟合 Almon 分布滞后模型分析生态记忆权重 (Fig.7)...")
    dlm_results = {}
    for vtype in ['Grassland', 'Shrubland', 'Forest']:
        # 增加最大滞后期至 12，多项式阶数至 3，显著提升复杂季节动态下的 R² 解释度
        dlm_results[vtype] = stats.fit_almon_dlm(df, vtype, K=12, poly_degree=3)

    # ============================================================
    # PART 2: 空间栅格影像分析与计算
    # ============================================================
    print("\n--- [PART 2] 正在加载并分析空间栅格图像 ---")
    spatial_data = None
    transform, crs, bounds = None, None, None
    
    if data_loader.RASTERIO_AVAILABLE:
        try:
            spatial_data, transform, crs, bounds = data_loader.load_spatial_tif(config.SPATIAL_TIF)
            print(f"  成功载入多年平均空间数据，影像范围: {bounds}")
        except Exception as e:
            print(f"  [WARNING] 载入多年平均 TIFF 空间图失败: {e}，将跳过 Fig.1 空间图绘制。")
    else:
        print("  [WARNING] rasterio 库未安装，无法进行栅格空间分析。跳过 Fig.1 空间图绘制。")

    # 像元级 Sen+MK 趋势计算 (Fig.4 补充)
    pixel_trend_computed = False
    evi_slope, evi_pval, vod_slope, vod_pval = None, None, None, None
    
    if data_loader.RASTERIO_AVAILABLE:
        try:
            print("  正在读取逐年 GeoTIFF 并计算像元级 Sen Slope + MK 趋势 (Fig.4 空间图)...")
            evi_stack, vod_stack, t_trans, t_crs, t_bounds = data_loader.load_annual_tifs(
                config.ANNUAL_TIF_DIR, 2010, 2021
            )
            evi_slope, evi_pval, vod_slope, vod_pval = stats.compute_pixel_trends(evi_stack, vod_stack)
            pixel_trend_computed = True
            print("  像元级年际趋势计算完毕。")
        except Exception as e:
            print(f"  [WARNING] 像元级趋势分析计算跳过: {e}")
            print("            可能是因为缺少逐年 tif 文件 (evi_vod_annual_YYYY.tif)。")

    # ============================================================
    # PART 3: 可视化与出版级制图生成
    # ============================================================
    print("\n--- [PART 3] 正在生成出版级图表 ---")
    
    # 绘制 Fig.1 空间格局图 (如果成功加载栅格)
    if spatial_data is not None:
        print("  正在绘制并保存 Fig.1: 空间格局及解耦指数 DI 分布图 (含地理坐标、比例尺、指北针)...")
        plotting.plot_fig1_spatial_patterns(spatial_data, transform, crs, bounds, config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.2 季节律动双Y轴折线图
    print("  正在绘制并保存 Fig.2: 三种典型地表季节动态折线图...")
    plotting.plot_fig2_seasonal(df, ['Grassland', 'Shrubland', 'Forest'], config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.3 滞后相关柱状图
    print("  正在绘制并保存 Fig.3: 互相关 Lag Analysis 柱状图...")
    plotting.plot_fig3_lag(lag_results, config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.4 区域趋势折线图
    print("  正在绘制并保存 Fig.4: 区域平均长期趋势对比折线图...")
    plotting.plot_fig4_trends(df, trend_results, config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.4 像元级趋势空间图 (如果成功计算)
    if pixel_trend_computed:
        print("  正在绘制并保存 Fig.4(补): 像元级趋势与显著性空间分布图 (含地理要素)...")
        plotting.plot_fig4_pixel_trends(evi_slope, evi_pval, vod_slope, vod_pval, bounds, config.OUTPUT_DIR, show_title=show_title)
        
    # 绘制 Fig.5 z-score 异常柱状图
    print("  正在绘制并保存 Fig.5: z-score 月度标准化气候异常时序图...")
    plotting.plot_fig5_zscore(anomaly_df, "全部植被", config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.6 饱和效应散点图
    print("  正在绘制并保存 Fig.6: EVI-VOD 饱和效应探究散点密度图...")
    plotting.plot_fig6_saturation(df, config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.7 Almon DLM 滞后权重曲线
    print("  正在绘制并保存 Fig.7: Almon 多项式分布滞后权重曲线...")
    plotting.plot_fig7_dlm(dlm_results, 6, config.OUTPUT_DIR, show_title=show_title)
    
    # 绘制 Fig.8 迟滞回环相空间图
    print("  正在绘制并保存 Fig.8: EVI-VOD 季节迟滞回环图...")
    plotting.plot_fig8_hysteresis(df, ['Grassland', 'Shrubland', 'Forest'], config.OUTPUT_DIR, show_title=show_title)

    print("\n" + "=" * 70)
    print("   分析流程运行成功！")
    print(f"   所有出版级图表已成功保存至目录: {os.path.abspath(config.OUTPUT_DIR)}")
    if mock_mode:
        print("   [Demo Note] 本次运行使用的是自动生成的拟真模拟数据。")
        print("               您只需将 GEE 导出的真实文件放入 data/ 目录替换后，即可直接得到真实分析报告图表！")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="黄土高原退耕还林分析流程")
    parser.add_argument('--no-title', action='store_true', help="生成不带主标题的图表(适合论文排版)")
    args = parser.parse_args()
    
    run_pipeline(show_title=not args.no_title)
