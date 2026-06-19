# -*- coding: utf-8 -*-
"""
============================================================
 黄土高原退耕还林 EVI–VOD 时空差异性分析
 数据加载模块 (data_loader.py)
============================================================
"""
import os
import numpy as np
import pandas as pd

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_csv_data(csv_path):
    """
    加载 GEE 导出的月度时间序列 CSV，并进行缺失值线性插值。
    
    参数:
        csv_path (str): CSV文件路径
        
    返回:
        pd.DataFrame: 整理并补齐后的时间序列 Dataframe，包含列 ['date', 'year', 'month', 'veg_type', 'EVI', 'VOD']
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 CSV 格式的时间序列数据: {csv_path}。请确保已运行 GEE 脚本并导出数据。")
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )
    df = df.sort_values(['veg_type', 'date']).reset_index(drop=True)

    filled = []
    for vtype in df['veg_type'].unique():
        sub = df[df['veg_type'] == vtype].set_index('date')
        
        # 重建完整的月度时间索引，防止 GEE 导出时由于某些月份完全无数据而缺失行
        full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq='MS')
        sub = sub.reindex(full_idx)
        
        # 线性插值补全缺失值，使用 limit_direction='both' 补全首尾缺失
        sub['EVI'] = sub['EVI'].interpolate(method='linear', limit_direction='both')
        sub['VOD'] = sub['VOD'].interpolate(method='linear', limit_direction='both')
        sub['veg_type'] = vtype
        sub['year']  = sub.index.year
        sub['month'] = sub.index.month
        
        sub = sub.reset_index().rename(columns={'index': 'date'})
        filled.append(sub)

    return pd.concat(filled, ignore_index=True)


def load_spatial_tif(spatial_tif_path):
    """
    读取多年均值 GeoTIFF 文件 (EVI, VOD, DI, IGBP)。
    
    参数:
        spatial_tif_path (str): GeoTIFF 路径
        
    返回:
        tuple: (data_dict, transform, crs, bounds)
            - data_dict: 包含 'EVI', 'VOD', 'DI', 'IGBP' 2D numpy 数组的字典
            - transform: rasterio.Affine 转换矩阵
            - crs: 投影坐标系
            - bounds: 空间四至 (left, bottom, right, top)
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("未安装 'rasterio' 库，无法读取空间 GeoTIFF 图像。请运行 `pip install rasterio` 进行安装。")
        
    if not os.path.exists(spatial_tif_path):
        raise FileNotFoundError(f"未找到多年均值空间 TIFF 文件: {spatial_tif_path}")

    with rasterio.open(spatial_tif_path) as src:
        # 依次读取波段（对应 GEE 导出顺序）
        # Band 1: EVI
        # Band 2: VOD
        # Band 3: DI
        # Band 4: IGBP
        data_dict = {
            'EVI': src.read(1),
            'VOD': src.read(2),
            'DI': src.read(3),
            'IGBP': src.read(4)
        }
        
        # 替换 nodata 为 np.nan
        nodata = src.nodata
        for key in data_dict:
            if nodata is not None:
                mask = (data_dict[key] == nodata)
                data_dict[key] = data_dict[key].astype(float)
                data_dict[key][mask] = np.nan
                
        transform = src.transform
        crs = src.crs
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        
    return data_dict, transform, crs, bounds


def load_annual_tifs(dir_path, start_year=2010, end_year=2021):
    """
    批量加载逐年 GeoTIFF 并堆叠为 3D numpy 数组，用于像元级趋势分析。
    
    参数:
        dir_path (str): 存储年均值 TIFF 文件的文件夹目录
        start_year (int): 起始年份
        end_year (int): 结束年份
        
    返回:
        tuple: (evi_stack, vod_stack, transform, crs, bounds)
            - evi_stack/vod_stack: (n_years, n_rows, n_cols) 3D numpy 数组
            - transform: 仿射变换矩阵
            - crs: 坐标系
            - bounds: 空间范围
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("未安装 'rasterio' 库，无法读取空间 GeoTIFF 图像。")
        
    years = range(start_year, end_year + 1)
    tif_files = [os.path.join(dir_path, f'evi_vod_annual_{yr}.tif') for yr in years]
    
    # 检查是否存在足够的文件
    existing = [f for f in tif_files if os.path.exists(f)]
    if len(existing) < 3:
        raise FileNotFoundError(
            f"在 {dir_path} 目录下没有找到足够的年均值 GeoTIFF (格式: evi_vod_annual_YYYY.tif)。"
            f"仅找到 {len(existing)} 个，需要至少 3 个以进行像元趋势计算。"
        )
        
    evi_stack = []
    vod_stack = []
    transform = None
    crs = None
    bounds = None
    
    for f in existing:
        with rasterio.open(f) as src:
            data = src.read()  # (bands, rows, cols)
            
            # 质量转换，处理 nodata
            nodata = src.nodata
            band_evi = data[0].astype(float)
            band_vod = data[1].astype(float)
            
            if nodata is not None:
                band_evi[band_evi == nodata] = np.nan
                band_vod[band_vod == nodata] = np.nan
                
            evi_stack.append(band_evi)
            vod_stack.append(band_vod)
            
            if transform is None:
                transform = src.transform
                crs = src.crs
                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                
    return np.array(evi_stack), np.array(vod_stack), transform, crs, bounds


def generate_mock_datasets(data_dir):
    """
    自动生成高度拟真的黄土高原模拟数据集（CSV和TIF），用于一键式测试与排版演示。
    
    参数:
        data_dir (str): 生成的模拟数据存放目录
    """
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "evi_vod_monthly_ts.csv")
    spatial_path = os.path.join(data_dir, "evi_vod_di_igbp_mean.tif")
    
    print(f"[DataLoader] 正在生成模拟数据集至 {data_dir}...")
    
    # ------------------ 1. 生成 CSV 时间序列 ------------------
    years = range(2010, 2022)
    veg_types = ['Grassland', 'Shrubland', 'Forest', 'Cropland', 'All_Vegetation']
    months = range(1, 13)
    
    rows = []
    np.random.seed(42)
    
    # 模拟物候差异参数
    # peak_month: EVI 峰值月；lag: VOD 相对 EVI 滞后月
    veg_params = {
        'Grassland':  {'base_evi': 0.12, 'amp_evi': 0.10, 'trend_evi': 0.003,  'peak_evi': 7, 'lag': 0, 'base_vod': 0.08, 'amp_vod': 0.04, 'trend_vod': 0.001},
        'Shrubland':  {'base_evi': 0.18, 'amp_evi': 0.15, 'trend_evi': 0.005,  'peak_evi': 7, 'lag': 1, 'base_vod': 0.15, 'amp_vod': 0.08, 'trend_vod': 0.003},
        'Forest':     {'base_evi': 0.28, 'amp_evi': 0.22, 'trend_evi': 0.008,  'peak_evi': 7, 'lag': 2, 'base_vod': 0.25, 'amp_vod': 0.12, 'trend_vod': 0.006},
        'Cropland':   {'base_evi': 0.15, 'amp_evi': 0.18, 'trend_evi': 0.001,  'peak_evi': 8, 'lag': 0, 'base_vod': 0.10, 'amp_vod': 0.06, 'trend_vod': 0.001},
        'All_Vegetation': {'base_evi': 0.16, 'amp_evi': 0.12, 'trend_evi': 0.004, 'peak_evi': 7, 'lag': 1, 'base_vod': 0.12, 'amp_vod': 0.06, 'trend_vod': 0.002}
    }
    
    for yr in years:
        for m in months:
            # 引入 2015 年夏旱（5-8月 z-score 变负）
            drought_factor = 1.0
            if yr == 2015 and m in [5, 6, 7, 8]:
                drought_factor = 0.5  # 模拟干旱胁迫
                
            for vtype in veg_types:
                p = veg_params[vtype]
                
                # EVI 季节信号 (正弦波拟合)
                t_evi = (m - p['peak_evi']) / 12.0 * 2.0 * np.pi
                evi_season = p['amp_evi'] * (np.cos(t_evi) + 1.0) / 2.0
                evi_val = p['base_evi'] + evi_season + (yr - 2010) * p['trend_evi']
                # 加入随机扰动与干旱影响
                evi_val += np.random.normal(0, 0.01)
                if drought_factor < 1.0:
                    # 模拟干旱：EVI 稍慢响应，下跌较浅
                    evi_val -= (p['amp_evi'] * 0.25 * (1.0 - (m-5)/3)) if m >= 6 else (p['amp_evi'] * 0.1)
                
                # VOD 季节信号 (VOD 峰值滞后)
                m_vod = m - p['lag']
                if m_vod <= 0: m_vod += 12
                t_vod = (m_vod - p['peak_evi']) / 12.0 * 2.0 * np.pi
                vod_season = p['amp_vod'] * (np.cos(t_vod) + 1.0) / 2.0
                vod_val = p['base_vod'] + vod_season + (yr - 2010) * p['trend_vod']
                vod_val += np.random.normal(0, 0.008)
                if drought_factor < 1.0:
                    # 模拟干旱：VOD 极敏感，快速下跌且深
                    vod_val -= (p['amp_vod'] * 0.45)
                    
                evi_val = max(0.01, evi_val)
                vod_val = max(0.01, vod_val)
                
                rows.append({
                    'year': yr,
                    'month': m,
                    'veg_type': vtype,
                    'EVI': round(evi_val, 4),
                    'VOD': round(vod_val, 4)
                })
                
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  已生成模拟 CSV 时间序列: {csv_path}")
    
    # ------------------ 2. 生成空间 GeoTIFF 数据 ------------------
    # 模拟黄土高原的大致地理范围: 经度 103°E - 114°E，纬度 34°N - 41°N
    # 像元大小设定为 60x40 像素
    n_rows, n_cols = 40, 60
    
    # 创建空间渐变梯度 (南绿北黄，模拟降水线)
    y_grad, x_grad = np.meshgrid(np.linspace(1, 0, n_rows), np.linspace(0, 1, n_cols), indexing='ij')
    
    # 多年平均 EVI (从南向北递减)
    evi_mean = 0.08 + 0.35 * y_grad + np.random.normal(0, 0.02, (n_rows, n_cols))
    # 多年平均 VOD (与EVI类似，但包含局部不同步区)
    vod_mean = 0.05 + 0.28 * y_grad + np.random.normal(0, 0.015, (n_rows, n_cols))
    # 在黄土高原中北部人为制造 EVI 极高（大量种树）但 VOD 较低（树未长成）的“DI负背离区”
    vod_mean[10:20, 20:45] *= 0.65
    
    # 归一化并计算 DI = VOD_norm - EVI_norm
    evi_norm = (evi_mean - np.nanmin(evi_mean)) / (np.nanmax(evi_mean) - np.nanmin(evi_mean))
    vod_norm = (vod_mean - np.nanmin(vod_mean)) / (np.nanmax(vod_mean) - np.nanmin(vod_mean))
    di = vod_norm - evi_norm
    
    # 模拟 IGBP 土地覆盖 (南部林地5，中部灌木7，北部草地10，关中盆地农田12)
    igbp = np.full((n_rows, n_cols), 10, dtype=np.uint8)  # 默认草地
    igbp[y_grad > 0.75] = 12                              # 南部边缘农田
    igbp[(y_grad > 0.55) & (y_grad <= 0.75)] = 5         # 中南部混交林
    igbp[(y_grad > 0.35) & (y_grad <= 0.55)] = 7         # 中部灌木
    igbp[y_grad < 0.15] = 16                              # 最北端荒漠化裸地
    
    # 添加一个外层掩膜（模拟不规则的研究区边界）
    mask = (x_grad - 0.5)**2 + (y_grad - 0.5)**2 > 0.23
    for grid in [evi_mean, vod_mean, di]:
        grid[mask] = np.nan
    igbp[mask] = 0
    
    if RASTERIO_AVAILABLE:
        # 定义 Affine 放射变换参数 (黄土高原 103E-114E, 34N-41N)
        # 经度跨度 11度，纬度跨度 7度
        from rasterio.transform import from_bounds
        transform = from_bounds(103.0, 34.0, 114.0, 41.0, n_cols, n_rows)
        
        # 写入 Fig.1 空间多波段 TIFF
        with rasterio.open(
            spatial_path, 'w',
            driver='GTiff',
            height=n_rows, width=n_cols,
            count=4,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            nodata=-999.0
        ) as dst:
            dst.write(evi_mean.astype(np.float32), 1)
            dst.write(vod_mean.astype(np.float32), 2)
            dst.write(di.astype(np.float32), 3)
            dst.write(igbp.astype(np.float32), 4)
            
        print(f"  已生成模拟多年平均空间 GeoTIFF: {spatial_path}")
        
        # ------------------ 3. 生成逐年年均值 GeoTIFF (用于 Sen+MK) ------------------
        for yr in years:
            yr_path = os.path.join(data_dir, f"evi_vod_annual_{yr}.tif")
            
            # 引入生态绿化工程的长期增长趋势 (EVI 增长快，VOD 增长慢)
            yr_factor_evi = 1.0 + (yr - 2010) * 0.025
            yr_factor_vod = 1.0 + (yr - 2010) * 0.012
            
            # 在 2015 年加入干旱波动
            if yr == 2015:
                yr_factor_evi *= 0.90
                yr_factor_vod *= 0.82
                
            yr_evi = evi_mean * yr_factor_evi + np.random.normal(0, 0.01, (n_rows, n_cols))
            yr_vod = vod_mean * yr_factor_vod + np.random.normal(0, 0.005, (n_rows, n_cols))
            
            # 应用相同的掩膜
            yr_evi[mask] = np.nan
            yr_vod[mask] = np.nan
            
            with rasterio.open(
                yr_path, 'w',
                driver='GTiff',
                height=n_rows, width=n_cols,
                count=2,
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform,
                nodata=-999.0
            ) as dst:
                dst.write(yr_evi.astype(np.float32), 1)
                dst.write(yr_vod.astype(np.float32), 2)
                
        print(f"  已成功生成 2010-2021 共 12 年的年度均值 GeoTIFF 空间序列")
    else:
        print("  [WARNING] 未检测到 rasterio 库，跳过空间 GeoTIFF 的生成与导出。")
        
    print("[DataLoader] 模拟数据生成完毕！")
