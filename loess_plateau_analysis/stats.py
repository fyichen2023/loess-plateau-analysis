# -*- coding: utf-8 -*-
"""
============================================================
 黄土高原退耕还林 EVI–VOD 时空差异性分析
 统计计算模块 (stats.py)
============================================================
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import pymannkendall as mk
    PYMANNKENDALL_AVAILABLE = True
except ImportError:
    PYMANNKENDALL_AVAILABLE = False


class FallbackTrendResult:
    """
    当未安装 pymannkendall 库时，使用 scipy.stats.linregress 进行线性拟合
    并模拟 pymannkendall 的输出结构，防止调用程序报错。
    """
    def __init__(self, slope, intercept, p_value):
        self.slope = slope
        self.intercept = intercept
        self.p = p_value
        if p_value < 0.05:
            self.trend = 'increasing' if slope > 0 else 'decreasing'
        else:
            self.trend = 'no trend'


def compute_seasonal_climatology(df, veg_type):
    """
    计算特定植被类型的 EVI 和 VOD 的 12 个月的多年平均值和标准差。
    """
    sub = df[df['veg_type'] == veg_type]
    monthly = sub.groupby('month').agg(
        evi_mean=('EVI', 'mean'), evi_std=('EVI', 'std'),
        vod_mean=('VOD', 'mean'), vod_std=('VOD', 'std')
    ).reset_index()
    return monthly


def compute_cross_correlation(df, veg_type, max_lag=6):
    """
    计算 EVI 与 VOD 的 Pearson 滞后相关关系。
    正 lag 表示 VOD 滞后于 EVI（EVI 领先）；负 lag 表示 VOD 领先 EVI。
    """
    sub = df[df['veg_type'] == veg_type].sort_values('date')
    evi = sub['EVI'].values
    vod = sub['VOD'].values
    n = len(evi)

    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    pvals = []

    for lag in lags:
        if lag >= 0:
            e = evi[:n - lag] if lag > 0 else evi
            v = vod[lag:] if lag > 0 else vod
        else:
            e = evi[-lag:]
            v = vod[:n + lag]

        valid_mask = np.isfinite(e) & np.isfinite(v)
        e_valid = e[valid_mask]
        v_valid = v[valid_mask]

        if len(e_valid) > 1:
            r, p = scipy_stats.pearsonr(e_valid, v_valid)
        else:
            r, p = np.nan, np.nan

        corrs.append(r)
        pvals.append(p)

    # Use nanargmax to ignore NaNs when finding the best lag
    best_idx = np.nanargmax(corrs) if not np.all(np.isnan(corrs)) else 0
    
    return {
        'lags': lags,
        'corrs': corrs,
        'pvals': pvals,
        'best_lag': lags[best_idx],
        'best_r': corrs[best_idx],
        'best_p': pvals[best_idx]
    }


def run_trend_test(years, values):
    """
    计算一维序列的趋势斜率和显著性，优先使用 Mann-Kendall，若无则使用 OLS。
    """
    if PYMANNKENDALL_AVAILABLE:
        try:
            return mk.original_test(values)
        except Exception:
            pass
            
    # Fallback: 使用 SciPy 线性回归拟合
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, values)
    return FallbackTrendResult(slope, intercept, p_value)


def compute_zscore_anomalies(df, veg_type='All_Vegetation'):
    """
    计算标准化月度异常值 (z-score)，并提取双指标同时出现严重负异常 (<-1) 的时段。
    z_t = (x_t - x_bar_month) / sigma_month
    """
    sub = df[df['veg_type'] == veg_type].copy().sort_values('date')

    # 计算每月的气候态均值和标准差
    climatology = sub.groupby('month').agg(
        evi_clim_mean=('EVI', 'mean'), evi_clim_std=('EVI', 'std'),
        vod_clim_mean=('VOD', 'mean'), vod_clim_std=('VOD', 'std'),
    ).reset_index()

    # 合并气候态并算 z-score
    sub = sub.merge(climatology, on='month', how='left')
    sub['EVI_z'] = (sub['EVI'] - sub['evi_clim_mean']) / sub['evi_clim_std']
    sub['VOD_z'] = (sub['VOD'] - sub['vod_clim_mean']) / sub['vod_clim_std']

    # 标记双重负异常 (EVI_z < -1 且 VOD_z < -1)
    sub['both_negative'] = (sub['EVI_z'] < -1) & (sub['VOD_z'] < -1)
    sub['anomaly_score'] = np.where(sub['both_negative'], sub['EVI_z'] + sub['VOD_z'], 0.0)

    return sub


def fit_almon_dlm(df, veg_type, K=6, poly_degree=2):
    """
    Almon 多项式分布滞后模型 (PDL) 拟合。
    VOD_t = a + sum_{k=0}^K beta_k * EVI_{t-k} + epsilon
    约束: beta_k = c_0 + c_1 * k + c_2 * k^2
    """
    sub = df[df['veg_type'] == veg_type].sort_values('date')
    evi = sub['EVI'].values
    vod = sub['VOD'].values
    n = len(evi)

    # 1. 构造滞后矩阵 X: (n-K) x (K+1)
    X_lag = np.column_stack([evi[K - k: n - k] for k in range(K + 1)])
    y = vod[K:]

    # Mask out rows with NaNs
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X_lag), axis=1)
    X_lag = X_lag[valid_mask]
    y = y[valid_mask]

    if len(y) == 0:
        return {'k_vals': np.arange(K + 1, dtype=float), 'beta': np.zeros(K + 1), 'gamma': np.zeros(poly_degree + 2), 'r_squared': 0.0, 'peak_lag': 0}

    # 2. 构造 Almon 变换矩阵 H: (K+1) x (poly_degree+1)
    k_vals = np.arange(K + 1, dtype=float)
    H = np.column_stack([k_vals**d for d in range(poly_degree + 1)])

    # 3. 变换设计矩阵 Z = X @ H
    Z = X_lag @ H

    # 4. 加入截距项
    Z_const = np.column_stack([np.ones(len(Z)), Z])

    # 5. OLS 估计系数
    gamma, residuals, rank, sv = np.linalg.lstsq(Z_const, y, rcond=None)

    # 6. 反推权重 beta = H @ gamma_poly
    gamma_poly = gamma[1:]
    beta = H @ gamma_poly

    # 7. 计算拟合值与 R²
    y_hat = Z_const @ gamma
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'k_vals': k_vals,
        'beta': beta,
        'gamma': gamma,
        'r_squared': r_squared,
        'peak_lag': np.argmax(beta)
    }


def calculate_loop_area(evi, vod):
    """
    使用 Shoelace (鞋带) 几何多边形求积公式，计算 EVI-VOD 季节迟滞回环的面积。
    Area = 0.5 * |sum(X_i * Y_{i+1} - X_{i+1} * Y_i)|
    """
    n = len(evi)
    area = 0.5 * abs(sum(
        evi[i] * vod[(i + 1) % n] - evi[(i + 1) % n] * vod[i]
        for i in range(n)
    ))
    return area


def compute_pixel_trends(evi_stack, vod_stack):
    """
    逐像元计算 EVI 和 VOD 的 Sen's slope 趋势和 Mann-Kendall 显著性 p 值。
    
    参数:
        evi_stack/vod_stack: (years, rows, cols) 的 3D 阵列
        
    返回:
        tuple: (evi_slope, evi_pval, vod_slope, vod_pval)
    """
    n_years, n_rows, n_cols = evi_stack.shape
    
    # 初始化输出网格
    evi_slope = np.full((n_rows, n_cols), np.nan)
    evi_pval  = np.full((n_rows, n_cols), np.nan)
    vod_slope = np.full((n_rows, n_cols), np.nan)
    vod_pval  = np.full((n_rows, n_cols), np.nan)
    
    years = np.arange(2010, 2010 + n_years)

    for i in range(n_rows):
        for j in range(n_cols):
            evi_ts = evi_stack[:, i, j]
            vod_ts = vod_stack[:, i, j]

            # 剔除无效值像元
            if np.any(np.isnan(evi_ts)) or np.any(np.isnan(vod_ts)):
                continue
            if np.all(evi_ts == 0) or np.all(vod_ts == 0):
                continue

            try:
                # EVI 趋势检验
                evi_res = run_trend_test(years, evi_ts)
                evi_slope[i, j] = evi_res.slope
                evi_pval[i, j]  = evi_res.p

                # VOD 趋势检验
                vod_res = run_trend_test(years, vod_ts)
                vod_slope[i, j] = vod_res.slope
                vod_pval[i, j]  = vod_res.p
            except Exception:
                pass
                
    return evi_slope, evi_pval, vod_slope, vod_pval
