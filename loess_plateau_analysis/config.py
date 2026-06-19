# -*- coding: utf-8 -*-
"""
============================================================
 黄土高原退耕还林 EVI–VOD 时空差异性分析
 配置文件 (config.py)
============================================================
"""
import os

# 基础目录结构定义
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)

# 默认数据输入输出路径
DEFAULT_DATA_DIR = "/mnt/d/文件夹/26春/微波遥感/data/gee_downloads/drive-download"
CSV_PATH = os.path.join(DEFAULT_DATA_DIR, "evi_vod_monthly_ts.csv")
SPATIAL_TIF = os.path.join(DEFAULT_DATA_DIR, "evi_vod_di_igbp_mean.tif")
ANNUAL_TIF_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = "/mnt/d/文件夹/26春/微波遥感/data/outputs"

# 绘图全局样式设定
PLOT_STYLE = {
    "font.family": ["serif"],
    "font.sans-serif": ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"],
    "font.serif": ["Noto Serif CJK JP", "Times New Roman", "SimSun", "DejaVu Serif", "Liberation Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,  # 确保负号和特殊符号正常显示
}

# 植被类型配色映射 (Fig.2, Fig.3, Fig.7, Fig.8 等多系列图件中使用)
VEG_COLORS = {
    "Grassland": "#66c2a5",
    "Shrubland": "#fc8d62",
    "Forest": "#8da0cb",
    "Cropland": "#e78ac3",
    "All_Vegetation": "#a6d854",
}

# 植被中文标签映射
VEG_LABELS_ZH = {
    "Grassland": u"草地",
    "Shrubland": u"灌木",
    "Forest": u"森林",
    "Cropland": u"农田",
    "All_Vegetation": u"全部植被",
}

# 月份英文字标
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# IGBP 土地覆盖类别映射描述 (MCD12C1 17分类体系)
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

# IGBP 对齐后的 GEE 色卡配色映射 (Fig.1d 专用)
IGBP_COLORS = {
    1: '#05450a',  # 针叶常绿林
    2: '#086a10',  # 阔叶常绿林
    3: '#54a708',  # 针叶落叶林
    4: '#78d203',  # 阔叶落叶林
    5: '#009900',  # 混交林
    6: '#c6b044',  # 郁闭灌木林
    7: '#dcd159',  # 开阔灌木林
    8: '#dade48',  # 稀树郁闭灌木林
    9: '#fbff13',  # 稀树草地
    10: '#b6ff05', # 草地
    11: '#27ff87', # 永久湿地
    12: '#c24f44', # 农田
    13: '#a5a5a5', # 城镇与建成区
    14: '#ff6d4c', # 农田/自然植被斑块
    15: '#69fff8', # 永久冰雪
    16: '#f9ffa4', # 裸地
    17: '#1c0dff'  # 水体
}
