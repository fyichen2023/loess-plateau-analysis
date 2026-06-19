# Loess Plateau EVI-VOD Spatiotemporal Coupling Analysis

## Overview
This repository contains the complete analytical pipeline for investigating the spatiotemporal coupling and decoupling characteristics between vegetation greenness (EVI) and structural biomass (VODCA L-band VOD) across the Loess Plateau. 

By analyzing the long-term ecological responses to the "Grain for Green" program, this project evaluates optical saturation effects, climate response lag (ecological memory), and the divergent trajectories between canopy greening and woody biomass restoration.

## Project Structure
The workflow is divided into two main components: Google Earth Engine (GEE) preprocessing and local Python statistical analysis.

### 1. GEE Preprocessing (`01_preprocessing.js`)
A JavaScript tool designed for the Google Earth Engine Code Editor. It handles massive spatial data alignment and extraction:
- **Data Sources**: MODIS MYD13C2 (EVI, 0.05°), VODCA v2 L-band (VOD, 0.25°), and MCD12C1 (IGBP Land Cover).
- **QA Masking**: Bitwise parsing for MODIS DetailedQA and processing flags for VOD.
- **Spatial Alignment**: Resamples EVI and IGBP data using `reduceResolution` to strictly match the 0.25° VODCA grid.
- **Temporal Aggregation**: Aggregates 10-day VOD to monthly composites to match MODIS frequency.

### 2. Python Analytical Pipeline (`main.py`)
The local Python engine takes the exported CSV/TIFF from GEE and performs in-depth ecological and statistical modeling:
- **Spatial Trend Analysis**: Pixel-level Sen's Slope and Mann-Kendall (MK) significance tests for long-term trends, generating publication-ready geographic maps.
- **Time-Series & Seasonal Dynamics**: Automated seasonal climatology profiling across different IGBP vegetation types.
- **Cross-Correlation Analysis**: Quantitative assessment of phase delays (lag) between optical greening and microwave biomass accumulation.
- **Almon Distributed Lag Models (DLM)**: Evaluates ecological memory and the cumulative impact of climate anomalies with Almon polynomial constraints.
- **Optical Saturation Assessment**: Bivariate density scatter plots mapping EVI saturation asymptotes in high-biomass regions.

## File Organization
- `01_preprocessing.js`: GEE preprocessing and export script.
- `main.py`: The main entry point for the local analytical pipeline.
- `loess_plateau_analysis/`
  - `config.py`: Global configurations, plot styles, IGBP class mappings, and I/O paths.
  - `data_loader.py`: Handles CSV/GeoTIFF parsing and automated mock dataset generation for testing.
  - `stats.py`: Core mathematical and statistical algorithms.
  - `plotting.py`: The `matplotlib`-based visualization engine tailored for high-quality, multi-panel figures.

## Requirements
Ensure you have Python 3.8+ installed. The following core dependencies are required:
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn rasterio pyproj
```

## Usage
Run the complete pipeline to process the data and generate all 8 figures (`Fig.1` to `Fig.8`) in the designated output directory:
```bash
python main.py
```

For academic publications where figures require external captions instead of embedded titles, use the `--no-title` flag to suppress main/sub-titles while preserving sub-panel lettering (a, b, c):
```bash
python main.py --no-title
```
