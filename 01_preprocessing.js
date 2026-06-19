/*
 * ============================================================
 *  黄土高原退耕还林 EVI–VOD 时空差异性分析
 *  第一部分：数据导入与预处理
 * ============================================================
 *
 *  数据源：
 *    - MODIS/061/MYD13C2        (EVI, 0.05°, 月)
 *    - VODCA v2 L-band          (VOD, 0.25°, 10日)
 *    - MODIS/061/MCD12C1        (IGBP 土地覆盖, 0.05°, 年)
 *
 *  预处理流程：
 *    1. 研究区掩膜 & 参数设定
 *    2. QA Masking（EVI 按位解析 DetailedQA；VOD 按 processing_flag 筛选）
 *    3. VOD 10日 → 月聚合
 *    4. EVI 空间对齐（reduceResolution + mean → 0.25°）
 *    5. IGBP 空间对齐（reduceResolution + mode → 0.25°）
 *    6. 可视化验证 & 导出
 */

// ============================================================
// 0. 研究区 & 全局参数
// ============================================================

// ---- 黄土高原矢量掩膜（占位符，替换为实际路径） ----
// 可选方案：
//   A. 自行上传 shapefile 至 GEE Assets
//   B. 使用 FAO GAUL / 中国行政区划数据裁剪
//   C. 手动绘制 geometry
var loessPlateau = ee.FeatureCollection(
  'projects/sinuous-aviary-469104-t4/assets/loess_plateau'
);
var studyArea = loessPlateau.geometry();

// 时间范围
var startDate = '2010-01-01';
var endDate   = '2021-12-31';
var startYear = 2010;
var endYear   = 2021;

// 目标投影参数（与 VOD 0.25° 网格对齐）
var targetCRS   = 'EPSG:4326';
var targetScale = 27830;  // 约 0.25° ≈ 27830 m（在赤道处）

// ============================================================
// 1. 数据导入
// ============================================================

// 1.1  MODIS EVI（16天，0.05°，后续按月求平均）
var modisEVI = ee.ImageCollection('MODIS/061/MYD13C1')
  .filterDate(startDate, endDate)
  .select(['EVI', 'DetailedQA']);

// 1.2  VODCA v2 L-band（10日，0.25°）
//      波段：VOD, sensor_flag, processing_flag
var vodcaL = ee.ImageCollection('projects/sat-io/open-datasets/VODCA/L_BAND_V2')
  .filterDate(startDate, endDate);

// 1.3  MCD12C1 IGBP 土地覆盖（年度，0.05°）
//      用于饱和效应分析中的分类赋色
var igbpCollection = ee.ImageCollection('MODIS/061/MCD12C1')
  .filterDate(startDate, endDate)
  .select('Majority_Land_Cover_Type_1');

print('EVI collection size:',  modisEVI.size());
print('VOD collection size:',  vodcaL.size());
print('IGBP collection size:', igbpCollection.size());

// ============================================================
// 2. QA Masking
// ============================================================

// -------------------------------------------------------
// 2.1  EVI QA Masking
//      DetailedQA 波段按位解析（16-bit unsigned integer）
//      Bit 0-1 : MODLAND_QA (00 = good quality)
//      Bit 6-7 : Aerosol quantity (00=climatology, 01=low, 10=avg, 11=high)
//      Bit 8   : Adjacent cloud detected (0 = no)
//      Bit 10  : Mixed clouds (0 = no)
//      Bit 14  : Possible snow/ice (0 = no)
//      Bit 15  : Possible shadow (0 = no)
// -------------------------------------------------------
function bitwiseExtract(value, fromBit, toBit) {
  // 从 value 中提取 [fromBit, toBit] 范围的位（0-indexed）
  if (toBit === undefined) toBit = fromBit;
  var maskSize = ee.Number(1).add(toBit).subtract(fromBit);
  var mask = ee.Number(1).leftShift(maskSize).subtract(1);
  return value.rightShift(fromBit).bitwiseAnd(mask);
}

function maskEVI(image) {
  var qa = image.select('DetailedQA');

  // Bits 0-1: MODLAND_QA —— 仅保留 00（good）和 01（marginal but produced）
  var viQuality = bitwiseExtract(qa, 0, 1).lte(1);

  // Bit 8: Adjacent cloud —— 0 = no cloud
  var noAdjacentCloud = bitwiseExtract(qa, 8).eq(0);

  // Bit 10: Mixed clouds —— 0 = no
  var noMixedClouds = bitwiseExtract(qa, 10).eq(0);

  // Bit 14: Possible snow/ice —— 0 = no
  var noSnowIce = bitwiseExtract(qa, 14).eq(0);

  // Bit 15: Possible shadow —— 0 = no
  var noShadow = bitwiseExtract(qa, 15).eq(0);

  var mask = viQuality
    .and(noAdjacentCloud)
    .and(noMixedClouds)
    .and(noSnowIce)
    .and(noShadow);

  // EVI scale factor = 0.0001, 有效范围 [-0.2, 1.0]
  return image.select('EVI')
    .updateMask(mask)
    .clamp(-0.2, 1.0)
    .rename('EVI')
    .copyProperties(image, ['system:time_start']);
}

// -------------------------------------------------------
// 2.2  VOD QA Masking
//      processing_flag == 0 表示无异常
//      同时限制 VOD 在物理合理范围 (0, 3) 内
//      VODCA v2 已在生产阶段剔除了冻土/雪/RFI，
//      此处做二次质量把控
// -------------------------------------------------------
function maskVOD(image) {
  var vod  = image.select('VOD');

  // 注意：当前 GEE Community Catalog 里的 VODCA V2 数据集只保留了 VOD 单波段，没有 QA flag 波段。
  // 数据本身已经被预处理过，所以我们仅使用物理有效范围 (0, 3) 进行异常值过滤。
  
  // VOD 物理有效范围 (0, 3)
  var validRange = vod.gt(0).and(vod.lt(3));

  return vod
    .updateMask(validRange)
    .rename('VOD')
    .copyProperties(image, ['system:time_start']);
}

// 应用 QA Masking
var eviMasked = modisEVI.map(maskEVI).filterBounds(studyArea);
var vodMasked = vodcaL.map(maskVOD).filterBounds(studyArea);

print('EVI after QA masking:', eviMasked.first());
print('VOD after QA masking:', vodMasked.first());

// ============================================================
// 3. VOD 10日 → 月均值聚合
// ============================================================

// 构建 year-month 列表
var yearMonthList = [];
for (var y = startYear; y <= endYear; y++) {
  for (var m = 1; m <= 12; m++) {
    yearMonthList.push({year: y, month: m});
  }
}

// 对每个月聚合 VOD
var vodMonthly = ee.ImageCollection(
  yearMonthList.map(function(ym) {
    var start = ee.Date.fromYMD(ym.year, ym.month, 1);
    var end   = start.advance(1, 'month');
    var monthImages = vodMasked.filterDate(start, end);

    return monthImages.mean()
      .rename('VOD')
      .set('year', ym.year)
      .set('month', ym.month)
      .set('system:time_start', start.millis())
      .set('system:index', ym.year + '_' + ('0' + ym.month).slice(-2));
  })
);

// 过滤掉全空图像（某些月份可能完全无有效观测）
vodMonthly = vodMonthly.filter(ee.Filter.listContains(
  'system:band_names', 'VOD'));

print('VOD monthly collection size:', vodMonthly.size());

// ============================================================
// 4. EVI 空间对齐（0.05° → 0.25°）
// ============================================================
// 使用 reduceResolution + mean 做空间聚合（5×5 像元取均值）
// 比双线性插值更适合上采样（upscaling）场景，
// 保留了所有精细像元的信息而非仅中心点插值

// 获取 VOD 的原生投影作为目标投影
var vodProjection = ee.Image(vodMonthly.first()).projection();
print('VOD native projection:', vodProjection);

function alignEVItoVOD(eviImage) {
  return eviImage
    .setDefaultProjection(targetCRS, null, 5566)  // EVI 原生 ~0.05° ≈ 5566m
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024,
      bestEffort: true
    })
    .reproject({
      crs: vodProjection
    })
    .rename('EVI')
    .copyProperties(eviImage, ['system:time_start']);
}

var eviAligned = eviMasked.map(alignEVItoVOD);

print('EVI aligned sample:', eviAligned.first());

// ============================================================
// 5. IGBP 空间对齐（0.05° → 0.25°，众数重采样）
// ============================================================
// 分类数据为离散整数（1-17），严禁使用双线性插值！
// 使用 mode reducer：每个 0.25° 网格取 5×5 像元中出现频率最高的类别

// 取研究期间的众数 IGBP 分类（多年稳定分类）
var igbpImage = igbpCollection.mode()  // 多年取众数
  .rename('IGBP');

// 保持 IGBP 在原生 0.05° 分辨率，不做粗化
// 破碎的灌木斑块在 0.25° 众数聚合中会全部消失
var igbpAligned = igbpImage;

print('IGBP aligned:', igbpAligned);

// IGBP 类别查找表（用于图例和后续分析）
var igbpNames = {
  1:  'Evergreen Needleleaf',
  2:  'Evergreen Broadleaf',
  3:  'Deciduous Needleleaf',
  4:  'Deciduous Broadleaf',
  5:  'Mixed Forest',
  6:  'Closed Shrublands',
  7:  'Open Shrublands',
  8:  'Woody Savannas',
  9:  'Savannas',
  10: 'Grasslands',
  11: 'Permanent Wetlands',
  12: 'Croplands',
  13: 'Urban',
  14: 'Cropland/Natural Mosaic',
  15: 'Snow and Ice',
  16: 'Barren',
  17: 'Water Bodies'
};

// ============================================================
// 6. 构建对齐后的月度 EVI-VOD 联合数据集
// ============================================================
// 按 year-month 进行 inner join，确保每个月同时有 EVI 和 VOD

function joinEVIandVOD(ym) {
  var start = ee.Date.fromYMD(ym.year, ym.month, 1);
  var end   = start.advance(1, 'month');

  var evi = eviAligned.filterDate(start, end).mean();
  var vod = vodMonthly.filterDate(start, end).mean();

  return evi.addBands(vod)
    .clip(studyArea)
    .set('year', ym.year)
    .set('month', ym.month)
    .set('system:time_start', start.millis());
}

var eviVodMonthly = ee.ImageCollection(
  yearMonthList.map(function(ym) {
    return joinEVIandVOD(ym);
  })
);

// 过滤掉缺失 EVI 或 VOD 的无效月份
eviVodMonthly = eviVodMonthly
  .filter(ee.Filter.listContains('system:band_names', 'EVI'))
  .filter(ee.Filter.listContains('system:band_names', 'VOD'));

print('Aligned EVI-VOD monthly collection:', eviVodMonthly.size());
print('Sample image bands:', eviVodMonthly.first().bandNames());

// ============================================================
// 7. 可视化验证
// ============================================================

// 地图中心定位到研究区
Map.centerObject(studyArea, 6);

// 研究区边界
Map.addLayer(studyArea, {color: 'white'}, 'Study Area Boundary', true, 0.5);

// 多年平均 EVI 空间图
var eviMeanMap = eviVodMonthly.select('EVI').mean().clip(studyArea);
Map.addLayer(eviMeanMap, {
  min: 0.05, max: 0.55,
  palette: ['#f7fcb9', '#addd8e', '#41ab5d', '#006837']
}, 'Multi-year Mean EVI');

// 多年平均 VOD 空间图
var vodMeanMap = eviVodMonthly.select('VOD').mean().clip(studyArea);
Map.addLayer(vodMeanMap, {
  min: 0.05, max: 0.55,
  palette: ['#fff7bc', '#fec44f', '#d95f0e', '#7f2704']
}, 'Multi-year Mean VOD');

// IGBP 土地覆盖分类图
Map.addLayer(igbpAligned.clip(studyArea), {
  min: 1, max: 17,
  palette: [
    '#05450a', '#086a10', '#54a708', '#78d203', '#009900',  // 1-5 森林
    '#c6b044', '#dcd159',                                    // 6-7 灌木
    '#dade48', '#fbff13',                                    // 8-9 稀树草原
    '#b6ff05',                                                // 10 草地
    '#27ff87',                                                // 11 湿地
    '#c24f44',                                                // 12 农田
    '#a5a5a5',                                                // 13 城市
    '#ff6d4c',                                                // 14 农/自然镶嵌
    '#69fff8',                                                // 15 冰雪
    '#f9ffa4',                                                // 16 裸地
    '#1c0dff'                                                 // 17 水体
  ]
}, 'IGBP Land Cover');

// ============================================================
// 8. 解耦指数 DI 空间图（全文独创）
// ============================================================
// DI = VOD_norm - EVI_norm
// 仅具区域内相对比较意义，不可跨区域对比

// 计算研究区内 EVI 和 VOD 的 min/max
var eviStats = eviMeanMap.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: targetScale,
  maxPixels: 1e9
});
var vodStats = vodMeanMap.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: studyArea,
  scale: targetScale,
  maxPixels: 1e9
});

var eviMin = ee.Number(eviStats.get('EVI_min'));
var eviMax = ee.Number(eviStats.get('EVI_max'));
var vodMin = ee.Number(vodStats.get('VOD_min'));
var vodMax = ee.Number(vodStats.get('VOD_max'));

print('EVI range:', eviMin, '–', eviMax);
print('VOD range:', vodMin, '–', vodMax);

// Min-Max 归一化
var eviNorm = eviMeanMap.subtract(eviMin).divide(eviMax.subtract(eviMin));
var vodNorm = vodMeanMap.subtract(vodMin).divide(vodMax.subtract(vodMin));

// 解耦指数
var DI = vodNorm.subtract(eviNorm).rename('DI');

Map.addLayer(DI.clip(studyArea), {
  min: -0.5, max: 0.5,
  palette: [
    '#d73027', '#fc8d59', '#fee08b',  // DI<0: EVI偏高（红-橙-黄）
    '#d9ef8b', '#91cf60', '#1a9850'   // DI>0: VOD偏高（浅绿-深绿）
  ]
}, 'Decoupling Index (DI)');

// ============================================================
// 9. 导出数据（供 Python 端后续分析）
// ============================================================

// ----- 9.1 导出区域平均月度时间序列（CSV） -----
// 按植被类型分组提取时间序列，用于季节动态、lag analysis 等

// 定义感兴趣的植被类型掩膜
var grassMask  = igbpAligned.eq(10);                              // 草地
var shrubMask  = igbpAligned.eq(6).or(igbpAligned.eq(7));         // 灌木（含开放+封闭）
var forestMask = igbpAligned.gte(1).and(igbpAligned.lte(5));      // 森林（5种）
var cropMask   = igbpAligned.eq(12).or(igbpAligned.eq(14));       // 农田

// 提取每种植被类型的区域平均时间序列
function extractTimeSeries(collection, mask, label) {
  return collection.map(function(image) {
    var maskedImage = image.updateMask(mask);
    var stats = maskedImage.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: studyArea,
      scale: 5566,  // 修复：使用 IGBP 原生 0.05° 分辨率，以保留细碎的灌木斑块
      maxPixels: 1e13 // 放宽像素限制
    });
    return ee.Feature(null, stats)
      .set('year',  image.get('year'))
      .set('month', image.get('month'))
      .set('system:time_start', image.get('system:time_start'))
      .set('veg_type', label);
  });
}

var tsGrass  = extractTimeSeries(eviVodMonthly, grassMask,  'Grassland');
var tsShrub  = extractTimeSeries(eviVodMonthly, shrubMask,  'Shrubland');
var tsForest = extractTimeSeries(eviVodMonthly, forestMask, 'Forest');
var tsCrop   = extractTimeSeries(eviVodMonthly, cropMask,   'Cropland');
var tsAll    = extractTimeSeries(eviVodMonthly, 
  igbpAligned.gt(0).and(igbpAligned.lt(13)), 'All_Vegetation');

// 合并所有时间序列
var allTimeSeries = tsGrass.merge(tsShrub).merge(tsForest)
  .merge(tsCrop).merge(tsAll);

// 导出为 CSV
Export.table.toDrive({
  collection: allTimeSeries,
  description: 'EVI_VOD_monthly_timeseries_by_vegtype',
  folder: 'LoessPlateau_Microwave_Data',
  fileNamePrefix: 'evi_vod_monthly_ts',
  fileFormat: 'CSV',
  selectors: ['year', 'month', 'veg_type', 'EVI', 'VOD']
});

// ----- 9.2 导出逐像元月度数据（GeoTIFF） -----
// 导出多年均值 EVI、VOD、DI 空间图（用于 Python 端精美制图）

Export.image.toDrive({
  image: eviMeanMap.addBands(vodMeanMap).addBands(DI)
    .addBands(igbpAligned).clip(studyArea).toFloat(),
  description: 'EVI_VOD_DI_IGBP_multiyear_mean',
  folder: 'LoessPlateau_Microwave_Data',
  fileNamePrefix: 'evi_vod_di_igbp_mean',
  region: studyArea,
  scale: targetScale,
  crs: targetCRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

// ----- 9.3 导出逐年均值（用于 Sen slope 趋势分析） -----
for (var yr = startYear; yr <= endYear; yr++) {
  var yearStart = ee.Date.fromYMD(yr, 1, 1);
  var yearEnd   = ee.Date.fromYMD(yr, 12, 31);

  var annualEVI = eviVodMonthly.filterDate(yearStart, yearEnd)
    .select('EVI').mean().rename('EVI');
  var annualVOD = eviVodMonthly.filterDate(yearStart, yearEnd)
    .select('VOD').mean().rename('VOD');

  Export.image.toDrive({
    image: annualEVI.addBands(annualVOD).clip(studyArea).toFloat(),
    description: 'EVI_VOD_annual_' + yr,
    folder: 'LoessPlateau_Microwave_Data',
    fileNamePrefix: 'evi_vod_annual_' + yr,
    region: studyArea,
    scale: targetScale,
    crs: targetCRS,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
}

// ----- 9.4 导出逐像元完整月度序列（用于像元级 lag analysis） -----
// 注意：这是最大的导出，仅在需要像元级分析时启用
/*
Export.image.toDrive({
  image: eviVodMonthly.select('EVI').toBands()
    .clip(studyArea).toFloat(),
  description: 'EVI_monthly_allbands',
  folder: 'LoessPlateau_Microwave_Data',
  fileNamePrefix: 'evi_monthly_allbands',
  region: studyArea,
  scale: targetScale,
  crs: targetCRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: eviVodMonthly.select('VOD').toBands()
    .clip(studyArea).toFloat(),
  description: 'VOD_monthly_allbands',
  folder: 'LoessPlateau_Microwave_Data',
  fileNamePrefix: 'vod_monthly_allbands',
  region: studyArea,
  scale: targetScale,
  crs: targetCRS,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF'
});
*/

// ============================================================
// 10. 验证打印：检查对齐是否正确
// ============================================================

// 打印一张对齐后图像的空间信息
var sampleImage = ee.Image(eviVodMonthly.first());
print('Sample aligned image projection:', sampleImage.select('EVI').projection());
print('Sample aligned image scale (m):', sampleImage.select('EVI').projection().nominalScale());

// 统计信息快速检查
var sampleStats = sampleImage.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true),
  geometry: studyArea,
  scale: targetScale,
  maxPixels: 1e9
});
print('Sample image stats (EVI & VOD mean/std):', sampleStats);

print('========================================');
print('预处理完成！');
print('下一步：在 Python 中加载导出的 CSV/GeoTIFF');
print('进行 lag analysis, DLM, 趋势检验等分析');
print('========================================');
