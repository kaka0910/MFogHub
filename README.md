# MFogHub

The MFogHub dataset and benchmarks for "MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting".

![Fig1](https://github.com/kaka0910/MFogHub/blob/main/Figs/teaser.png)

Fig1: Overview of MFogHub. Right: MFogHub collects data from 15 marine fog-prone regions worldwide, captured by 6 geostationary satellites. Middle: Data for each region-satellite pair is organized in a cube-stream structure with dimensions of “timestamp-spectral band-latitude-longitude.” MFogHub includes 21 cube-streams in total, each with corresponding masks, supporting both detection and forecasting tasks. Left: MFogHub enables unique evaluations of model generalization across multiple regions and satellite.

## Introduction

We introduce the MFogHub dataset—the first multi-regional, multi-satellite dataset for global marine fog detection and forecasting. MFogHub contains over 68,000 samples, and spans 15 coastal fog-prone regions, consolidating 693 marine fog events. The dataset captures multi-band meteorological data from 6 geostationary satellites. The minimum time interval is 30 minutes, with a spatial resolution of 1 km and a size of 1024 × 1024 pixels. Additionally, more than 11,600 samples are meticulously annotated at the pixel level by meteorological experts. 

## Updates

- 2024.11.19 Several MFogHub sub-datasets for **marine fog forecasting** are available now！！！
  > **MeteoSat -- D.W.+D.C.+D.E.+N.S.+Na.+A.G./EU+AF/All sub-dataset**. It contains xxx samples. [BaiduNetDisk]()
  >
  > **H8/9 -- Y.B. sub-dataset**. It contains xxx samples. [BaiduNetDisk]()
  >
  > **FY4A -- Y.B. sub-dataset**. It contains xxx samples. [BaiduNetDisk]()
  >

- 2024.11.18 Several MFogHub sub-datasets for **marine fog detection** are available now！！！
  > **GOES16 -- B.C.+C.C+G.A. sub-dataset**. It contains xxx samples with labels. [BaiduNetDisk]()
  > 
  > **FY4A -- Y.B. sub-dataset**. It contains xxx samples with labels. [BaiduNetDisk]()
  > 
  > **H8/9 -- Y.B. sub-dataset**. It contains xxx samples with labels. [BaiduNetDisk]()
  > 

## Connection

If you require additional validation data for marine fog monitoring or forecasting tasks, please contact the authors of MFogHub.

