# MFogHub

The MFogHub dataset and benchmarks for "MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting".

![Fig1](https://github.com/kaka0910/MFogHub/blob/main/Figs/teaser.png)

Fig1: Overview of MFogHub. Right: MFogHub collects data from 15 marine fog-prone regions worldwide, captured by 6 geostationary satellites. Middle: Data for each region-satellite pair is organized in a cube-stream structure with dimensions of “timestamp-spectral band-latitude-longitude.” MFogHub includes 21 cube-streams in total, each with corresponding masks, supporting both detection and forecasting tasks. Left: MFogHub enables unique evaluations of model generalization across multiple regions and satellite.

## Introduction

We introduce the MFogHub dataset—the first multi-regional, multi-satellite dataset for global marine fog detection and forecasting. MFogHub contains over 68,000 samples, and spans 15 coastal fog-prone regions, consolidating 693 marine fog events. The dataset captures multi-band meteorological data from 6 geostationary satellites. The minimum time interval is 30 minutes, with a spatial resolution of 1 km and a size of 1024 × 1024 pixels. Additionally, more than 11,600 samples are meticulously annotated at the pixel level by meteorological experts. 

## Updates

- 2024.11.19 Several MFogHub sub-datasets for **marine fog forecasting** are available now！！！
  > **MeteoSat -- D.W.+D.C.+D.E.+N.S.+Na.+A.G./EU+AF/All sub-dataset**. It contains 472*6=2784 samples with the shape of TxCxHxW = 8x3x256x256. 
  >
  > **H8/9 -- Y.B. sub-dataset**. It contains 2,512 samples with the shape of TxCxHxW = 8x3x256x256, composed by visible bands (0.47μm, 0.51μm, 0.64μm). 
  >
  > **FY4A -- Y.B. sub-dataset**. It contains 3,931  samples with the shape of TxCxHxW = 8x3x256x256, composed by visible bands (0.47μm, 0.65μm) + Near-Infrared band (0.825μm). 
  >

- 2024.11.18 Several MFogHub sub-datasets for **marine fog detection** are available now！！！
  > **GOES16 -- B.C.+C.C.+G.A. sub-dataset**. It contains 474 (B.C.) + 404 (C.C.) + 408 (G.A.) samples with the shape of CxHxW = 16x1024x1024, from 2020 to 2023. 
  > 
  > **FY4A -- Y.B. sub-dataset**. It contains 1,724 samples with the shape of CxHxW = 14x1024x1024, from 2018 to 2021. 
  > 
  > **H8/9 -- Y.B. sub-dataset**. It contains 1,802 samples with the shape of CxHxW = 16x1024x1024, from 2018 to 2021.
  > 

## How to use

Due to repository size limitations and the requirements for anonymous submission, we provide a smaller version of the MFogHub dataset to showcase our examples shown as Folder `small_version`. The data is organized according to the tasks of marine fog detection or forecasting. Each data sample is named in the format `"SatelliteAbbreviation_RegionAbbreviation_Timestamp.npy"`, and its corresponding label is named `"SatelliteAbbreviation_RegionAbbreviation_Timestamp.png"`.

**Folder structure**

```
|-- ROOT
    |-- Detection # For marine fog detection task
        |-- Multi-regional GOES    # Tasking GOES16 as an example, including several samples of B.C./C.C./G.A. sub-dataset 
            |-- G16_BC_20220441600.npy  # Multi-spectral-bands data
            |-- G16_BC_20220441600.png  # Corresponding label
            |-- G16_GA_20200561400.npy
            |-- G16_GA_20200561400.png
            ....
        |-- Multi-satellite YB    # Tasking Y.B. region as an example, including FY4A and H8/9 satellite sub-dataset 
            |-- FY4A_YB_20210325_0000.npy  # Multi-spectral-bands data
            |-- FY4A_YB_20210325_0000.png  # Corresponding label
            |-- H89_YB_20210613_0000.npy
            |-- H89_YB_20210613_0000.png
            ....
    |-- Forecasting  # For marine fog forecasting task
        |-- Multi-regional MeteoSat   # Tasking MeteoSat as an example, including several samples of single-regional (D.W.+D.C.+D.E.+N.S.+Na.+A.G.) and multi-region (EU+AF+all) sub-dataset 
            |-- MeteoSat_All_2022.npy # Multi-regional sub-dataset
            |-- MeteoSat_AG_2022.npy  # Single-regional sub-dataset
            ....
        |-- Multi-satellite YB    # Tasking Y.B. region as an example, including FY4A and H8/9 satellite sub-dataset 
            |-- H89_YB_2021.npy
            |-- FY4A_YB_2021.npy
```


## How to create new samples

We provide the complete processing workflow from raw satellite radiometer data to images. The corresponding processing code, along with visualization examples of true-color and pseudo-color images, can be found in the `process` folder. The workflow supports generating customized multi-channel images with specific latitude-longitude ranges and spatial resolutions.

## Connection

If you require additional validation data for marine fog monitoring or forecasting tasks, please contact the authors of MFogHub.

