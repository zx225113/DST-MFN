# High-Order Motion Statistics Guided Weakly Supervised Group Activity Recognition

### Xiao Ling Zhu, Xiao Zhang, TaiGuo Deng, Di Wu, YaoNan Wang, Qing Wan 


## Paper Overview
This work introduces a Dual-branch Spatiotemporal Motion Fusion Network (DST-MFN) for weakly supervised group activity recognition (GAR). The core innovation lies in leveraging high-order motion statistics (Acceleration, Jerk, and Snap) through a Kinematic Semantic Decoupling Module (KSDM) to suppress smooth camera motion noise and filter out low-frequency interference. The framework further utilizes a High-Order Motion Statistical Transformer (HoMST) to reduce attention complexity to linear and a Spatiotemporal Fusion Module (STFM) to align local appearance features with global motion cues.

## Citation
If you find our code or paper useful, please consider citing our paper:
        Application unsuccessful.


## Requirements

- Ubuntu 16.04
- Python 3.8.5
- CUDA 11.0
- PyTorch 1.7.1

## Conda environment installation
    conda env create --file environment.yml

    conda activate zx
    
    Environments commented out with # in the environment.yml file need to be manually installed as required in the requirements.

## Install additional package
    sh scripts/setup.sh
    
## Download dataset
- Volleyball dataset <br/>
Download Volleyball dataset from:   <br/> 
https://drive.google.com/file/d/1DaUE3ODT_H5mBFi8JzOVBNzVldxfbPbX/view?usp=sharing      
Dataset should be located following the file structure described below. <br/>

- NBA dataset <br/>
The dataset is available upon request to the authors of 
  "Social Adaptive Module for Weakly-supervised Group Activity Recognition (ECCV 2020)". 

## Run test scripts

- Volleyball dataset (Merged 6 class classification)  

        sh scripts/test_volleyball_merged.sh

- Volleyball dataset (Original 8 class classification)   

        sh scripts/test_volleyball.sh

- NBA dataset  

        sh scripts/test_nba.sh


## Run train scripts

- Volleyball dataset (Merged 6 class classification)
    
        sh scripts/train_volleyball_merged.sh

- Volleyball dataset (Original 8 class classification)
    
        sh scripts/train_volleyball.sh

- NBA dataset
    
        sh scripts/train_nba.sh


## Acknowledgement
This work was supported by the National Natural Science Foundation of China (NSFC) under Grant 62031023.

