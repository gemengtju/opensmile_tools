# Drunk ASR Project

A drunk asr project is building based on opensmile tools and SVM classification. 


## Folder Structure

```
├── drunk_main_sicheng.sh - here's a main bash file that is responsible for the whole pipeline
│
│
├── data                  - this folder contains original non-processed sicheng speech data
│
│
├── local                 - this folder contains the source code for generating features
│   └── prepare_data.py   - the main code for preparing data
│
│
├── model                 - this folder contains the models, such as SVMs.
│
│
|
├── tools                 - this folder contains the some speech processing tools
│   └── opensmile-2.3.0   - OPENsmile tools
|   └── vadtools          - VAD tools based on webrtcvad 
│
│
└── workspace             - this folder contains all files during model training and test
    └── exp               - this folder contains all model files
    └── feature           - this folder contains various feature types from data preparation
    └── package_feature   - this folder contains all package features from feature folder

