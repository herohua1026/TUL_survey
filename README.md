# TUL_survey
instruction and codes for a survey on trajectory-user linking 

## Datasets

#### Raw data：

We utilize the following six datasets:<br/>
Gowalla: http://snap.stanford.edu/data/loc-gowalla.html<br/>
Brightkite: http://snap.stanford.edu/data/loc-brightkite.html<br/>
Foursquare(New York): https://sites.google.com/site/yangdingqi/home/foursquare-dataset<br/>
Geolife: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/<br/>
Tdrive and Blueteeh(BT_w) : These two datasets are not openly shared but are available on request from the authors.<br/>

Remark: These datasets are intended for academic purposes only and should not be used for commercial purposes. Please cite the relevant paper when using these datasets.

#### Prepocessed data：

- The dataset usage process is demonstrated in the following diagram:
![Screen Shot 2023-11-06 at 11 04 08 am](https://github.com/herohua1026/TUL_survey/assets/89980991/a35c5b1c-3aaf-40c6-9827-2b98cd791f2e)



















- The dataset usage pre-process includes two parts, reforming and segmentation.
- The pre-pocessed data sample is available in the folder [data](./data/reformed_data and segmented_data).
- Some models(e.g., TULER) requre POI embedding, which can be achieved using the word2vec method. The corresponding python file is located the folder[./data/word2vec].

## Usage
(Exajmple using TULER or GNNTUL. For other models, please refer to the instructions in their respetive code repositories.)
- Preprocess the original data according to the reprocessing diagram.
-  Modify the input path and filename in 'config.py'.
- Adjust the hyperparameters and strategies as needed.
- Run train_###.py(e.g, train_gnn.py for GNNTUL model)


## Dependencies

#Data Pre-process and Segmentation
- Python 3.9
- Pandas==1.3.4
- NumPy


#TULER/GNNTUL
The original code of TULER: https://github.com/KL-ice/TULER/tree/master
- Python 3.9.16
- torch==2.0.1
- NumPy

#TULVAE
The original code of TULVAE: https://github.com/AI-World/IJCAI-TULVAE/blob/master

-Tensorflow 1.0 or++-python 2.7
-numpy

#STULIG/STUL
The original code of STULIG/STUL:  https://github.com/gcooq/STULIG

-Tensorflow 1.0 or++-python 2.7
-numpy


#MainTUL
The original code of MainTUL : https://github.com/Onedean/MainTUL

- Python 3.9
- torch==1.10.1 (cuda10.2)
- scikit-learn==1.0.1
- tqdm==4.62.3
- pandas==1.3.4
- matplotlib==3.5.0



## Parameter setting of baselines
+ TULER and its variants: (LR: 0.00095 / Dimension: 250 / Hidden size: 300 / Dropout rate: 0.5 / Layers: 2)  
+ TULVAE: (LR: 0.001 / decays: 0.9 / $\beta$: 0.5-1 / POI Dimension: 250 / Hidden size: 300 / Latent Variable Dimension: 100)  
