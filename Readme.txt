This is a demonstration of the Combining Tensor Slice and Singular Value for Blind Light Field Image Quality Assessment(TSSV-LFIQA)described in:

Zhiyong Pan, Mei Yu, Gangyi Jiang, Haiyong Xu, and Yo-Sung Ho, " Combining Tensor Slice and Singular Value for Blind Light Field Image Quality Assessment," IEEE Journal of Selected Topics in Signal Processing, 15(3): 672-687, 2021, DOI: 10.1109/JSTSP.2021.3056959.

% Code provided by Zhiyong Pan 2020.10.23

The code was written in MATLAB 2017a, and tested on Windows 10/7.
=======================================================================

An exmaple (Matlab codes, light field image and mos in Win5-LID[1])are provided to demonstrate how to use the package. 

I. RUNNING CODE

1. Feature Extration and Quality Prediction

run Demo_TSSV_LFIQA.m 

2. Regression 

run Demo_SVR.m

II. Package Composition

1. Feature

(1)Tensor Slice Spatial Feature (Including First_Grad_Feature.mat;First_Grad_Feature_Color.mat;Other_Entropy_Feature.mat) 
(2)Singular Value Angular Feature (Including SinValue_inter_Feature.mat;SinValue_intra_Feature.mat)

2. Functions (Subfunction)

3. Libraries (Including SVR and tensor_toolbox)

4. Demo_SVR.m (Regression model)

5. Demo_TSSV_LFIQA.m (Main Function)

6. LN_dishes_50.bmp (¨Light field image in Win5-LID[1]), downloaded in "https://pan.baidu.com/s/1_CNTo2kJxP1UMJxpuk7XRA"
password：4xr9

7. model.mat (Model trained on Win5-LID[1])

8. win5mos.mat (MOS in Win5-LID[1],sorted by scene)

9. Readme.txt(Manual)

[1]L. Shi, S. Zhao, W. Zhou, and Z. Chen, "Perceptual evaluation of light field image," in Proc. IEEE Int. Conf. Imag. Process., Athens, Greece, 2018, pp. 41-45.

III. VERSION HISTORY

v1.0 - Initial release
