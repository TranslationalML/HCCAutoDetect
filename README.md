# HCCAutoDetect
This repository contains the code used in our study "Deep Learning for Automatic Detection of Hepatocellular Carcinoma in Dynamic Contrast-Enhanced MRI".

## 1. Datasets preprocessing
A folder was created to preprocess the data of each dataset separately. The preprocessing was organized as follow:

### 1.1 Imaging data preparation
- Convert data to nifti and organise in a BIDS-like structure
- Reorientation 
- Bias field correction
- T1wi DCE phases selection
- Liver mask prediction on venous phase with nnU-Net
- Pairwise registration
- Groupwise registration
- Liver mask prediction on venous phase with nnU-Net

### 1.2 Annotation data preparation
- Convert RTSS structures to 3D volumes
- Apply registration transforms

### 1.3 Post processing
- Totalsegmentator was used to generate a segmentation mask of the veins to remove false positive predictions

### 1.4 Create dataset
- Build a dataset with aligned images and annotations to train neural networks

## 2. Datasets specificities

| Dataset | Imaging type | Lesion annotations provided | Liver annotations provided | Specificity |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| [LiTS](LiTS/)  | CT  | 3D | 3D | No specific preprocessing, the liver mask was already provided in the annotations which allowed to extract the liver from the imaging to construct the dataset. The 3D volume of each patient was concatenated to form 4D volumes matching the input of the U-Net architecture|
| [LLD-MMRI](LLD_MMRI/)  | DCE MRI  | 3D | None | The liver probability maps from the nnunet were thresholded with specific thresholds and an additional margin around the mask was set to reduce undersegmentation in [4_connected_components_liver_predictions.py](LLD_MMRI/code/preprocessing/4_connected_components_liver_predictions.py) and [8_connected_components_liver_predictions.py](LLD_MMRI/code/preprocessing/8_connected_components_liver_predictions.py).|
| [HCC Surveillance](HCC_surveillance/)  | DCE MRI | 2D | None | Two different datasets were created with that cohort, 1 was made for training models with only HCC positive patients in [7_build_reg_T1_dataset.py](HCC_surveillance/code/preprocessing/7_build_reg_T1_dataset.py) and the second [7_build_reg_T1_test_dataset.py](HCC_surveillance/code/preprocessing/7_build_reg_T1_test_dataset.py) with a mixture of HCC positive and HCC negative patients for testing the models.|
| [HCC Pre-Ablation](HCC_pre_ablation/)  | DCE MRI | 2D | None | The dataset included only HCC positive patients and was used only for testing models.|

## 3. Liver and Tumor Segmentation Models

| Models | Training files | Experiment settings |
| --- | --- |--- |
| Liver segmentation|[nnunet_training_preprocessing.py](training/liver_segmentation/nnunet/nnunet_training_liver.py)|[nnunet_liver_segmentation.yaml](training_experiments/liver_segmentation/nnunet_liver_segmentation.yaml)|
| 1. Benchmark nnU-Net | [nnunet_training_multimodality_tumor.py](training/tumor_segmentation/nnunet/nnunet_training_multimodality_tumor.py)|[Benchmark_nnunet.yaml](training_experiments/tumor_segmentation/tumor_segmentation/1_Benchmark_nnunet.yaml)|
| 2. U-Net Tversky | [train_unet_tumor.py](training/tumor_segmentation/unet/train_unet_tumor.py)|[Unet_Tversky.yaml](training_experiments/tumor_segmentation/tumor_segmentation/2_Unet_Tversky.yaml)|
| 3. U-Net Pre-trained + Tversky |I. Attention U-Net trained on LiTS: [train_unet_tumor_LiTS_pretraining.py](training/tumor_segmentation/unet/train_unet_tumor_LiTS_pretraining.py)|[LiTS_model.yaml](training_experiments/tumor_segmentation/tumor_segmentation/3.1_LiTS_model.yaml)|
| |II. Fintuning on LLD-MMRI dataset: [train_unet_tumor_finetuning_LiTS_on_LLD_MMRI.py](training/tumor_segmentation/unet/train_unet_tumor_finetuning_LiTS_on_LLD_MMRI.py)|[LiTS_finetuned_on_LLD_MMRI.yaml](training_experiments/tumor_segmentation/tumor_segmentation/3.2_LiTS_finetuned_on_LLD_MMRI.yaml)|
| |III. Finetuning on HCC Surveillance dataset: [train_unet_tumor.py](training/tumor_segmentation/unet/train_unet_tumor.py)|[Unet_pretrained_Tversky_loss.yaml](training_experiments/tumor_segmentation/tumor_segmentation/3.3_Unet_pretrained_Tversky_loss.yaml)|

## 4. Models evaluation
### Inference and performance metrics
| Architecture  | Inference | Performance evaluation |
| ------------- | ------------- |------------- |
|  1. Benchmark nnU-Net  | [nnunet_predict_multimodality_tumor.py](training_evaluation/tumor_segmentation/nnunet/nnunet_predict_multimodality_tumor_test.py) | [evaluate_tumor_predictions_CV.py](training_evaluation/tumor_segmentation/nnunet/evaluate_tumor_predictions_CV.py) |
| 2. U-Net Tversky<br>3. U-Net Pre-trained + Tversky | Validation: [evaluate_unet_tumor.py](training_evaluation/tumor_segmentation/unet/evaluate_unet_tumor.py)<br> Test: [evaluate_unet_tumor_test_screening.py](training_evaluation/tumor_segmentation/unet/evaluate_unet_tumor_test_screening.py)  | Validation: [evaluate_unet_tumor.py](training_evaluation/tumor_segmentation/unet/evaluate_unet_tumor.py)<br> Test: [evaluate_unet_tumor_test_screening.py](training_evaluation/tumor_segmentation/unet/evaluate_unet_tumor_test_screening.py) |
| |||

### FROC Curves
The comparison of model performance was made with FROC curves in [AUC_validation.py](training_evaluation/tumor_segmentation/eval_all/AUC_validation.py) and [AUC_test.py](training_evaluation/tumor_segmentation/eval_all/AUC_test.py)

## Technology
- Python 3.8.16 and 3.9.18 

