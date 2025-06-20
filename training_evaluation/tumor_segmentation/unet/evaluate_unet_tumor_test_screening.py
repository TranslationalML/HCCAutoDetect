import re
import yaml
import os
import sys
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import SimpleITK as sitk
import json
from joblib import Parallel, delayed

eval_dir = os.path.dirname(os.getcwd())
print(eval_dir)
sys.path.append(eval_dir)
from eval_utils import *

import torch
from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    RandFlipd,
    ThresholdIntensityd,
    CastToTyped,
)


def main(data_dir, config):
    os.makedirs(config["output_dir"], exist_ok=True)
    config["saving_5CV_name"] = "tta{}_T{}_contrast_crit_{}_T{}_vein_crit_{}_T{}_prob_T{}".format(
        config["tta_num_examples"], config["tta_threshold"], config["contrast_criterion"], config["vein_mask_criterion"],
        config["vein_mask_threshold"], config["post_trans_threshold"])


    prob_map_per_fold = []
    images_per_fold = []
    preds_per_fold = []
    gt_per_fold = []
    gt_multilabels_per_fold = []
    liver_labels_per_fold = []
    vein_labels_per_fold = []
    for fold in range(0, 5, 1):
        print('Fold: ', fold)
        if config['model_params']['in_channels'] > 1:
            train_images = sorted(glob.glob(os.path.join(data_dir, "images_4D", "*.nii.gz")))
        else:
            train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*{}*.nii.gz".format(config['3D_image_type']))))
        train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
        train_multilabels = sorted(glob.glob(os.path.join(data_dir, "multi_labels", "*.nii.gz")))
        train_liver_labels = sorted(glob.glob(os.path.join(data_dir, "liver_labels", "*.nii.gz")))
        data_dicts = [{"img": image_name,
                            "seg": label_name,
                            "multi_labels": multi_labels,
                            "liver_label": liver_label_name,
                            }
                           for image_name, label_name, multi_labels, liver_label_name
                           in zip(train_images, train_labels, train_multilabels,
                                  train_liver_labels)]
        
        for data_dict in data_dicts:
            image_name = os.path.basename(data_dict['img']).split('_')
            label_name = os.path.basename(data_dict['seg']).split('_')
            liver_name = os.path.basename(data_dict['liver_label']).split('_')
            multi_label_name = os.path.basename(data_dict['multi_labels']).split('_')
            img_shape = sitk.GetArrayFromImage(sitk.ReadImage(data_dict['img']))[0, :, :, :].shape
            seg_shape = sitk.GetArrayFromImage(sitk.ReadImage(data_dict['seg'])).shape
            liver_shape = sitk.GetArrayFromImage(sitk.ReadImage(data_dict['liver_label'])).shape
            multi_label_shape = sitk.GetArrayFromImage(sitk.ReadImage(data_dict['multi_labels'])).shape

            assert image_name == label_name == label_name == liver_name == multi_label_name
            assert img_shape == seg_shape == liver_shape == multi_label_shape


        test_files = data_dicts
        test_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg", "multi_labels", "liver_label"]),
                EnsureChannelFirstd(keys=["img", "seg", "multi_labels", "liver_label"]),
                Orientationd(keys=["img", "seg", "multi_labels", "liver_label"], axcodes="RAS"),
                CastToTyped(keys=["img", "seg", "multi_labels", "liver_label"], dtype="float32"),
                Spacingd(keys=["img", "seg", "multi_labels", "liver_label"],
                         pixdim=config['pixel_resampling_size'],
                         mode=("nearest", "nearest", "nearest", "nearest"),
                         padding_mode=('zeros', 'zeros', 'zeros', 'zeros'),),

                ScaleIntensityRangePercentilesd(keys=["img"], lower=5, upper=100, b_min=0, b_max=1,
                                channel_wise=True),
                ThresholdIntensityd(keys=["img"], threshold=0, above=True, cval=0),
                ThresholdIntensityd(keys=["img"], threshold=1, above=False, cval=1),
                
            ])

        # create a validation data loader
        test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)

        test_loader = DataLoader(test_ds,
                                batch_size=config['val_batch_size'],
                                num_workers=config['num_workers'],
                                collate_fn=list_data_collate)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        files_path = os.path.join(train_dir, config["wandb_run_names"][fold] + "/files")
        model_list = sorted([os.path.join(files_path, file) for file in os.listdir(files_path) if config["max_epoch_criteria"] in file])
        if config["max_epoch_criteria"] == 'checkpoint':
            pattern = config["max_epoch_criteria"] + r'_(.*?)_CV'
            model_epoch_nbr = [int(re.search(pattern, file).group(1)) for file in model_list]
            epoch_max = max(model_epoch_nbr)
            model_path = [path for path in model_list if str(epoch_max) in path][0]
            output_dir = config["output_dir"] + "/predictions_checkpoint_" + str(epoch_max)
        elif config["max_epoch_criteria"] == 'last_checkpoint':
            model_path = model_list[0]
            output_dir = config["output_dir"] + "/predictions_last_checkpoint"
        elif config["max_epoch_criteria"] == 'best_metric_model':
            model_path = model_list[0]
            output_dir = config["output_dir"] + "/predictions_best_metric_model"
        else:
            raise ValueError("max_epoch_criteria not recognized")

        if len(model_list) == 0:
            print('No model found')
            continue

        lesion_detector = UNETTumorSegmentationEval(config, model_path)

        transform = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(keys=["img"], lower=5, upper=100, b_min=0, b_max=1, channel_wise=True),
            Spacingd(keys=["img", "seg"], pixdim=config['pixel_resampling_size'], mode=("nearest", "nearest"),
                     padding_mode=('zeros', 'zeros')),
            

            RandFlipd(keys=["img", "seg"], spatial_axis=[0], prob=0.50),
            RandFlipd(keys=["img", "seg"], spatial_axis=[1], prob=0.50),
            ThresholdIntensityd(keys=["img"], threshold=0, above=True, cval=0),
            ThresholdIntensityd(keys=["img"], threshold=1, above=False, cval=1),
        ])

        multi_mod = config['model_params']['in_channels']
        if not (os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/test' + f'/preprocessed_img')) and 
                os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/test' + f'/logit'))):
            print('Predict validation files')
            lesion_detector.predict(device, test_loader, output_dir, config, sub_dir='fold_' + str(fold) + '/test', multi_mod=multi_mod, fold=fold)
        
        
        if config['tta_num_examples'] == 0:
            if not os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/test' + f'/logit_{config["post_trans_threshold"]}')):
                print('Predict validation files')
                lesion_detector.predict(device, test_loader, output_dir, config, sub_dir='fold_' + str(fold) + '/test', multi_mod=multi_mod, fold=fold)
            elif len(os.listdir(os.path.join(output_dir, 'fold_' + str(fold) + '/test' + f'/preds_{config["post_trans_threshold"]}'))) != len(test_files):
                print('Predict validation files')
                lesion_detector.predict(device, test_loader, output_dir, config, sub_dir='fold_' + str(fold) + '/test', multi_mod=multi_mod, fold=fold)
            else:
                pass
        if config['tta_num_examples'] != 0:
            if (not os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/test/logit_tta'))):
                lesion_detector.test_time_predict(device, test_files, output_dir, config, transform,
                                                    sub_dir='fold_' + str(fold) + '/test', num_examples=config['tta_num_examples'], fold=fold)
            elif len(os.listdir(os.path.join(output_dir, 'fold_' + str(fold) + '/test/logit_tta'))) != len(test_files):
                lesion_detector.test_time_predict(device, test_files, output_dir, config, transform, sub_dir='fold_' + str(fold) + '/test', num_examples=config['tta_num_examples'], fold=fold)
            else:
                pass

        if config['tta_num_examples'] != 0:
            preds_path = (output_dir + '/fold_' + str(fold) + '/test/preds_tta_' + str(config['tta_num_examples']) + '_' + str(config['tta_threshold']))
        else:
            preds_path = output_dir + '/fold_' + str(fold) + f'/test/preds_{config["post_trans_threshold"]}'
        preds = sorted([os.path.join(preds_path, file) for file in os.listdir(preds_path)])

        if (not os.path.exists(os.path.dirname(preds_path)  + '/calibrated_prob_map_logit') and 
               not os.path.exists(os.path.dirname(preds_path)  + '/calibrated_prob_map_logit_tta')):
            if config['tta_num_examples'] != 0:
                logits = sorted([os.path.join(os.path.dirname(preds_path) + '/logit_tta', file) for file in
                                    os.listdir(preds_path)], key=lambda x: x[:-11])
            else:
                logits = sorted([os.path.join(os.path.dirname(preds_path) + '/logit', file) for file in os.listdir(preds_path)], key=lambda x: x[:-11])
            
            print('Calibrate predictions')
            lesion_detector.calibration(logits, output_dir + '/fold_' + str(fold) + '/test/', config)


        if config['tta_num_examples'] != 0:
            prob_maps_path = (output_dir + '/fold_' + str(fold) + '/test/calibrated_prob_map_logit_tta')
        else:
            prob_maps_path = (output_dir + '/fold_' + str(fold) + '/test/calibrated_prob_map')
        prob_maps = sorted([os.path.join(prob_maps_path, file) for file in os.listdir(prob_maps_path)])

        gt = sorted([file['seg'] for file in test_files])
        gt_multilabels = sorted([file.replace('labels', 'multi_labels') for file in gt])
        liver_labels = sorted([file.replace('labels', 'liver_labels') for file in gt])
        if config['vein_mask_criterion']:
            vein_labels = sorted(glob.glob(os.path.join(data_dir, config["vein_mask_path"], "*.nii.gz")))
        else:
            vein_labels = [[] for x in range(len(liver_labels))]
        prob_map_per_fold.append(prob_maps)
        gt_per_fold.append(gt)
        gt_multilabels_per_fold.append(gt_multilabels)
        liver_labels_per_fold.append(liver_labels)
        vein_labels_per_fold.append(vein_labels)

        if config['model_params']['in_channels'] > 1:
            images = sorted([file.replace('labels', 'images_4D') for file in gt])
        else:
            images = sorted([file['images'] for file in test_files])


        images_per_fold.append(images)
    plot_saving_path = (os.path.dirname(output_dir) + "/plots_tta_{}_thresh_{}_".format(config["tta_num_examples"], config['tta_threshold'])
                        + "post_thresh_" + str(config["post_trans_threshold"]) + "_mean_prob_map_thresh_{}".format(config['mean_prob_map_threshold']) + "_" + config["saving_5CV_name"])
    os.makedirs(plot_saving_path, exist_ok=True)

    if config['tta_num_examples'] != 0:
        prob_map_output_dir = os.path.dirname(output_dir) + "/mean_prob_maps/prob_map_tta"
    else:
        prob_map_output_dir = os.path.dirname(output_dir) + "/mean_prob_maps/prob_map"
    preds_output_dir = os.path.dirname(output_dir) + "/mean_prob_maps/preds_" + str(config["mean_prob_map_threshold"])

    
    if not os.path.exists(preds_output_dir):
        os.makedirs(prob_map_output_dir, exist_ok=True)
        os.makedirs(preds_output_dir, exist_ok=True)
        for patient in range(len(prob_map_per_fold[0])):
            patient_prob_maps = []
            for CV in range(len(prob_map_per_fold)):
                print(prob_map_per_fold[CV][patient])
                prob_map = sitk.GetArrayFromImage(sitk.ReadImage(prob_map_per_fold[CV][patient]))
                patient_prob_maps.append(prob_map)
            liver_label_ar = sitk.GetArrayFromImage(sitk.ReadImage(liver_labels_per_fold[0][patient]))
            mean_prob_map_ar = np.mean(patient_prob_maps, axis=0)
            mean_prob_map_ar = mean_prob_map_ar * liver_label_ar

            mean_prob_map = sitk.GetImageFromArray(mean_prob_map_ar)
            mean_prob_map.CopyInformation(sitk.ReadImage(prob_map_per_fold[0][patient]))
            sitk.WriteImage(mean_prob_map, prob_map_output_dir + '/' + os.path.basename(prob_map_per_fold[0][patient]).replace('_seg', ''))
            binary_map = np.zeros_like(mean_prob_map_ar)
            binary_map[mean_prob_map_ar > config["mean_prob_map_threshold"]] = 1
            binary_map[mean_prob_map_ar <= config["mean_prob_map_threshold"]] = 0
            binary_map = sitk.GetImageFromArray(binary_map)
            binary_map.CopyInformation(sitk.ReadImage(prob_map_per_fold[0][patient]))
            sitk.WriteImage(binary_map, preds_output_dir + '/' + os.path.basename(prob_map_per_fold[0][patient]).replace('_seg', ''))
    prob_maps = sorted([os.path.join(prob_map_output_dir, file) for file in os.listdir(prob_map_output_dir)])
    preds = sorted([os.path.join(preds_output_dir, file) for file in os.listdir(preds_output_dir)])
    gt = sorted(gt_per_fold[0])
    gt_multilabels = sorted(gt_multilabels_per_fold[0])
    liver_labels = sorted(liver_labels_per_fold[0])

    images = sorted(images_per_fold[0])
    lesions_characteristics = pd.read_csv(data_dir + '/tumors_characteristics.csv', sep=';')
    if lesions_characteristics.shape[1] == 1:
        lesions_characteristics = pd.read_csv(data_dir + '/tumors_characteristics.csv', sep=',')
    
    if config["FROC"] == True:
        for tumor_size in config["min_tumor_size"]:

            FROC_metrics_dict = Parallel(n_jobs=20)(delayed(lambda threshold: (threshold, eval_perf_fold(prob_maps, images, preds, gt, gt_multilabels, liver_labels, 
                                                                     vein_labels, tumor_size, config, output_dir, lesions_characteristics, 
                                                                     threshold=threshold)))(threshold) for threshold in config["FROC_thresholds"])
            FROC_metrics_dict = dict(FROC_metrics_dict)

            TP_per_patient = [float(np.sum([float(np.sum(sub_list)) for sub_list in metric_dict['TP_list'] if len(sub_list) > 0])) for thresh_key, metric_dict in FROC_metrics_dict.items()]
            FN_per_patient = [float(np.sum([float(np.sum(sub_list)) for sub_list in metric_dict['FN_list'] if len(sub_list) > 0])) for thresh_key, metric_dict in FROC_metrics_dict.items()]
            FP_per_patient = [float(np.sum(np.sum(metric_dict['FP_list']))) for thresh_key, metric_dict in FROC_metrics_dict.items()]
            TN_per_patient = [np.sum([float(np.sum(sub_list)) for sub_list in metric_dict['TN_list'] if len(sub_list) > 0]) for thresh_key, metric_dict in FROC_metrics_dict.items()]
            recall_per_threshold = [float(np.divide(float(TP), float(TP) + float(FN), out=np.zeros_like(float(TP)),
                                            where=(float(TP) + float(FN)) != 0)) for TP, FN in zip(TP_per_patient, FN_per_patient)]

            mean_FP_per_patient = [np.mean(metric_dict['FP_list']) for thresh_key, metric_dict in FROC_metrics_dict.items()]

            # strat perf
            for thresh in config['FROC_thresholds']:
                os.makedirs(plot_saving_path + '/strat_perf_thresh' + str(thresh), exist_ok=True)

                LIRADSWashoutCapsuleEval([[score for patient in FROC_metrics_dict[thresh]['LIRADS_list'] for score in patient],], 
                                         [FROC_metrics_dict[thresh]['TP_list']], 
                                         [FROC_metrics_dict[thresh]['FP_list']], 
                                         [FROC_metrics_dict[thresh]['caps_ven_list']],
                                         [FROC_metrics_dict[thresh]['caps_del_list']],
                                         [FROC_metrics_dict[thresh]['wash_ven_list']],
                                         [FROC_metrics_dict[thresh]['wash_del_list']],
                                         [FROC_metrics_dict[thresh]['hyper_art_list']],
                                         tumor_size,
                                         plot_saving_path + '/strat_perf_thresh' + str(thresh))

            plt.figure()
            plt.plot([0] + list(mean_FP_per_patient)[::-1], [0] + list(recall_per_threshold)[::-1], marker='o')
            plt.xlabel('Mean number of false Positives')
            plt.ylabel('Sensitivity')
            plt.ylim(0, 1)
            plt.title('Mean FROC Curve')
            plt.savefig(plot_saving_path + "/FROC_tumor_wise{}.png".format(tumor_size))
            plt.close()

            healthy_subjects = ['sub-' + str(i).zfill(3) + '.nii.gz' for i in range(65, 76)] + ['sub-' + str(i).zfill(3) + '.nii.gz' for i in range(118, 300)]
            TP_patients = [float(sum([1 if np.sum(TP) > 0 else 0 for patient, TP in zip(metric_dict['patients'], metric_dict['TP_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
            FP_patients = [float(sum([1 if np.sum(FP) > 0 and patient in healthy_subjects else 0 for patient, FP in zip(metric_dict['patients'], metric_dict['FP_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
            FN_patients = [float(sum([1 if np.sum(FN) > 0 else 0 for patient, FN in zip(metric_dict['patients'], metric_dict['FN_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
            TN_patients = [float(sum([1 if np.sum(TN) > 0 and patient in healthy_subjects else 0 for patient, TN in zip(metric_dict['patients'], metric_dict['TN_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
            patient_recall = [float(np.divide(float(TP), float(TP) + float(FN), out=np.zeros_like(float(TP)),
                                            where=(float(TP) + float(FN)) != 0)) for TP, FN in zip(TP_patients, FN_patients)]

            FROC_metrics_str = str(FROC_metrics_dict)
            if config['contrast_criterion']:
                save_name = 'FROC_metrics_size_{}_art_{}_art.json'.format(tumor_size, config['min_art_contrast'])
            else:
                save_name = 'FROC_metrics_size_{}.json'.format(tumor_size)

            with open(os.path.join(plot_saving_path, save_name), 'w') as f:
                json.dump({'TP_per_patient': TP_per_patient, 
                           'FP_per_patient': FP_per_patient, 
                           'FN_per_patient': FN_per_patient, 
                           'TN_per_patient': TN_per_patient, 
                           'recall_per_patient': recall_per_threshold, 
                           'mean_FP_per_patient': mean_FP_per_patient, 
                           'TP_patients': TP_patients, 
                           'FP_patients': FP_patients,
                            'FN_patients': FN_patients, 
                            'TN_patients': TN_patients, 
                            'patient_recall': patient_recall, 
                            'FROC_metrics': FROC_metrics_str}, f)


    else:
        for tumor_size in config["min_tumor_size"]:
            FROC_metrics_dict = eval_perf_fold(prob_maps, images, preds, gt, gt_multilabels, liver_labels, tumor_size,
                                        config, output_dir, lesions_characteristics)

    print()


if __name__ == "__main__":
    train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'training/tumor_segmentation/unet'))
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))

    data_dir = os.path.join(main_dir, 'HCC_Surveillance/derivatives/10_test_T1_dataset')
    # data_dir = os.path.join(main_dir, 'HCC_pre_ablation/derivatives/6_T1_dataset')

    # experiment = 'tumor_segmentation/tumor_segmentation_pretraining/2. Unet_Tversky.yaml'
    experiment = 'tumor_segmentation/tumor_segmentation_finetuning/3. Unet_pretrained_Tversky_loss.yaml.yaml'

    FROC_thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]


    output_folder_dir = main_dir + '/training_evaluation/tumor_segmentation/unet'
    train_config_path = os.path.join(main_dir, 'training_experiments', experiment)
    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load(file)

    wandb_run_names = [train_config["saving_name_wandb_fold_{}".format(fold)] for fold in range(5)]
    wandb_run_names = ['wandb/' + run_name for run_name in wandb_run_names]

    if len(sys.argv) > 1:
        fold = int(sys.argv[1])
    else:
        fold = 0
    
    for fold in range(0, 1, 1):
        output_dir = os.path.join("tumor_eval",  experiment[:-5], os.path.basename(data_dir) + '_' + str(fold))

        eval_config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_finetuning/evaluation.yaml')
        with open(eval_config_path, 'r') as file:
            eval_config = yaml.safe_load(file)
        output_dir = output_folder_dir + '/' + output_dir
        merged_config = {**train_config, **eval_config}
        merged_config["output_dir"] = output_dir
        merged_config["wandb_run_names"] = wandb_run_names
        merged_config["predict_test"] = True
        merged_config["FROC"] = True
        merged_config['padding_mode'] = 'circular'
        merged_config["max_epoch_criteria"] = "last_checkpoint"
        merged_config["overlap"] = (0.5, 0.5, 0.5)
        merged_config['tta_num_examples'] = 0

        merged_config["FROC_thresholds"] = FROC_thresholds
        merged_config['min_tumor_size'] = [10]
        merged_config['min_art_contrast'] = 0.9
        merged_config['min_del_contrast'] = 0.5
        merged_config['contrast_criterion'] = False
        merged_config['save_plot'] = False
        merged_config['vein_mask_criterion'] = True

        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        merged_config['lesion_characteristics'] = os.path.join(os.path.dirname(data_dir), 'tumors_characteristics.csv')
        main(data_dir, merged_config)