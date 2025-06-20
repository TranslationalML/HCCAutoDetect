import re
import yaml
from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import json
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed


training_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'training/tumor_segmentation/unet'))
sys.path.append(training_dir)

eval_dir = os.path.dirname(os.getcwd())
sys.path.append(eval_dir)
from eval_utils import *

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    RandFlipd,
    ThresholdIntensityd
)

def main(data_dir, train_dir, config):
    os.makedirs(config['output_dir'], exist_ok=True)
    config["saving_5CV_name"] = "evaluation_5CV"
    for tumor_size in config["min_tumor_size"]:
        patient_tumor_dice_per_fold = []
        tumor_wise_dice_per_fold = []

        TP_per_patient_per_fold = []
        FP_per_patient_per_fold = []
        FN_per_patient_per_fold = []
        TN_per_patient_per_fold = []
        gen_neg_samples_per_fold = []
        LIRADS_per_fold = []
        ven_wash_per_fold = []
        del_wash_per_fold = []
        ven_caps_per_fold = []
        del_caps_per_fold = []
        hyper_art_per_fold = []
        FROC_metrics_per_fold = []
        patients_per_fold = []
        gt_per_fold = []
        prob_map_per_fold = []
        liver_labels_per_fold = []
        vein_labels_per_fold = []
        images_per_fold = []
        gt_multilabels_per_fold = []

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

            if fold == 0:
                val_files = data_dicts[int(0.8 * len(data_dicts)):]
                train_files = [file for file in data_dicts if file not in val_files]
            if fold == 1:
                val_files = data_dicts[int(0.6 * len(data_dicts)):int(0.8 * len(data_dicts))]
                train_files = [file for file in data_dicts if file not in val_files]
            if fold == 2:
                val_files = data_dicts[int(0.4 * len(data_dicts)):int(0.6 * len(data_dicts))]
                train_files = [file for file in data_dicts if file not in val_files]
            if fold == 3:
                val_files = data_dicts[int(0.2 * len(data_dicts)):int(0.4 * len(data_dicts))]
                train_files = [file for file in data_dicts if file not in val_files]
            if fold == 4:
                val_files = data_dicts[:int(0.2 * len(data_dicts))]
                train_files = [file for file in data_dicts if file not in val_files]

            train_files = train_files
            val_files = val_files
            print("Validation files found: ", len(val_files))
            val_transforms = Compose(
                [
                    LoadImaged(keys=["img", "seg", "multi_labels", "liver_label"]),
                    EnsureChannelFirstd(keys=["img", "seg", "multi_labels", "liver_label"]),
                    Orientationd(keys=["img", "seg", "multi_labels", "liver_label"], axcodes="RAS"),
                    ScaleIntensityRangePercentilesd(keys=["img"], lower=5, upper=100, b_min=0, b_max=1,
                                                    channel_wise=True),
                    Spacingd(keys=["img", "seg", "multi_labels", "liver_label"],
                             pixdim=config['pixel_resampling_size'],
                             mode=("nearest", "nearest", "nearest", "nearest"),
                             padding_mode=('zeros', 'zeros', 'zeros', 'zeros')),
                    ThresholdIntensityd(keys=["img"], threshold=0, above=True, cval=0),
                    ThresholdIntensityd(keys=["img"], threshold=1, above=False, cval=1),
                ])

            train_ds = monai.data.Dataset(data=train_files, transform=val_transforms)
            train_loader = DataLoader(
                train_ds,
                batch_size=config['train_batch_size'],
                num_workers=config['num_workers'],
                collate_fn=list_data_collate,
            )

            val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
            val_loader = DataLoader(val_ds,
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
                output_dir = os.getcwd() + "/" + config["output_dir"] + "/predictions_checkpoint_" + str(epoch_max)
            elif config["max_epoch_criteria"] == 'last_checkpoint':
                model_path = model_list[0]
                output_dir = os.getcwd() + "/" + config["output_dir"] + "/predictions_last_checkpoint"
            elif config["max_epoch_criteria"] == 'best_metric_model':
                model_path = model_list[0]
                output_dir = os.getcwd() + "/" + config["output_dir"] + "/predictions_best_metric_model"
                
            else:
                pattern = config["max_epoch_criteria"] + '_epoch_' + r'(.*?)_'
                model_epoch_nbr = [int(re.search(pattern, file).group(1)) for file in model_list]
                epoch_max = max(model_epoch_nbr)
                model_path = [path for path in model_list if str(epoch_max) in path][0]
                output_dir = os.getcwd() + "/" + config["output_dir"] + "/predictions_checkpoint_" + str(epoch_max)
            
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
            if not (os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/val' + f'/preprocessed_img')) and 
                    os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/val' + f'/logit')) and 
                    os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/val' + f'/preds_0.5'))):
                print('Predict validation files')
                lesion_detector.predict(device, val_loader, output_dir, config, sub_dir='fold_' + str(fold) + '/val', multi_mod=multi_mod, fold=fold)
            else:
                if len(os.listdir(os.path.join(output_dir, 'fold_' + str(fold) + '/val' + f'/logit'))) < len(val_files):
                    print('Predict validation files')
                    lesion_detector.predict(device, val_loader, output_dir, config, sub_dir='fold_' + str(fold) + '/val', multi_mod=multi_mod, fold=fold)

            if (not os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/val/logit_tta_' + str(config['tta_num_examples'])))
                and config['tta_num_examples'] != 0):
                lesion_detector.test_time_predict(device, val_files, output_dir, config, transform,
                                                  sub_dir='fold_' + str(fold) + '/val', num_examples=config['tta_num_examples'], fold=fold)
                
            # predict train
            if config["predict_train"] and not os.path.isdir(os.path.join(output_dir, 'fold_' + str(fold) + '/train')):
                print('Predict training files')
                lesion_detector.predict(device, train_loader, output_dir, config, sub_dir='fold_' + str(fold) + '/train', multi_mod=multi_mod, fold=fold)

            if config['tta_num_examples'] != 0:
                preds_path = (output_dir + '/fold_' + str(fold) + '/val/preds_tta_' +
                                  str(config['tta_num_examples']) + '_' + str(config['tta_threshold']))
            else:
                preds_path = output_dir + '/fold_' + str(fold) + f'/val/preds_{config["post_trans_threshold"]}'
                    
            preds = sorted([os.path.join(preds_path, file) for file in os.listdir(preds_path)], key=lambda x: x[:-11])

            if (not os.path.exists(os.path.dirname(preds_path)  + '/calibrated_prob_map_logit') and 
               not os.path.exists(os.path.dirname(preds_path)  + '/calibrated_prob_map_logit_tta')):
                if config['tta_num_examples'] != 0:
                    logits = sorted([os.path.join(os.path.dirname(preds_path) + '/logit_tta', file) for file in
                                        os.listdir(preds_path)], key=lambda x: x[:-11])
                else:
                    logits = sorted([os.path.join(os.path.dirname(preds_path) + '/logit', file) for file in os.listdir(preds_path)], key=lambda x: x[:-11])
                
                print('Calibrate predictions')
                lesion_detector.calibration(logits, output_dir + '/fold_' + str(fold) + '/val', config)

            prob_maps = sorted([os.path.join(os.path.dirname(preds_path) + '/calibrated_prob_map_logit_tta', file) for file in
                                os.listdir(preds_path)], key=lambda x: x[:-11])
            prob_map_per_fold.append(prob_maps)

            gt = sorted([file['seg'] for file in val_files], key=lambda x: x[:-7])
            gt_multilabels = sorted([file.replace('labels', 'multi_labels') for file in gt], key=lambda x: x[:-7])
            liver_labels = sorted([file.replace('labels', 'liver_labels') for file in gt], key=lambda x: x[:-7])
            if config['vein_mask_criterion']:
                vein_labels =  sorted([file.replace('labels', 'total_segmentator') for file in gt], key=lambda x: x[:-7])
            else:
                vein_labels = [[] for x in range(len(liver_labels))]

            if config['model_params']['in_channels'] > 1:
                images = sorted([file.replace('labels', 'images_4D') for file in gt])
            else:
                images = sorted([file['images'] for file in val_files])

            gt_per_fold.append(gt)
            gt_multilabels_per_fold.append(gt_multilabels)
            liver_labels_per_fold.append(liver_labels)
            vein_labels_per_fold.append(vein_labels)
            images_per_fold.append(images)
            

            lesions_characteristics = pd.read_csv(data_dir + '/tumors_characteristics.csv')
            if config["FROC"] == True:
                FROC_metrics = {}
                FROC_metrics = Parallel(n_jobs=20)(delayed(lambda threshold: (threshold, eval_perf_fold(prob_maps, images, preds, gt, gt_multilabels, liver_labels, 
                                                                     vein_labels, tumor_size, config, output_dir, lesions_characteristics, 
                                                                     threshold=threshold, nbr_of_neg_patch_trials=50)))(threshold) for threshold in config["FROC_thresholds"])
                FROC_metrics_dict = dict(FROC_metrics)
                FROC_metrics_per_fold.append(FROC_metrics_dict)
            else:
                (TP_list, FP_list, FN_list, TN_list, patients,
                 patient_tumor_metrics_list, tumor_metrics_list, generated_neg_samples, LIRADS,
                     ven_wash, del_wash, ven_caps, del_caps, hyper_art
                     ) = eval_perf_fold(prob_maps, images, preds, gt, gt_multilabels, liver_labels, vein_labels, tumor_size,
                                        config, output_dir, lesions_characteristics)
                LIRADS_per_fold.append(LIRADS)
                ven_wash_per_fold.append(ven_wash)
                del_wash_per_fold.append(del_wash)
                ven_caps_per_fold.append(ven_caps)
                del_caps_per_fold.append(del_caps)
                hyper_art_per_fold.append(hyper_art)
                TP_per_patient_per_fold.append(TP_list)
                FP_per_patient_per_fold.append(FP_list)
                FN_per_patient_per_fold.append(FN_list)
                TN_per_patient_per_fold.append(TN_list)
                gen_neg_samples_per_fold.append(generated_neg_samples)
                patient_tumor_dice_per_fold.append(patient_tumor_metrics_list)
                tumor_wise_dice_per_fold.append(tumor_metrics_list)
                patients_per_fold.append(patients)
        
        
        if config["predict_test"]:
            plot_saving_path = (os.path.dirname(output_dir) + "/Test_plots_tta_{}_thresh_{}_".format(config["tta_num_examples"],
                                config['tta_threshold']) + "post_thresh_" + str(config["post_trans_threshold"]) + "_" + config["saving_5CV_name"])
        else:
            plot_saving_path = (os.path.dirname(output_dir) + "/plots_tta_{}_thresh_{}_".format(config["tta_num_examples"], config['tta_threshold'])
                            + "post_thresh_" + str(config["post_trans_threshold"]) + "_" + config["saving_5CV_name"])
        os.makedirs(plot_saving_path, exist_ok=True)
        if config["FROC"] == True:
            # mean FROC CV
            TP_froc_per_patient_all_fold = []
            for FROC_list in FROC_metrics_per_fold:
                TP_froc_per_patient = [[np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_list.items()]
                TP_froc_per_patient_all_fold.append(TP_froc_per_patient)
            TP_froc_per_patient_all = np.concatenate(TP_froc_per_patient_all_fold, axis=1)
            
            TP_froc_per_patient_all = []
            FP_froc_per_patient_all = []
            FN_froc_per_patient_all = []
            TN_froc_per_patient_all = []
            TP_LR_5_froc_per_patient_all = []
            TP_LR_4_froc_per_patient_all = []
            TP_LR_3_froc_per_patient_all = []
            TP_LR_TIV_froc_per_patient_all = []
            TP_LR_M_froc_per_patient_all = []
            TP_ven_wash_froc_per_patient_all = []
            TP_del_wash_froc_per_patient_all = []
            TP_ven_caps_froc_per_patient_all = []
            TP_del_caps_froc_per_patient_all = []
            TP_hyper_art_froc_per_patient_all = []
            total_lesion_per_patient_all = []
            dice_per_patient = []
            for FROC_metrics in FROC_metrics_per_fold:
                total_lesion_per_patient = {threshold: [len(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics.items()}
                TP_froc_per_patient = {threshold: [np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics.items()}
                FP_froc_per_patient = {threshold: FROC_list['FP_list'] for threshold, FROC_list in FROC_metrics.items()}
                FN_froc_per_patient = {threshold: [np.sum(FN_list) for FN_list in FROC_list['FN_list']] for threshold, FROC_list in FROC_metrics.items()}
                TN_froc_per_patient = {threshold: [np.sum(TN_list) for TN_list in FROC_list['TN_list']] for threshold, FROC_list in FROC_metrics.items()}
                patient_dice = {threshold: FROC_list['patient_tumor_metrics_list'] for threshold, FROC_list in FROC_metrics.items()}
                dice_per_patient.append(patient_dice)

                total_lesion_per_patient_all.append(total_lesion_per_patient)
                TP_froc_per_patient_all.append(TP_froc_per_patient)
                FP_froc_per_patient_all.append(FP_froc_per_patient)
                FN_froc_per_patient_all.append(FN_froc_per_patient)
                TN_froc_per_patient_all.append(TN_froc_per_patient)
                
                TP_LR_5_froc_per_patient = {threshold: [np.sum([1 for TP, LIRADS in zip(TP_list, LIRADS_list) if LIRADS == '5' and TP == 1]) for TP_list, LIRADS_list in zip(FROC_list['TP_list'], FROC_list['LIRADS_list'])] 
                                           for threshold, FROC_list in FROC_metrics.items()}
                TP_LR_4_froc_per_patient = {threshold: [np.sum([1 for TP, LIRADS in zip(TP_list, LIRADS_list) if LIRADS == '4' and TP == 1]) for TP_list, LIRADS_list in zip(FROC_list['TP_list'], FROC_list['LIRADS_list'])]
                                             for threshold, FROC_list in FROC_metrics.items()}
                TP_LR_3_froc_per_patient = {threshold: [np.sum([1 for TP, LIRADS in zip(TP_list, LIRADS_list) if LIRADS == '3' and TP == 1]) for TP_list, LIRADS_list in zip(FROC_list['TP_list'], FROC_list['LIRADS_list'])]
                                                for threshold, FROC_list in FROC_metrics.items()}
                TP_LR_TIV_froc_per_patient = {threshold: [np.sum([1 for TP, LIRADS in zip(TP_list, LIRADS_list) if LIRADS == 'TIV' and TP == 1]) for TP_list, LIRADS_list in zip(FROC_list['TP_list'], FROC_list['LIRADS_list'])]
                                                  for threshold, FROC_list in FROC_metrics.items()}
                TP_LR_M_froc_per_patient = {threshold: [np.sum([1 for TP, LIRADS in zip(TP_list, LIRADS_list) if LIRADS == 'M' and TP == 1]) for TP_list, LIRADS_list in zip(FROC_list['TP_list'], FROC_list['LIRADS_list'])]
                                                for threshold, FROC_list in FROC_metrics.items()}
                TP_ven_wash_froc_per_patient = {threshold: [np.sum([1 for TP, ven_wash in zip(TP_list, ven_wash_list) if ven_wash == 1 and TP == 1]) for TP_list, ven_wash_list in zip(FROC_list['TP_list'], FROC_list['wash_ven_list'])]
                                                    for threshold, FROC_list in FROC_metrics.items()}
                TP_del_wash_froc_per_patient = {threshold: [np.sum([1 for TP, del_wash in zip(TP_list, del_wash_list) if del_wash == 1 and TP == 1]) for TP_list, del_wash_list in zip(FROC_list['TP_list'], FROC_list['wash_del_list'])]
                                                    for threshold, FROC_list in FROC_metrics.items()}
                TP_ven_caps_froc_per_patient = {threshold: [np.sum([1 for TP, ven_caps in zip(TP_list, ven_caps_list) if ven_caps == 1 and TP == 1]) for TP_list, ven_caps_list in zip(FROC_list['TP_list'], FROC_list['caps_ven_list'])]
                                                    for threshold, FROC_list in FROC_metrics.items()}
                TP_del_caps_froc_per_patient = {threshold: [np.sum([1 for TP, del_caps in zip(TP_list, del_caps_list) if del_caps == 1 and TP == 1]) for TP_list, del_caps_list in zip(FROC_list['TP_list'], FROC_list['caps_del_list'])]
                                                    for threshold, FROC_list in FROC_metrics.items()}
                TP_hyper_art_froc_per_patient = {threshold: [np.sum([1 for TP, hyper_art in zip(TP_list, hyper_art_list) if hyper_art == 1 and TP == 1]) for TP_list, hyper_art_list in zip(FROC_list['TP_list'], FROC_list['hyper_art_list'])]
                                                    for threshold, FROC_list in FROC_metrics.items()}
                
                TP_LR_5_froc_per_patient_all.append(TP_LR_5_froc_per_patient)
                TP_LR_4_froc_per_patient_all.append(TP_LR_4_froc_per_patient)
                TP_LR_3_froc_per_patient_all.append(TP_LR_3_froc_per_patient)
                TP_LR_TIV_froc_per_patient_all.append(TP_LR_TIV_froc_per_patient)
                TP_LR_M_froc_per_patient_all.append(TP_LR_M_froc_per_patient)
                TP_ven_wash_froc_per_patient_all.append(TP_ven_wash_froc_per_patient)
                TP_del_wash_froc_per_patient_all.append(TP_del_wash_froc_per_patient)
                TP_ven_caps_froc_per_patient_all.append(TP_ven_caps_froc_per_patient)
                TP_del_caps_froc_per_patient_all.append(TP_del_caps_froc_per_patient)
                TP_hyper_art_froc_per_patient_all.append(TP_hyper_art_froc_per_patient)

            patient_dice_all_CV = {key_threshold: [elem for patient_dice in dice_per_patient for elem in patient_dice[key_threshold]] for key_threshold in dice_per_patient[0].keys()}

            TP_froc_per_patient_all_CV = {key_threshold: [elem for TP_froc in TP_froc_per_patient_all for elem in TP_froc[key_threshold]] for key_threshold in TP_froc_per_patient_all[0].keys()}
            FP_froc_per_patient_all_CV = {key_threshold: [elem for FP_froc in FP_froc_per_patient_all for elem in FP_froc[key_threshold]] for key_threshold in FP_froc_per_patient_all[0].keys()}
            mean_FP_per_patient_all_CV = {key_threshold: np.mean([np.sum(FP_) for FP_ in FP_list ]) for key_threshold, FP_list in FP_froc_per_patient_all_CV.items()}
            FN_froc_per_patient_all_CV = {key_threshold: [elem for FN_froc in FN_froc_per_patient_all for elem in FN_froc[key_threshold]] for key_threshold in FN_froc_per_patient_all[0].keys()}
            TN_froc_per_patient_all_CV = {key_threshold: [elem for TN_froc in TN_froc_per_patient_all for elem in TN_froc[key_threshold]] for key_threshold in TN_froc_per_patient_all[0].keys()}
            TP_LR_5_froc_per_patient_all_CV = {key_threshold: [elem for TP_LR_5_froc in TP_LR_5_froc_per_patient_all for elem in TP_LR_5_froc[key_threshold]] for key_threshold in TP_LR_5_froc_per_patient_all[0]}
            TP_LR_4_froc_per_patient_all_CV = {key_threshold: [elem for TP_LR_4_froc in TP_LR_4_froc_per_patient_all for elem in TP_LR_4_froc[key_threshold]] for key_threshold in TP_LR_4_froc_per_patient_all[0]}
            TP_LR_3_froc_per_patient_all_CV = {key_threshold: [elem for TP_LR_3_froc in TP_LR_3_froc_per_patient_all for elem in TP_LR_3_froc[key_threshold]] for key_threshold in TP_LR_3_froc_per_patient_all[0]}
            TP_LR_TIV_froc_per_patient_all_CV = {key_threshold: [elem for TP_LR_TIV_froc in TP_LR_TIV_froc_per_patient_all for elem in TP_LR_TIV_froc[key_threshold]] for key_threshold in TP_LR_TIV_froc_per_patient_all[0]}
            TP_LR_M_froc_per_patient_all_CV = {key_threshold: [elem for TP_LR_M_froc in TP_LR_M_froc_per_patient_all for elem in TP_LR_M_froc[key_threshold]] for key_threshold in TP_LR_M_froc_per_patient_all[0]}
            TP_ven_wash_froc_per_patient_all_CV = {key_threshold: [elem for TP_ven_wash_froc in TP_ven_wash_froc_per_patient_all for elem in TP_ven_wash_froc[key_threshold]] for key_threshold in TP_ven_wash_froc_per_patient_all[0]}
            TP_del_wash_froc_per_patient_all_CV = {key_threshold: [elem for TP_del_wash_froc in TP_del_wash_froc_per_patient_all for elem in TP_del_wash_froc[key_threshold]] for key_threshold in TP_del_wash_froc_per_patient_all[0]}
            TP_ven_caps_froc_per_patient_all_CV = {key_threshold: [elem for TP_ven_caps_froc in TP_ven_caps_froc_per_patient_all for elem in TP_ven_caps_froc[key_threshold]] for key_threshold in TP_ven_caps_froc_per_patient_all[0]}
            TP_del_caps_froc_per_patient_all_CV = {key_threshold: [elem for TP_del_caps_froc in TP_del_caps_froc_per_patient_all for elem in TP_del_caps_froc[key_threshold]] for key_threshold in TP_del_caps_froc_per_patient_all[0]}
            TP_hyper_art_froc_per_patient_all_CV = {key_threshold: [elem for TP_hyper_art_froc in TP_hyper_art_froc_per_patient_all for elem in TP_hyper_art_froc[key_threshold]] for key_threshold in TP_hyper_art_froc_per_patient_all[0]}

            total_lesion_per_patient_all_CV = {key_threshold: [elem for total_lesion in total_lesion_per_patient_all for elem in total_lesion[key_threshold]] for key_threshold in total_lesion_per_patient_all[0].keys()}
            total_TP_per_threshold = {threshold_key: float(np.sum(TP_values)) for threshold_key, TP_values in TP_froc_per_patient_all_CV.items()}
            total_FP_per_threshold = {threshold_key: float(np.sum(FP_values)) for threshold_key, FP_values in FP_froc_per_patient_all_CV.items()}
            total_FN_per_threshold = {threshold_key: float(np.sum(FN_values)) for threshold_key, FN_values in FN_froc_per_patient_all_CV.items()}
            total_TN_per_threshold = {threshold_key: float(np.sum(TN_values)) for threshold_key, TN_values in TN_froc_per_patient_all_CV.items()}
            total_TP_LR_5_per_threshold = {threshold_key: np.sum(TP_LR_5_values) for threshold_key, TP_LR_5_values in TP_LR_5_froc_per_patient_all_CV.items()}
            total_TP_LR_4_per_threshold = {threshold_key: np.sum(TP_LR_4_values) for threshold_key, TP_LR_4_values in TP_LR_4_froc_per_patient_all_CV.items()}
            total_TP_LR_3_per_threshold = {threshold_key: np.sum(TP_LR_3_values) for threshold_key, TP_LR_3_values in TP_LR_3_froc_per_patient_all_CV.items()}
            total_TP_LR_TIV_per_threshold = {threshold_key: np.sum(TP_LR_TIV_values) for threshold_key, TP_LR_TIV_values in TP_LR_TIV_froc_per_patient_all_CV.items()}
            total_TP_LR_M_per_threshold = {threshold_key: np.sum(TP_LR_M_values) for threshold_key, TP_LR_M_values in TP_LR_M_froc_per_patient_all_CV.items()}
            total_TP_ven_wash_per_threshold = {threshold_key: np.sum(TP_ven_wash_values) for threshold_key, TP_ven_wash_values in TP_ven_wash_froc_per_patient_all_CV.items()}
            total_TP_del_wash_per_threshold = {threshold_key: np.sum(TP_del_wash_values) for threshold_key, TP_del_wash_values in TP_del_wash_froc_per_patient_all_CV.items()}
            total_TP_ven_caps_per_threshold = {threshold_key: np.sum(TP_ven_caps_values) for threshold_key, TP_ven_caps_values in TP_ven_caps_froc_per_patient_all_CV.items()}
            total_TP_del_caps_per_threshold = {threshold_key: np.sum(TP_del_caps_values) for threshold_key, TP_del_caps_values in TP_del_caps_froc_per_patient_all_CV.items()}
            total_TP_hyper_art_per_threshold = {threshold_key: np.sum(TP_hyper_art_values) for threshold_key, TP_hyper_art_values in TP_hyper_art_froc_per_patient_all_CV.items()}

            lowest_threshold = min(FROC_metrics.keys())
            lesions_total = np.sum([np.sum([len(TP_list) for TP_list in FROC_metrics[lowest_threshold]['TP_list']]) for FROC_metrics in FROC_metrics_per_fold])
            LR_5_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == '5']) for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
            LR_4_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == '4']) for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
            LR_3_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == '3']) for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
            LR_TIV_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'TIV']) for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
            LR_M_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'M']) for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
            ven_wash_total = np.sum([np.sum([np.sum([1 for ven_wash in ven_wash_list if ven_wash == 1]) for ven_wash_list in FROC_metrics[lowest_threshold]['wash_ven_list']]) for FROC_metrics in FROC_metrics_per_fold])
            del_wash_total = np.sum([np.sum([np.sum([1 for del_wash in del_wash_list if del_wash == 1]) for del_wash_list in FROC_metrics[lowest_threshold]['wash_del_list']]) for FROC_metrics in FROC_metrics_per_fold])
            ven_caps_total = np.sum([np.sum([np.sum([1 for ven_caps in ven_caps_list if ven_caps == 1]) for ven_caps_list in FROC_metrics[lowest_threshold]['caps_ven_list']]) for FROC_metrics in FROC_metrics_per_fold])
            del_caps_total = np.sum([np.sum([np.sum([1 for del_caps in del_caps_list if del_caps == 1]) for del_caps_list in FROC_metrics[lowest_threshold]['caps_del_list']]) for FROC_metrics in FROC_metrics_per_fold])
            hyper_art_total = np.sum([np.sum([np.sum([1 for hyper_art in hyper_art_list if hyper_art == 1]) for hyper_art_list in FROC_metrics[lowest_threshold]['hyper_art_list']]) for FROC_metrics in FROC_metrics_per_fold])

            
            LR_5_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == '5']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]
            LR_4_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == '4']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]
            LR_3_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == '3']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]
            LR_TIV_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'TIV']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]
            LR_M_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'M']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[lowest_threshold]['LIRADS_list']]
            ven_wash_total_per_patient = [np.sum([1 for ven_wash in ven_wash_list if ven_wash == 1]) for FROC_metrics in FROC_metrics_per_fold for ven_wash_list in FROC_metrics[lowest_threshold]['wash_ven_list']]
            del_wash_total_per_patient = [np.sum([1 for del_wash in del_wash_list if del_wash == 1]) for FROC_metrics in FROC_metrics_per_fold for del_wash_list in FROC_metrics[lowest_threshold]['wash_del_list']]
            ven_caps_total_per_patient = [np.sum([1 for ven_caps in ven_caps_list if ven_caps == 1]) for FROC_metrics in FROC_metrics_per_fold for ven_caps_list in FROC_metrics[lowest_threshold]['caps_ven_list']]
            del_caps_total_per_patient = [np.sum([1 for del_caps in del_caps_list if del_caps == 1]) for FROC_metrics in FROC_metrics_per_fold for del_caps_list in FROC_metrics[lowest_threshold]['caps_del_list']]
            hyper_art_total_per_patient = [np.sum([1 for hyper_art in hyper_art_list if hyper_art == 1]) for FROC_metrics in FROC_metrics_per_fold for hyper_art_list in FROC_metrics[lowest_threshold]['hyper_art_list']]
            recall_LR_5_per_patient = {key_threshold: [float(TP_LR_5_froc_per_patient_all_CV[key_threshold][patient_idx]/LR_5_total_per_patient[patient_idx])
                                        for patient_idx in range(len(LR_5_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_LR_4_per_patient = {key_threshold: [float(TP_LR_4_froc_per_patient_all_CV[key_threshold][patient_idx]/LR_4_total_per_patient[patient_idx])
                                        for patient_idx in range(len(LR_4_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_LR_3_per_patient = {key_threshold: [float(TP_LR_3_froc_per_patient_all_CV[key_threshold][patient_idx]/LR_3_total_per_patient[patient_idx])
                                        for patient_idx in range(len(LR_3_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_LR_TIV_per_patient = {key_threshold: [float(TP_LR_TIV_froc_per_patient_all_CV[key_threshold][patient_idx]/LR_TIV_total_per_patient[patient_idx])
                                        for patient_idx in range(len(LR_TIV_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_LR_M_per_patient = {key_threshold: [float(TP_LR_M_froc_per_patient_all_CV[key_threshold][patient_idx]/LR_M_total_per_patient[patient_idx])
                                        for patient_idx in range(len(LR_M_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_ven_wash_per_patient = {key_threshold: [float(TP_ven_wash_froc_per_patient_all_CV[key_threshold][patient_idx]/ven_wash_total_per_patient[patient_idx])
                                        for patient_idx in range(len(ven_wash_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_del_wash_per_patient = {key_threshold: [float(TP_del_wash_froc_per_patient_all_CV[key_threshold][patient_idx]/del_wash_total_per_patient[patient_idx])
                                        for patient_idx in range(len(del_wash_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_ven_caps_per_patient = {key_threshold: [float(TP_ven_caps_froc_per_patient_all_CV[key_threshold][patient_idx]/ven_caps_total_per_patient[patient_idx])
                                        for patient_idx in range(len(ven_caps_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_del_caps_per_patient = {key_threshold: [float(TP_del_caps_froc_per_patient_all_CV[key_threshold][patient_idx]/del_caps_total_per_patient[patient_idx])
                                        for patient_idx in range(len(del_caps_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}
            recall_hyper_art_per_patient = {key_threshold: [float(TP_hyper_art_froc_per_patient_all_CV[key_threshold][patient_idx]/hyper_art_total_per_patient[patient_idx])
                                        for patient_idx in range(len(hyper_art_total_per_patient))] for key_threshold in total_lesion_per_patient_all_CV.keys()}

            recall_per_threshold = {threshold: float(TP / lesions_total) for threshold, TP in total_TP_per_threshold.items()}
            recall_per_patient = {key_threshold: [float(TP_froc_per_patient_all_CV[key_threshold][patient_idx]/total_lesion_per_patient_all_CV[key_threshold][patient_idx]) 
                                  for patient_idx in range(len(TP_froc_per_patient_all_CV[key_threshold]))] for key_threshold in TP_froc_per_patient_all_CV.keys()}
            recall_LR_5_per_threshold = {threshold: float(TP_LR_5 / LR_5_total) for threshold, TP_LR_5 in total_TP_LR_5_per_threshold.items()}
            recall_LR_4_per_threshold = {threshold: float(TP_LR_4 / LR_4_total) for threshold, TP_LR_4 in total_TP_LR_4_per_threshold.items()}
            recall_LR_3_per_threshold = {threshold: float(TP_LR_3 / LR_3_total) for threshold, TP_LR_3 in total_TP_LR_3_per_threshold.items()}
            recall_LR_TIV_per_threshold = {threshold: float(TP_LR_TIV / LR_TIV_total) for threshold, TP_LR_TIV in total_TP_LR_TIV_per_threshold.items()}
            recall_LR_M_per_threshold = {threshold: float(TP_LR_M / LR_M_total) for threshold, TP_LR_M in total_TP_LR_M_per_threshold.items()}
            recall_ven_wash_per_threshold = {threshold: float(TP_ven_wash / ven_wash_total) for threshold, TP_ven_wash in total_TP_ven_wash_per_threshold.items()}
            recall_del_wash_per_threshold = {threshold: float(TP_del_wash / del_wash_total) for threshold, TP_del_wash in total_TP_del_wash_per_threshold.items()}
            recall_ven_caps_per_threshold = {threshold: float(TP_ven_caps / ven_caps_total) for threshold, TP_ven_caps in total_TP_ven_caps_per_threshold.items()}
            recall_del_caps_per_threshold = {threshold: float(TP_del_caps / del_caps_total) for threshold, TP_del_caps in total_TP_del_caps_per_threshold.items()}
            recall_hyper_art_per_threshold = {threshold: float(TP_hyper_art / hyper_art_total) for threshold, TP_hyper_art in total_TP_hyper_art_per_threshold.items()}
            all_patients = [fold[lowest_threshold]['patients'] for fold in FROC_metrics_per_fold]


            saving_name = os.path.join(plot_saving_path, 'FROC_valid_metrics_size_{}.json'.format(tumor_size))

            FROC_metrics_per_fold_str = str(FROC_metrics_per_fold)
            with open(saving_name, 'w') as f:
                json.dump({ 'patients_list': all_patients,
                            'TP_per_threshold': total_TP_per_threshold,
                            'FP_per_threshold': total_FP_per_threshold,
                            'FP_per_patient': FP_froc_per_patient_all_CV,
                            'FN_per_threshold': total_FN_per_threshold,
                            'TN_per_threshold': total_TN_per_threshold,
                            'Dice_per_patient': patient_dice_all_CV,
                            'mean_FP_per_threshold': mean_FP_per_patient_all_CV,
                            'recall_per_patient': recall_per_patient,
                            'recall_LR_5_per_patient': recall_LR_5_per_patient,
                            'recall_LR_4_per_patient': recall_LR_4_per_patient,
                            'recall_LR_3_per_patient': recall_LR_3_per_patient,
                            'recall_LR_TIV_per_patient': recall_LR_TIV_per_patient,
                            'recall_LR_M_per_patient': recall_LR_M_per_patient,
                            'recall_ven_wash_per_patient': recall_ven_wash_per_patient,
                            'recall_del_wash_per_patient': recall_del_wash_per_patient,
                            'recall_ven_caps_per_patient': recall_ven_caps_per_patient,
                            'recall_del_caps_per_patient': recall_del_caps_per_patient,
                            'recall_hyper_art_per_patient': recall_hyper_art_per_patient,
                            'recall_per_threshold': recall_per_threshold,
                            'recall_LR_5': recall_LR_5_per_threshold,
                            'recall_LR_4': recall_LR_4_per_threshold,
                            'recall_LR_3': recall_LR_3_per_threshold,
                            'recall_LR_TIV': recall_LR_TIV_per_threshold,
                            'recall_LR_M': recall_LR_M_per_threshold,
                            'recall_ven_wash': recall_ven_wash_per_threshold,
                            'recall_del_wash': recall_del_wash_per_threshold,
                            'recall_ven_caps': recall_ven_caps_per_threshold,
                            'recall_del_caps': recall_del_caps_per_threshold,
                            'recall_hyper_art': recall_hyper_art_per_threshold,
                            'FROC_metrics_per_fold_str': FROC_metrics_per_fold_str,
                            'total_lesion_per_patient': total_lesion_per_patient_all_CV,
                            }, f, indent=4)

        else:
            mean_patient_dice_per_fold = [np.mean(metrics_list) for metrics_list in patient_tumor_dice_per_fold]
            std_patient_dice_per_fold = [np.std(metrics_list) for metrics_list in patient_tumor_dice_per_fold]

            full_patient_dice_list = [subitem for sublist in patient_tumor_dice_per_fold for item in sublist for subitem in item]
            full_tumor_dice_list = [subitem for sublist in tumor_wise_dice_per_fold for item in sublist for subitem in item]
            mean_tumor_dice = np.mean(full_tumor_dice_list)
            std_tumor_dice = np.std(full_tumor_dice_list)
            mean_patient_dice = np.mean(full_patient_dice_list)
            std_patient_dice = np.std(full_patient_dice_list)

            TP_per_patient_per_fold_flat = [[item for sublist in TP_list for item in sublist] for TP_list in TP_per_patient_per_fold]
            FN_per_patient_per_fold_flat = [[item for sublist in FN_list for item in sublist] for FN_list in FN_per_patient_per_fold]
            TN_per_patient_per_fold_flat = [[item for sublist in TN_list for item in sublist] for TN_list in TN_per_patient_per_fold]

            TP_per_fold = [np.sum(TP_list) if isinstance(np.sum(TP_list), list) else [np.sum(TP_list)] for TP_list in TP_per_patient_per_fold_flat]
            FN_per_fold = [np.sum(FN_list) if isinstance(np.sum(FN_list), list) else [np.sum(FN_list)] for FN_list in FN_per_patient_per_fold_flat]
            FP_per_fold = [FP_list for FP_list in FP_per_patient_per_fold]
            TN_per_fold = [np.sum(TN_list) if isinstance(np.sum(TN_list), list) else [np.sum(TN_list)] for TN_list in TN_per_patient_per_fold_flat]
            LIRADS_list_per_fold = [[LIRAD for patient in LIRADS_fold for LIRAD in patient] for LIRADS_fold in LIRADS_per_fold]

            recall_per_fold = [np.sum(np.sum(TP_list)) / (np.sum(np.sum(TP_list)) + np.sum(np.sum(FN_list))) for TP_list, FN_list in zip(TP_per_fold, FN_per_fold)]
            precision_per_fold = [np.sum(np.sum(TP_list)) / (np.sum(np.sum(TP_list)) + np.sum(FP_list)) for TP_list, FP_list in zip(TP_per_fold, FP_per_fold)]
            f2_score_per_fold = [5 * (precision * recall) / (4 * precision + recall) for recall, precision in zip(recall_per_fold, precision_per_fold)]
            f1_score_per_fold = [2 * (precision * recall) / (precision + recall) for recall, precision in zip(recall_per_fold, precision_per_fold)]
            specificity_per_fold = [np.sum(np.sum(TN_list)) / (np.sum(np.sum(TN_list)) + np.sum(np.sum(FP_list))) for TN_list, FP_list in zip(TN_per_fold, FP_per_fold)]


            TP_sum_per_fold = [np.sum(np.sum(TP_list)) for TP_list in TP_per_patient_per_fold_flat]
            FN_sum_per_fold = [np.sum(np.sum(FN_list)) for FN_list in FN_per_patient_per_fold_flat]
            FP_sum_per_fold = [np.sum(FP_list) for FP_list in FP_per_patient_per_fold]
            tumor_per_fold = [tp + fn for (tp, fn) in zip(TP_sum_per_fold, FN_sum_per_fold)]
            print('Tumors per fold: ', tumor_per_fold)
            print('Patients per fold: ', [len(patient_tumors) for patient_tumors in patients_per_fold])


            # Mean and std tumor dice per patient
            x = np.arange(len(mean_patient_dice_per_fold))
            plt.bar(x, mean_patient_dice_per_fold, yerr=std_patient_dice_per_fold, capsize=5)
            plt.xlabel('Fold')
            plt.ylim(0, 1)
            plt.title('Dice per patient')
            plt.ylabel('Tumor Dice')
            for i, (mean, std) in enumerate(zip(mean_patient_dice_per_fold, std_patient_dice_per_fold)):
                plt.text(i, 0.01, f'{mean:.2f} ± {std:.2f}', ha='center')
            plt.hlines(np.mean(mean_patient_dice_per_fold), -1, 5, linestyles='dashed', colors='red')
            plt.xlim(-0.6, 4.6)
            plt.savefig(plot_saving_path + "/dice_per_patient_tumor_size_{}.png".format(tumor_size))
            plt.close()

            # Recall, Precision, f2 score
            recall_mean = np.mean(recall_per_fold)
            recall_std = np.std(recall_per_fold)
            specificity_mean = np.mean(specificity_per_fold)
            specificity_std = np.std(specificity_per_fold)
            precision_mean = np.mean(precision_per_fold)
            precision_std = np.std(precision_per_fold)
            f2_score_mean = np.mean(f2_score_per_fold)
            f2_score_std = np.std(f2_score_per_fold)
            f1_score_mean = np.mean(f1_score_per_fold)
            f1_score_std = np.std(f1_score_per_fold)

            plt.figure(figsize=(10, 5))
            x_recall = [0]
            x_precision = [1]
            x_f2_score = [2]
            plt.bar(x_recall, recall_mean, yerr=recall_std, capsize=5, align='center', label='Recall')
            plt.bar(x_precision, precision_mean, yerr=precision_std, capsize=5, align='center', label='Precision')
            plt.bar(x_f2_score, f2_score_mean, yerr=f2_score_std, capsize=5, align='center', label='f2 Score')
            plt.xticks([x_recall[0], x_precision[0], x_f2_score[0]], ['Recall', 'Precision', 'f2 Score'])

            plt.text(0, 0.01, f'{recall_mean:.2f} ± {recall_std:.2f}', ha='center')
            plt.text(1, 0.01, f'{precision_mean:.2f} ± {precision_std:.2f}', ha='center')
            plt.text(2, 0.01, f'{f2_score_mean:.2f} ± {f2_score_std:.2f}', ha='center')

            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.title('Recall, Precision, F2 Score')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_saving_path + "/detection_tumor_size_{}.png".format(tumor_size))
            plt.close()
            print()

            tumor_nbr_per_fold = [TP + FN for (TP, FN) in zip(TP_sum_per_fold, FN_sum_per_fold)]
            print('Total number of tumors: ', tumor_nbr_per_fold)
            print('Mean patient dice', mean_patient_dice)
            print('Std patient dice', std_patient_dice)
            print('Mean tumor dice', mean_tumor_dice)
            print('Std tumor dice', std_tumor_dice)
            print('TP:', TP_sum_per_fold)
            print('FN:', FN_sum_per_fold)
            print('FP:', FP_sum_per_fold)
            print('recall', np.round(recall_mean, 2), '+/-', np.round(recall_std, 2))
            print('precision: ', np.round(precision_mean, 2), '+/-',
                  np.round(precision_std, 2))
            print('F2 score:', np.round(f2_score_mean, 2), '+/-', np.round(f2_score_std, 2))
            print('F1 score:', np.round(f1_score_mean, 2), '+/-', np.round(f1_score_std, 2))
            print('Specificity:', np.round(specificity_mean, 2), '+/-', np.round(specificity_std, 2))


            # Stratified performance
            LIRADSWashoutCapsuleEval(LIRADS_list_per_fold, TP_per_patient_per_fold_flat, FP_per_fold, ven_caps_per_fold, del_caps_per_fold, ven_wash_per_fold,
             del_wash_per_fold, hyper_art_per_fold, tumor_size, plot_saving_path)

            # TP per patient, create dict sub: TP, FN, FP
            TP_per_patient_per_fold_dict = {}
            for patient_fold, TP_fold in zip(patients_per_fold, TP_per_patient_per_fold):
                for patient, TP in zip(patient_fold, TP_fold):
                    TP_per_patient_per_fold_dict[patient] = TP

            # write metrics as json file to reload them later
            with open(os.path.join(plot_saving_path, 'metrics_size_{}.json'.format(tumor_size)), 'w') as f:
                json.dump({'mean_patient_dice_per_fold': mean_patient_dice,
                           'std_patient_dice_per_fold': std_patient_dice,
                           'mean_tumor_dice_per_fold': mean_tumor_dice,
                           'std_tumor_dice_per_fold': std_tumor_dice,
                           'recall': recall_mean,
                           'recall_std': recall_std,
                           'precision': precision_mean,
                           'precision_std': precision_std,
                           'f2_score': f2_score_mean,
                            'f2_score_std': f2_score_std,
                           'f1_score': f1_score_mean,
                            'f1_score_std': f1_score_std,
                           'specificity': specificity_mean,
                            'specificity_std': specificity_std,
                           'TP': [float(item) for item in TP_sum_per_fold],
                           'FN': [float(item) for item in FN_sum_per_fold],
                           'FP': [float(item) for item in FP_sum_per_fold],
                           'tumor_per_fold': [float(item) for item in tumor_per_fold],
                           'patients_per_fold': [len(patient_tumors) for patient_tumors in patients_per_fold],
                           'tumor_nbr_per_fold': [float(item) for item in tumor_nbr_per_fold],
                            'TP_per_patient_per_fold': TP_per_patient_per_fold_dict
                           }, f, indent=4)

            #write config
            with open(os.path.join(plot_saving_path, 'config.txt'), 'w') as f:
                json.dump(config, f, indent=4)

            # check tta
            if config["predict_test"]:
                if config['tta_num_examples'] != 0:
                    prob_map_output_dir = os.path.dirname(output_dir) + "/mean_prob_maps/prob_map_tta"
                else:
                    prob_map_output_dir = os.path.dirname(output_dir) + "/mean_prob_maps/prob_map"
                preds_output_dir = os.path.dirname(output_dir) + "/mean_prob_maps/preds"
                os.makedirs(prob_map_output_dir, exist_ok=True)
                os.makedirs(preds_output_dir, exist_ok=True)

                for patient in range(len(prob_map_per_fold[0])):
                    patient_prob_maps = []
                    for CV in range(len(prob_map_per_fold)):
                        prob_map = sitk.GetArrayFromImage(sitk.ReadImage(prob_map_per_fold[CV][patient]))
                        patient_prob_maps.append(prob_map)
                    mean_prob_map_ar = np.mean(patient_prob_maps, axis=0)
                    mean_prob_map = sitk.GetImageFromArray(mean_prob_map_ar)
                    mean_prob_map.CopyInformation(sitk.ReadImage(prob_map_per_fold[0][patient]))
                    sitk.WriteImage(mean_prob_map,
                                    prob_map_output_dir + '/' + os.path.basename(prob_map_per_fold[0][patient]))
                    binary_map = np.zeros_like(mean_prob_map_ar)
                    binary_map[mean_prob_map_ar > config["mean_prob_map_threshold"]] = 1
                    binary_map[mean_prob_map_ar <= config["mean_prob_map_threshold"]] = 0
                    binary_map = sitk.GetImageFromArray(binary_map)
                    binary_map.CopyInformation(sitk.ReadImage(prob_map_per_fold[0][patient]))
                    sitk.WriteImage(binary_map, preds_output_dir + '/' + os.path.basename(prob_map_per_fold[0][patient]))


                prob_maps = sorted([os.path.join(prob_map_output_dir, file) for file in os.listdir(prob_map_output_dir)])
                preds = sorted([os.path.join(preds_output_dir, file) for file in os.listdir(preds_output_dir)])
                gt = gt_per_fold[0]
                gt_multilabels = gt_multilabels_per_fold[0]
                liver_labels = liver_labels_per_fold[0]
                vein_labels = vein_labels_per_fold[0]
                images = images_per_fold[0]

                lesions_characteristics = pd.read_csv(data_dir + '/tumors_characteristics.csv')
                for tumor_size in config["min_tumor_size"]:
                    (TP_list, FP_list, FN_list, TN_list, patients,
                     patient_tumor_metrics_list, tumor_metrics_list, generated_neg_samples, LIRADS,
                     ven_wash, del_wash, ven_caps, del_caps, hyper_art
                     ) = eval_perf_fold(prob_maps, images, preds, gt, gt_multilabels, liver_labels, vein_labels, tumor_size,
                                        config, output_dir, lesions_characteristics)
                    LIRADS_flat = [item for sublist in LIRADS for item in sublist]
                    TP_list_flat = [item for sublist in TP_list for item in sublist]
                    FN_list_flat = [item for sublist in FN_list for item in sublist]
                    TN_list_flat = [item for sublist in TN_list for item in sublist]

                    LIRADS_per_fold = [LIRADS_flat]
                    TP_per_fold = [TP_list_flat]
                    FP_per_fold = [FP_list]
                    ven_caps_per_fold = [ven_caps]
                    del_caps_per_fold = [del_caps]
                    ven_wash_per_fold = [ven_wash]
                    del_wash_per_fold = [del_wash]
                    hyper_art_per_fold = [hyper_art]

                    LIRADSWashoutCapsuleEval(LIRADS_per_fold, TP_per_fold, FP_per_fold, ven_caps_per_fold, del_caps_per_fold,
                                             ven_wash_per_fold, del_wash_per_fold, hyper_art_per_fold, tumor_size, plot_saving_path, is_test=True)
                    mean_tumor_dice = np.mean(tumor_metrics_list)
                    std_tumor_dice = np.std(tumor_metrics_list)
                    mean_patient_dice = np.mean(patient_tumor_metrics_list)
                    std_patient_dice = np.std(patient_tumor_metrics_list)

                    recall = np.sum(np.sum(TP_list_flat)) / (np.sum(np.sum(TP_list_flat)) + np.sum(np.sum(FN_list_flat)))
                    precision = np.sum(np.sum(TP_list_flat)) / (np.sum(np.sum(TP_list_flat)) + np.sum(FP_list))
                    f2_score = 5 * (precision * recall) / (4 * precision + recall)
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    specificity = np.sum(np.sum(TN_list_flat)) / (np.sum(np.sum(TN_list_flat)) + np.sum(np.sum(FP_list)))
                    print('Mean tumor dice: ', mean_tumor_dice)
                    print('Std tumor dice: ', std_tumor_dice)
                    print('Mean patient dice: ', mean_patient_dice)
                    print('Std patient dice: ', std_patient_dice)
                    print('Recall: ', recall)
                    print('Precision: ', precision)
                    print('F2 score: ', f2_score)
                    print('F1 score: ', f1_score)
                    print('Specificity: ', specificity)

                    with open(os.path.join(plot_saving_path, 'mean_metrics_size_{}_mean_map_threshold_{}.json'.format(tumor_size, config["mean_prob_map_threshold"])), 'w') as f:
                        json.dump({'mean_patient_dice': mean_patient_dice,
                                   'std_patient_dice': std_patient_dice,
                                   'mean_tumor_dice': mean_tumor_dice,
                                   'std_tumor_dice': std_tumor_dice,
                                   'recall': recall,
                                   'precision': precision,
                                   'f2_score': f2_score,
                                   'f1_score': f1_score,
                                   'specificity': specificity
                                   }, f, indent=4)



if __name__ == "__main__":
    train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'training/tumor_segmentation/unet'))
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))

    # experiment = 'tumor_segmentation/tumor_segmentation_pretraining/2. Unet_Tversky.yaml'
    experiment = 'tumor_segmentation/tumor_segmentation_finetuning/3. Unet_pretrained_Tversky_loss.yaml.yaml'

    eval_config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_finetuning/evaluation.yaml')
    FROC_thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    train_config_path = os.path.join(main_dir, 'training_experiments', experiment)
    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load(file)
    
    data_dir = os.path.join(main_dir, train_config["dataset"])

    wandb_run_names = [train_config["saving_name_wandb_fold_{}".format(fold)] for fold in range(5)]
    wandb_run_names = ['wandb/' + run_name for run_name in wandb_run_names]
    output_dir = os.path.join("tumor_eval",  experiment[:-5], os.path.basename(data_dir))
    
    with open(eval_config_path, 'r') as file:
        eval_config = yaml.safe_load(file)

    merged_config = {**train_config, **eval_config}
    merged_config["output_dir"] = output_dir
    merged_config["wandb_run_names"] = wandb_run_names
    merged_config["FROC"] = True
    merged_config["predict_test"] = False
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
    
    merged_config['lesion_characteristics'] = os.path.join(os.path.dirname(data_dir), 'tumors_characteristics.csv')
    main(data_dir, train_dir, merged_config)