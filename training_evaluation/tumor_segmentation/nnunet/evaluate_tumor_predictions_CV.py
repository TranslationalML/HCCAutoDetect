import os
import numpy as np
import json
import pandas as pd
import sys
import yaml
from joblib import Parallel, delayed
from utils import rename_patients

training_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'training/tumor_segmentation/nnunet'))
sys.path.append(training_dir)

eval_dir = os.path.dirname(os.getcwd())
print(eval_dir)
sys.path.append(eval_dir)
from eval_utils import eval_perf_fold



def evaluate_predictions(seg_dir, all_folds_preds_dir, dataset_dir, region_based, FROC, tumor_size_limit, config):
    config["saving_5CV_name"] = "tta{}_T{}_contrast_crit_{}_T{}_vein_crit_{}_T{}_prob_T{}".format(
    config["tta_num_examples"], config["tta_threshold"], config["contrast_criterion"], config["vein_mask_criterion"],
    config["vein_mask_threshold"], config["post_trans_threshold"])

    FROC_metrics_per_fold = []
    for fold in ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']:

        preds_fold_dir = os.path.join(all_folds_preds_dir, fold + '/validation')
        prob_maps_path = os.path.join(preds_fold_dir, 'prob_maps')
        gt_path = os.path.join(preds_fold_dir, 'ground_truth')
        images_path = os.path.join(dataset_dir, 'images_4D')
        liver_labels_path = os.path.join(dataset_dir, 'liver_labels')
        multi_gt_path = os.path.join(dataset_dir, 'multi_labels')
        
        prob_maps = sorted([os.path.join(prob_maps_path, file) for file in os.listdir(prob_maps_path) if not '_T0' in file])
        gt = sorted([os.path.join(gt_path, file) for file in os.listdir(gt_path) if file in os.listdir(prob_maps_path)])
        images = sorted([os.path.join(images_path, file) for file in os.listdir(images_path) if file in os.listdir(prob_maps_path)])
        liver_labels = sorted([os.path.join(liver_labels_path, file) for file in os.listdir(liver_labels_path) if file in os.listdir(prob_maps_path)])
        multi_gt = sorted([os.path.join(multi_gt_path, file) for file in os.listdir(multi_gt_path) if file in os.listdir(prob_maps_path)])
        if config['vein_mask_criterion']:
            vein_labels = sorted([file.replace('ground_truth', 'total_segmentator') for file in gt], key=lambda x: x[:-7])
        else:
            vein_labels = [[] for x in range(len(liver_labels))]

        lesions_characteristics = pd.read_csv(dataset_dir + '/tumors_characteristics.csv')
        if FROC:
            FROC_metrics_dict = Parallel(n_jobs=20)(delayed(lambda threshold: (threshold, eval_perf_fold(prob_maps, images, prob_maps, gt, multi_gt, liver_labels, 
                                                                        vein_labels, tumor_size_limit, config, output_dir, lesions_characteristics, 
                                                                        threshold=threshold)))(threshold) for threshold in config["FROC_thresholds"])
            FROC_metrics_dict = dict(FROC_metrics_dict)
            FROC_metrics_per_fold.append(FROC_metrics_dict)
            
        else:
            threshold = 0
            FROC_metrics_dict = eval_perf_fold(prob_maps, images, prob_maps, gt, multi_gt, liver_labels, 
                                                                        vein_labels, tumor_size_limit, config, output_dir, lesions_characteristics, 
                                                                        threshold=threshold)
            FROC_metrics_per_fold.append(FROC_metrics_dict)

    plot_saving_path = (os.path.dirname(config['output_dir']) + "/plots_tta_{}_thresh_{}_".format(config["tta_num_examples"], config['tta_threshold'])
                    + "post_thresh_" + str(config["post_trans_threshold"]) + "_" + config["saving_5CV_name"])
    os.makedirs(plot_saving_path, exist_ok=True)


    # mean FROC CV
    TP_froc_per_patient_all_fold_0 = [[np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics_per_fold[0].items()]
    TP_froc_per_patient_all_fold_1 = [[np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics_per_fold[1].items()]
    TP_froc_per_patient_all_fold_2 = [[np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics_per_fold[2].items()]
    TP_froc_per_patient_all_fold_3 = [[np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics_per_fold[3].items()]
    TP_froc_per_patient_all_fold_4 = [[np.sum(TP_list) for TP_list in FROC_list['TP_list']] for threshold, FROC_list in FROC_metrics_per_fold[4].items()]
    TP_froc_per_patient_all = np.concatenate([TP_froc_per_patient_all_fold_0, TP_froc_per_patient_all_fold_1, TP_froc_per_patient_all_fold_2, TP_froc_per_patient_all_fold_3, TP_froc_per_patient_all_fold_4], axis=1)
    
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


    lesions_total = np.sum([np.sum([len(TP_list) for TP_list in FROC_metrics[0.01]['TP_list']]) for FROC_metrics in FROC_metrics_per_fold])
    LR_5_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == '5']) for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
    LR_4_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == '4']) for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
    LR_3_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == '3']) for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
    LR_TIV_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'TIV']) for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
    LR_M_total = np.sum([np.sum([np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'M']) for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]) for FROC_metrics in FROC_metrics_per_fold])
    ven_wash_total = np.sum([np.sum([np.sum([1 for ven_wash in ven_wash_list if ven_wash == 1]) for ven_wash_list in FROC_metrics[0.01]['wash_ven_list']]) for FROC_metrics in FROC_metrics_per_fold])
    del_wash_total = np.sum([np.sum([np.sum([1 for del_wash in del_wash_list if del_wash == 1]) for del_wash_list in FROC_metrics[0.01]['wash_del_list']]) for FROC_metrics in FROC_metrics_per_fold])
    ven_caps_total = np.sum([np.sum([np.sum([1 for ven_caps in ven_caps_list if ven_caps == 1]) for ven_caps_list in FROC_metrics[0.01]['caps_ven_list']]) for FROC_metrics in FROC_metrics_per_fold])
    del_caps_total = np.sum([np.sum([np.sum([1 for del_caps in del_caps_list if del_caps == 1]) for del_caps_list in FROC_metrics[0.01]['caps_del_list']]) for FROC_metrics in FROC_metrics_per_fold])
    hyper_art_total = np.sum([np.sum([np.sum([1 for hyper_art in hyper_art_list if hyper_art == 1]) for hyper_art_list in FROC_metrics[0.01]['hyper_art_list']]) for FROC_metrics in FROC_metrics_per_fold])

    
    LR_5_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == '5']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]
    LR_4_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == '4']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]
    LR_3_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == '3']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]
    LR_TIV_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'TIV']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]
    LR_M_total_per_patient = [np.sum([1 for LIRAD in LIRADS_list if LIRAD == 'M']) for FROC_metrics in FROC_metrics_per_fold for LIRADS_list in FROC_metrics[0.01]['LIRADS_list']]
    ven_wash_total_per_patient = [np.sum([1 for ven_wash in ven_wash_list if ven_wash == 1]) for FROC_metrics in FROC_metrics_per_fold for ven_wash_list in FROC_metrics[0.01]['wash_ven_list']]
    del_wash_total_per_patient = [np.sum([1 for del_wash in del_wash_list if del_wash == 1]) for FROC_metrics in FROC_metrics_per_fold for del_wash_list in FROC_metrics[0.01]['wash_del_list']]
    ven_caps_total_per_patient = [np.sum([1 for ven_caps in ven_caps_list if ven_caps == 1]) for FROC_metrics in FROC_metrics_per_fold for ven_caps_list in FROC_metrics[0.01]['caps_ven_list']]
    del_caps_total_per_patient = [np.sum([1 for del_caps in del_caps_list if del_caps == 1]) for FROC_metrics in FROC_metrics_per_fold for del_caps_list in FROC_metrics[0.01]['caps_del_list']]
    hyper_art_total_per_patient = [np.sum([1 for hyper_art in hyper_art_list if hyper_art == 1]) for FROC_metrics in FROC_metrics_per_fold for hyper_art_list in FROC_metrics[0.01]['hyper_art_list']]
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

    all_patients = FROC_metrics_per_fold[0][0.01]['patients'] + FROC_metrics_per_fold[1][0.01]['patients'] + FROC_metrics_per_fold[2][0.01]['patients'] + FROC_metrics_per_fold[3][0.01]['patients'] + FROC_metrics_per_fold[4][0.01]['patients']

    saving_name = os.path.join(plot_saving_path, 'FROC_valid_metrics_size_{}.json'.format(tumor_size_limit))
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
                    }, f, indent=4)



if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    training_dir = os.path.join(main_dir, "training/tumor_segmentation/nnunet")

    print('All tumors model')
    seg_dir = training_dir + '/work_dir/nnUNet_raw_data_base/Dataset030_HCC_Surveillance_patients/labelsTr'
    preds_dir = training_dir + '/work_dir/nnUNet_results/Dataset030_HCC_Surveillance_patients/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres'
    data_dir = os.path.join(main_dir, 'HCC_Surveillance/derivatives/10_T1_dataset')
    thresholds_list = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]

    for fold in range(5):
        if not os.path.exists(os.path.join(preds_dir, 'fold_{}'.format(fold), 'validation', 'prob_maps')):
            preds_dir_fold = os.path.join(preds_dir, 'fold_{}'.format(fold), 'validation')
            data_list_json = os.path.dirname(seg_dir) + '/datalist.json'
            rename_patients(preds_dir_fold, seg_dir, data_list_json)


    output_dir = os.path.join(preds_dir, 'FROC_metrics')
    eval_config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_finetuning/evaluation.yaml')
    with open(eval_config_path, 'r') as file:
        eval_config = yaml.safe_load(file)
    eval_config['FROC_thresholds'] = thresholds_list
    eval_config['output_dir'] = output_dir
    evaluate_predictions(seg_dir, preds_dir, data_dir, region_based=False, FROC=True, tumor_size_limit=10, config=eval_config)
