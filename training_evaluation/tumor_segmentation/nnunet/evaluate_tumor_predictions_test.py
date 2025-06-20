import os
import yaml
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
from joblib import Parallel, delayed
eval_dir = os.path.dirname(os.getcwd())
print(eval_dir)
sys.path.append(eval_dir)
from eval_utils import eval_perf_fold, LIRADSWashoutCapsuleEval


def evaluate_predictions(seg_dir, preds_dir, dataset_dir, FROC, tumor_size_limit, config, plot_saving_path=None):
    prob_maps_path = os.path.join(preds_dir, 'prob_maps')
    gt_path = os.path.join(preds_dir, 'ground_truth')
    images_path = os.path.join(dataset_dir, 'images_4D')
    liver_labels_path = os.path.join(dataset_dir, 'liver_labels')
    multi_gt_path = os.path.join(dataset_dir, 'multi_labels')
    
    prob_maps = sorted([os.path.join(prob_maps_path, file) for file in os.listdir(prob_maps_path) if not '_T' in file])
    gt = sorted([os.path.join(gt_path, file) for file in os.listdir(gt_path)])
    images = sorted([os.path.join(images_path, file) for file in os.listdir(images_path)])
    liver_labels = sorted([os.path.join(liver_labels_path, file) for file in os.listdir(liver_labels_path)])
    multi_gt = sorted([os.path.join(multi_gt_path, file) for file in os.listdir(multi_gt_path)])
    if config['vein_mask_criterion']:
        vein_labels = sorted([file.replace('labels', 'total_segmentator') for file in gt], key=lambda x: x[:-7])
    else:
        vein_labels = [[] for x in range(len(liver_labels))]


    lesions_characteristics = pd.read_csv(dataset_dir + '/tumors_characteristics.csv', sep=';')
    if lesions_characteristics.shape[1] == 1:
        lesions_characteristics = pd.read_csv(dataset_dir + '/tumors_characteristics.csv', sep=',')
    if FROC:
        FROC_metrics_dict = Parallel(n_jobs=20)(delayed(lambda threshold: (threshold, eval_perf_fold(prob_maps, images, prob_maps, gt, multi_gt, liver_labels, 
                                                                    vein_labels, tumor_size_limit, config, output_dir, lesions_characteristics, 
                                                                    threshold=threshold)))(threshold) for threshold in config["FROC_thresholds"])
        FROC_metrics_dict = dict(FROC_metrics_dict)
        
        TP_per_patient = [float(np.sum([float(np.sum(sub_list)) for sub_list in metric_dict['TP_list'] if len(sub_list) > 0])) for thresh_key, metric_dict in FROC_metrics_dict.items()]
        FN_per_patient = [float(np.sum([float(np.sum(sub_list)) for sub_list in metric_dict['FN_list'] if len(sub_list) > 0])) for thresh_key, metric_dict in FROC_metrics_dict.items()]
        FP_per_patient = [float(np.sum(np.sum(metric_dict['FP_list']))) for thresh_key, metric_dict in FROC_metrics_dict.items()]
        TN_per_patient = [np.sum([float(np.sum(sub_list)) for sub_list in metric_dict['TN_list'] if len(sub_list) > 0]) for thresh_key, metric_dict in FROC_metrics_dict.items()]
        recall_per_threshold = [float(np.divide(float(TP), float(TP) + float(FN), out=np.zeros_like(float(TP)),
                                        where=(float(TP) + float(FN)) != 0)) for TP, FN in zip(TP_per_patient, FN_per_patient)]

        mean_FP_per_patient = [np.mean(metric_dict['FP_list']) for thresh_key, metric_dict in FROC_metrics_dict.items()]

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
                                        tumor_size_limit,
                                        plot_saving_path + '/strat_perf_thresh' + str(thresh))

        plt.figure()
        plt.plot([0] + list(mean_FP_per_patient)[::-1], [0] + list(recall_per_threshold)[::-1], marker='o')
        plt.xlabel('Mean number of false Positives')
        plt.ylabel('Sensitivity')
        plt.ylim(0, 1)
        plt.title('Mean FROC Curve')
        plt.savefig(plot_saving_path + "/FROC_tumor_wise{}.png".format(tumor_size_limit))
        plt.close()

        healthy_subjects = ['sub-' + str(i).zfill(3) + '.nii.gz' for i in range(65, 76)] + ['sub-' + str(i).zfill(3) + '.nii.gz' for i in range(118, 300)]
        TP_patients = [float(sum([1 if np.sum(TP) > 0 else 0 for patient, TP in zip(metric_dict['patients'], metric_dict['TP_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
        FP_patients = [float(sum([1 if np.sum(FP) > 0 and patient in healthy_subjects else 0 for patient, FP in zip(metric_dict['patients'], metric_dict['FP_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
        FN_patients = [float(sum([1 if np.sum(FN) > 0 else 0 for patient, FN in zip(metric_dict['patients'], metric_dict['FN_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
        TN_patients = [float(sum([1 if np.sum(TN) > 0 and patient in healthy_subjects else 0 for patient, TN in zip(metric_dict['patients'], metric_dict['TN_list'])])) for threshold, metric_dict in FROC_metrics_dict.items()]
        patient_recall = [float(np.divide(float(TP), float(TP) + float(FN), out=np.zeros_like(float(TP)),
                                        where=(float(TP) + float(FN)) != 0)) for TP, FN in zip(TP_patients, FN_patients)]

        FROC_metrics_str = str(FROC_metrics_dict)
        with open(os.path.join(plot_saving_path, 'FROC_metrics_size_{}.json'.format(tumor_size_limit)), 'w') as f:
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
        threshold = 0
        FROC_metrics_dict = eval_perf_fold(threshold, eval_perf_fold(prob_maps, images, prob_maps, gt, multi_gt, liver_labels, 
                                                            vein_labels, tumor_size_limit, config, output_dir, lesions_characteristics, 
                                                            threshold=threshold))


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    train_dir = os.path.join(main_dir, 'training/tumor_segmentation/nnunet')
    # dataset_dir = os.path.join(main_dir, 'HCC_pre_ablation/derivatives/6_T1_dataset')
    dataset_dir = os.path.join(main_dir, 'HCC_Surveillance/derivatives/10_test_T1_dataset')

    eval_config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_finetuning/evaluation.yaml')
    with open(eval_config_path, 'r') as file:
        eval_config = yaml.safe_load(file)

    # HCC SUrveillance test set
    if os.path.basename(dataset_dir) == '10_test_T1_dataset':
        data_set = 'Dataset034_10_test_T1_dataset'
        labels_dir = main_dir + f'/training_evaluation/tumor_segmentation/nnunet/work_dir/nnUNet_raw_data_base/{data_set}/labelsTr'
        output_dir = main_dir + f'/training_evaluation/tumor_segmentation/nnunet/work_dir/nnUNet_raw_data_base/{data_set}/imagesTrPreds'
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]
        tumor_size_limit = [10]

        eval_config['FROC_thresholds'] = thresholds
        eval_config['vein_mask_criterion'] = True
        for size_ in tumor_size_limit:
            evaluate_predictions(labels_dir, output_dir, dataset_dir, FROC=True,
                                tumor_size_limit=size_, config=eval_config, plot_saving_path=output_dir)

    # pre-ablation test set all lesions
    if os.path.basename(dataset_dir) == '6_T1_dataset':
        labels_dir = os.getcwd() + '/work_dir/nnUNet_raw_data_base/Dataset033_6_T1_dataset/labelsTr'
        output_dir = os.getcwd() + '/work_dir/nnUNet_raw_data_base/Dataset033_6_T1_dataset/imagesTrPreds'
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]
        tumor_size_limit = [10]

        eval_config['FROC_thresholds'] = thresholds
        eval_config['vein_mask_criterion'] = True
        for size_ in tumor_size_limit:
            evaluate_predictions(labels_dir, output_dir, dataset_dir, FROC=True,
                                 tumor_size_limit=size_, config=eval_config, plot_saving_path=output_dir)
