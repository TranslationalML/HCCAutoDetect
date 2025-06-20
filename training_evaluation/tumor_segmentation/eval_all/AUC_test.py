import os
import sys
import yaml
import numpy as np
import json
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import ast
from utils import run_permutation_test, AUC_FROC_curve, get_patient_metrics, sort_data_for_AUC, compute_IC_95, ROC_patient_wise, compute_IC_95_patientwise


def load_metrics(eval_dir, config, size, art_contrast, vein_crit, mean_prob_map_threshold, patient_list):
    if 'nnunet' in config['output_dir']:
        FROC_metrics_dict = json.load(open(os.path.join(config['output_dir'], f'FROC_metrics_size_{size}.json'), 'r'))
    else:
        output_folder_dir = eval_dir + '/' + config["output_dir"] + f'/results_mean_prob_map_thresh_{mean_prob_map_threshold}_tta5_T0.01_contrast_crit_{art_contrast}_T0.1_vein_crit_{vein_crit}_T0_prob_T0.5'
        if art_contrast:
            file_path = os.path.join(output_folder_dir, f'FROC_metrics_size_{size}_{art_contrast}_art.json')
            FROC_metrics_dict = json.load(open(file_path, 'r'))
        else:
            file_path = os.path.join(output_folder_dir, f'FROC_metrics_size_{size}.json')
            FROC_metrics_dict = json.load(open(file_path, 'r'))
    FROC_metrics_dict['FROC_metrics'] = re.sub(r'\bnan\b', 'None', FROC_metrics_dict['FROC_metrics'])
    FROC_metrics_init = ast.literal_eval(FROC_metrics_dict['FROC_metrics'])
    FROC_metrics_dict = get_patient_metrics(FROC_metrics_init, patient_list)
    return FROC_metrics_dict


def run_AUC_test(data_dir, patient_type, json_path, nbr_of_x_points, size, art_contrast, vein_crit, permutation_nbr, 
                 bootstrap_nbr, thresh_list, permutation_calculation, compute_patient_metrics):
    mean_prob_map_threshold = 0.5
    if patient_type is not None:
        with open(json_path, 'r') as f:
            patients_with_diff_lesion_overlap = json.load(f)
        patient_list = patients_with_diff_lesion_overlap[patient_type]
    else:
        patient_list = None

    figsize=(15, 8)
    colors = ['g', 'b', 'gold']
    proxy_list = [Patch(color=colors[0], label='Model 1: nnU-Net', alpha=0.3),
                  Patch(color=colors[1], label='Model 2: U-Net (Tversky)', alpha=0.3),
                  Patch(color=colors[2], label='Model 3: U-Net (Pre-training + Tversky)', alpha=0.3)]

    
    main_saving_path = 'FROC_evaluation'
    eval_dir = main_dir + '/training_evaluation/tumor_segmentation/unet'
    os.makedirs(os.path.join(main_dir, 'training_evaluation', 'tumor_segmentation', 'eval_all', main_saving_path), exist_ok=True)
    experiment_list = ['tumor_segmentation/tumor_segmentation_pretraining/2. Unet_Tversky.yaml',
                       'tumor_segmentation/tumor_segmentation_finetuning/3. Unet_pretrained_Tversky_loss.yaml']
    
    metrics_per_experiment = {}
    for experiment in experiment_list:
        train_config_path = os.path.join(main_dir, 'training_experiments', experiment)
        with open(train_config_path, 'r') as file:
            train_config = yaml.safe_load(file)

        wandb_run_names = [train_config["saving_name_wandb_fold_{}".format(fold)] for fold in range(5)]
        wandb_run_names = ['wandb/' + run_name for run_name in wandb_run_names]

        if len(sys.argv) > 1:
            fold = int(sys.argv[1])
        else:
            fold = 0
        
        FROC_metrics_dict_per_fold = []
        for fold in range(0, 1, 1):
            output_dir = os.path.join("tumor_eval",  experiment[:-5], os.path.basename(data_dir) + '_' + str(fold))
            eval_config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_finetuning/evaluation.yaml')
            with open(eval_config_path, 'r') as file:
                eval_config = yaml.safe_load(file)

            merged_config = {**train_config, **eval_config}
            merged_config["output_dir"] = output_dir
            merged_config["wandb_run_names"] = wandb_run_names
            merged_config["predict_test"] = True
            merged_config["FROC"] = True
            print(output_dir) 
            os.makedirs(output_dir, exist_ok=True)
            
            merged_config['lesion_characteristics'] = os.path.join(os.path.dirname(data_dir), 'tumors_characteristics.csv')
            FROC_metrics_dict = load_metrics(eval_dir, merged_config, size, art_contrast, vein_crit, mean_prob_map_threshold, patient_list)
            FROC_metrics_dict_per_fold.append(FROC_metrics_dict)
        metrics_per_experiment[experiment] = FROC_metrics_dict_per_fold
    with open(eval_config_path, 'r') as file:
        eval_config = yaml.safe_load(file)
    if '10_test_T1_dataset' in data_dir:
        eval_config["output_dir"] = os.path.join(main_dir, 'training_evaluation/tumor_segmentation/nnunet/work_dir/nnUNet_raw_data_base/Dataset034_10_test_T1_dataset/imagesTrPreds')
    if '6_T1_dataset' in data_dir:
        eval_config["output_dir"] = os.path.join(main_dir, 'training_evaluation/tumor_segmentation/nnunet/work_dir/nnUNet_raw_data_base/Dataset033_6_T1_dataset/imagesTrPreds')
    nnunet_FROC_metrics_dict = load_metrics(None, eval_config, size=0, art_contrast=None, mean_prob_map_threshold=None, patient_list=patient_list, vein_crit=False)

    AUC_per_experiment = {}
    AUC_IC_per_experiment = {}
    TPR_patient_wise_score = {}
    FPR_patient_wise_score = {}
    TPR_IC_95 = {}
    FPR_IC_95 = {}
    ROC_AUC_IC_per_experiment = {}
    max_TP_list_per_model = {}
    mean_FP_per_threshold = [nnunet_FROC_metrics_dict['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
    recall_per_threshold = [nnunet_FROC_metrics_dict['lesions_recall'][threshold] for threshold in thresh_list]
    if '10_test_screening' in data_dir and patient_type is None and compute_patient_metrics:
        TPR_patient_wise_score['0. nnunet'], FPR_patient_wise_score['0. nnunet'] = ROC_patient_wise(nnunet_FROC_metrics_dict, thresh_list)
        TPR_IC_95_lower, TPR_IC_95_upper, FPR_IC_95_interp, mean_AUC, AUC_lower_bound, AUC_upper_bound, bootstrap_TPR = compute_IC_95_patientwise(nnunet_FROC_metrics_dict, thresh_list, bootstrap_nbr, patient_list)
        TPR_IC_95['0. nnunet'] = [TPR_IC_95_lower, TPR_IC_95_upper, bootstrap_TPR]
        FPR_IC_95['0. nnunet'] = FPR_IC_95_interp
        ROC_AUC_IC_per_experiment['0. nnunet'] = [mean_AUC.round(2), AUC_lower_bound.round(2), AUC_upper_bound.round(2)]
 
    recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
    (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper, bootstrap_sensitivity) = compute_IC_95(nnunet_FROC_metrics_dict, 'lesions_recall', 
                                                                                                                        thresh_list, bootstrap_nbr, patient_list, 
                                                                                                                        nbr_of_x_points)
    df_nnunet = pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                              'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                              'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[0].get_label()]* len(mean_FP_per_threshold)]})
    df_list = [df_nnunet]
    max_TP_list_per_model['0. nnunet'] = {key: value for key, value in zip(nnunet_FROC_metrics_dict['patient_name'], nnunet_FROC_metrics_dict['TP_per_patient'][thresh_list[0]])}
    AUC_per_experiment['0. nnunet'] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold)
    AUC_IC_per_experiment['0. nnunet'] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
    for idx, (experiment, metric) in enumerate(metrics_per_experiment.items()):
        for metric_fold in metric:
            mean_FP_per_threshold = [metric_fold['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
            recall_per_threshold = [metric_fold['lesions_recall'][threshold] for threshold in thresh_list]

            recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
            (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper, bootstrap_sensitivity) = compute_IC_95(metric_fold, 'lesions_recall', thresh_list, 
                                                                                                       bootstrap_nbr, patient_list, nbr_of_x_points)

            max_TP_list_per_model[str(idx + 1) + '. ' + experiment] = {key: value for key, value in zip(metric_fold['patient_name'], metric_fold['TP_per_patient'][thresh_list[0]])}
            AUC_per_experiment[str(idx + 1) + '. ' + experiment] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold).round(2)
            AUC_IC_per_experiment[str(idx + 1) + '. ' + experiment] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
            if '10_test_screening' in data_dir and patient_type is None and compute_patient_metrics:
                TPR_IC_95_lower, TPR_IC_95_upper, FPR_IC_95_interp, mean_AUC, AUC_lower_bound, AUC_upper_bound, bootstrap_TPR = compute_IC_95_patientwise(metric_fold, thresh_list, bootstrap_nbr, patient_list)
                TPR_IC_95[str(idx + 1) + '. ' + experiment] = [TPR_IC_95_lower, TPR_IC_95_upper, bootstrap_TPR]
                FPR_IC_95[str(idx + 1) + '. ' + experiment] = FPR_IC_95_interp
                ROC_AUC_IC_per_experiment[str(idx + 1) + '. ' + experiment] = [mean_AUC.round(2), AUC_lower_bound.round(2), AUC_upper_bound.round(2)]
                TPR_patient_wise_score[str(idx + 1) + '. ' + experiment], FPR_patient_wise_score[str(idx + 1) + '. ' + experiment] = ROC_patient_wise(metric_fold, thresh_list)
            df_list.append(pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                            'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                            'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[idx+1].get_label()]* len(mean_FP_per_threshold)]}))
    # Plotting FROC curves lesion wise
    df_conc = pd.concat(df_list, ignore_index=True)
    sns.lineplot(data=df_conc, x='mean_FP_per_threshold', y='recall_per_threshold', hue='Experiments', palette=colors, markers=True,
                 estimator='mean', ci='sd', marker='o')
    plt.xlabel('Mean False Positive per Patient')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1)
    if '6_T1_dataset' in data_dir:
        plt.title('HCC Pre-Ablation Test Set FROC Curves (all lesions)')
    elif '10_test_screening' in data_dir:
        plt.title('HCC Surveillance Test Set FROC Curves (all lesions)')
    plt.grid()
    plt.xlim(0, nbr_of_x_points)
    os.makedirs(os.path.join(main_dir  + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + '/FROC_lesion_wise'), exist_ok=True)
    if patient_type is not None:
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}'), exist_ok=True)
        plt.savefig((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}/FROC_curves_tumor_wise_lesion_overlap_{patient_type}_size_{size}_art_{art_contrast}_seaborn.png'), dpi=300)
    else:
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal'), exist_ok=True)
        plt.savefig((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal/FROC_curves_tumor_wise_size_{size}_art_{art_contrast}_seaborn.png'), dpi=300)
    plt.legend(fontsize=8)
    plt.close()
    df_AUC_all_lesions = pd.DataFrame(AUC_per_experiment, index=['AUC'])

    CI_dict_all_lesions = {}
    for exp in AUC_IC_per_experiment.keys():
        CI_dict_all_lesions[exp] = str(AUC_per_experiment[exp].round(2)) + ' [95%CI ' + str(AUC_IC_per_experiment[exp][0]) + ', ' + str(AUC_IC_per_experiment[exp][1]) + ']'
    if '10_test_screening' in data_dir and patient_type is None and compute_patient_metrics:
        df_list = []
        for idx, ((model, TPR), (model, FPR)) in enumerate(zip(TPR_patient_wise_score.items(), FPR_patient_wise_score.items())):
            fpr_interp = np.linspace(0, 1, 20)
            bootstrap_TPR = TPR_IC_95[model][2]
            interp_reel_TPR = [sample for i in range(bootstrap_nbr) for sample in bootstrap_TPR[i]]
            fpr_interp = [sample for i in range(bootstrap_nbr) for sample in fpr_interp]
            df_list.append(pd.DataFrame({'TPR': interp_reel_TPR, 
                                         'FPR': fpr_interp, 
                                         'Experiments': [proxy_list[idx].get_label()]* len(fpr_interp)}))
            mean_AUC, low_IC_AUC, high_IC_AUC = ROC_AUC_IC_per_experiment[model]
        df_AUC_patient_wise = pd.DataFrame(ROC_AUC_IC_per_experiment, index=['AUC mean', 'AUC_IC_95_lower', 'AUC_IC_95_upper'])
        df_AUC_patient_wise.to_csv(os.path.join(main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/ROC_patient_wise/AUC_patient_wise_tumor_wise_size_{size}.csv'))
        
        # Seaborn plot for patient wise ROC curves
        roc_df = pd.concat(df_list, ignore_index=True)
        sns.lineplot(data=roc_df, x='FPR', y='TPR', hue='Experiments', palette=colors, markers=True, estimator='mean', marker='o', errorbar='sd')
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.axis('square')
        plt.title('Patient Level ROC Curves - HCC Surveillance Test Set')
        plt.grid()
        plt.ylim(0, 1)
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + '/ROC_patient_wise'), exist_ok=True)
        plt.savefig((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/ROC_patient_wise/ROC_curves_patient_wise_tumor_wise_size_{size}_art_{art_contrast}_seaborn.png'), dpi=300)
        plt.close()

    # Li-RADS stratified FROC curves
    AUC_LIRADS_per_LR = {}
    AUC_IC_LIRADS_per_LR = {}
    if '6_T1_dataset' in data_dir:
        LR_names = ['recall_HCC', 'recall_LR_5', 'recall_LR_4', 'recall_LR_3', 'recall_LR_M']
    if '10_test_screening' in data_dir:
        LR_names = ['recall_HCC', 'recall_LR_5', 'recall_LR_4', 'recall_LR_3', 'recall_LR_TIV', 'recall_ven_wash', 'recall_del_wash', 
                    'recall_ven_caps', 'recall_del_caps', 'recall_hyper_art']
        
    for LR_name in LR_names:
        figure = plt.figure(figsize=figsize)
        AUC_per_experiment = {}
        AUC_IC_LIRADS_per_experiment = {}

        if LR_name == 'recall_HCC':
            mean_FP_per_threshold = [nnunet_FROC_metrics_dict['FP_HCC_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        else:
            mean_FP_per_threshold = [nnunet_FROC_metrics_dict['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        recall_per_threshold = [nnunet_FROC_metrics_dict[LR_name][threshold] for threshold in thresh_list]

        recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
        (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper, bootstrap_sensitivity) = compute_IC_95(nnunet_FROC_metrics_dict, LR_name, thresh_list,
                                                                                                bootstrap_nbr, patient_list, nbr_of_x_points)
        df_nnunet = pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                            'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                            'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[0].get_label()]* len(mean_FP_per_threshold)]})
        df_list = [df_nnunet]

        AUC_IC_LIRADS_per_experiment['0. nnunet'] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
        AUC_per_experiment['0. nnunet'] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold).round(2)
        for idx, (experiment, metric) in enumerate(metrics_per_experiment.items()):
            for metric_fold in metric:
                mean_FP_per_threshold = [metric_fold['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
                recall_per_threshold = [metric_fold[LR_name][threshold] for threshold in thresh_list]
                
                recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
                (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper, bootstrap_sensitivity) = compute_IC_95(metric_fold, LR_name, thresh_list, 
                                                                                                        bootstrap_nbr, patient_list, nbr_of_x_points)
                if not isinstance(AUC_IC_95_lower, int):
                    AUC_IC_LIRADS_per_experiment[str(idx + 1) + '. ' + experiment] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
                else:
                    AUC_IC_LIRADS_per_experiment[str(idx + 1) + '. ' + experiment] = [AUC_IC_95_lower, AUC_IC_95_upper]

                AUC_LIRADS = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold)
                if not isinstance(AUC_LIRADS, int):
                    AUC_per_experiment[str(idx + 1) + '. ' + experiment] = AUC_LIRADS.round(2)
                else:
                    AUC_per_experiment[str(idx + 1) + '. ' + experiment] = AUC_LIRADS

                df_list.append(pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                                             'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                                             'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[idx+1].get_label()]* len(mean_FP_per_threshold)]}))
        AUC_LIRADS_per_LR[LR_name] = AUC_per_experiment
        AUC_IC_LIRADS_per_LR[LR_name] = AUC_IC_LIRADS_per_experiment

        if '6_T1_dataset' in data_dir:
            test_set_name = 'HCC Pre-Ablation Test Set'
        else:
            test_set_name = 'HCC Surveillance Follow-Up Test Set'

        if LR_name == 'recall_HCC':
            title_name = f'{test_set_name} FROC Curves (HCC)'
        elif LR_name == 'recall_LR_5':
            title_name = f'{test_set_name} FROC Curves (LR-5)'
        elif LR_name == 'recall_LR_4':
            title_name = f'{test_set_name} FROC Curves (LR-4)'
        elif LR_name == 'recall_LR_3':
            title_name = f'{test_set_name} FROC Curves (LR-3)'
        elif LR_name == 'recall_LR_TIV':
            title_name = f'{test_set_name} FROC Curves (LR-TIV)'
        elif LR_name == 'recall_LR_M':
            title_name = f'{test_set_name} FROC Curves (LR-M)'
        elif LR_name == 'recall_ven_wash':
            title_name = f'{test_set_name} FROC Curves (Venous washout)'
        elif LR_name == 'recall_del_wash':
            title_name = f'{test_set_name} FROC Curves (Delayed washout)'
        elif LR_name == 'recall_ven_caps':
            title_name = f'{test_set_name} FROC Curves (Venous caps)'
        elif LR_name == 'recall_del_caps':
            title_name = f'{test_set_name} FROC Curves (Delayed caps)'
        elif LR_name == 'recall_hyper_art':
            title_name = f'{test_set_name} FROC Curves (Hyperenhancing arterial phase)'
        else:
            break
    
        # Seaborn plot
        df_conc = pd.concat(df_list, ignore_index=True)
        sns.lineplot(data=df_conc, x='mean_FP_per_threshold', y='recall_per_threshold', hue='Experiments', palette=colors, markers=True,
                    estimator='mean', ci='sd', marker='o')
        plt.title(title_name, fontsize=16)
        plt.xlabel('Mean False Positive per Patient')
        plt.ylabel('True Positive Rate')
        plt.ylim(0, 1)
        plt.grid()
        plt.xlim(0, nbr_of_x_points)
        os.makedirs(os.path.join(main_dir  + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + '/FROC_lesion_wise'), exist_ok=True)
        if patient_type is not None:
            os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}'), exist_ok=True)
            plt.savefig((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}/FROC_LIRADS_{LR_name}_curves_tumor_wise_size_{size}_art_{art_contrast}_seaborn.png'), dpi=300)
        else:
            os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal'), exist_ok=True)
            plt.savefig((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal/FROC_LIRADS_{LR_name}_curves_tumor_wise_size_{size}_art_{art_contrast}_seaborn.png'), dpi=300)
        plt.legend(fontsize=10)
        plt.close()
    AUC_lesion_stratified_table = pd.DataFrame(AUC_LIRADS_per_LR)
    round_AUC_lesion_stratified_table = AUC_lesion_stratified_table.round(2)

    CI_dict_LIRADS = {}
    for LR_name in LR_names:
        CI_dict_LIRADS[LR_name] = {}
        for exp in AUC_IC_LIRADS_per_LR[LR_name].keys():
            CI_dict_LIRADS[LR_name][exp] = (str(AUC_LIRADS_per_LR[LR_name][exp]) + ' [95%CI ' + 
                                            str(AUC_IC_LIRADS_per_LR[LR_name][exp][0]) + ', ' + 
                                            str(AUC_IC_LIRADS_per_LR[LR_name][exp][1]) + ']')
    df_CI_all_lesions_T = pd.DataFrame(CI_dict_all_lesions, index=['All lesions']).T
    df_CI_LIRADS = pd.DataFrame(CI_dict_LIRADS)


    df_CI_all_merged = pd.concat([df_CI_all_lesions_T, df_CI_LIRADS], axis=1)
    if patient_type is not None:
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}'), exist_ok=True)
        df_CI_all_merged.to_csv((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}/AUC_CI_all_lesions_and_LIRADS_tumor_wise_size_{size}_art_{art_contrast}.csv'))
    else:
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal'), exist_ok=True)
        df_CI_all_merged.to_csv((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal/AUC_CI_all_lesions_and_LIRADS_tumor_wise_size_{size}_art_{art_contrast}.csv'))

    df_all_lesions_T = df_AUC_all_lesions.T
    df_all_lesions_T = df_all_lesions_T.rename(columns={'AUC': 'All lesions'})
    AUC_df_merged = pd.concat([df_all_lesions_T, round_AUC_lesion_stratified_table], axis=1)
    if patient_type is not None:
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}'), exist_ok=True)
        AUC_df_merged.to_csv((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/{patient_type}/AUC_all_lesions_and_LIRADS_tumor_wise_size_{size}_art_{art_contrast}.csv'))
    else:
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal'), exist_ok=True)
        AUC_df_merged.to_csv((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal/AUC_all_lesions_and_LIRADS_tumor_wise_size_{size}_art_{art_contrast}.csv'))

    if permutation_calculation == True:
        p_values_nnunet_vs_finetuning = {}
        if '6_T1_dataset' in data_dir:
            recall_keys = ['lesions_recall', 'recall_HCC', 'recall_LR_5', 'recall_LR_4', 'recall_LR_3', 'recall_LR_M']
        elif '10_test_screening' in data_dir:
            recall_keys = ['lesions_recall', 'recall_HCC', 'recall_LR_5', 'recall_LR_4', 'recall_LR_3', 'recall_LR_TIV', 'recall_ven_wash', 'recall_ven_caps']
        else:
            print('No recall keys defined')
            return

        if patient_type is not None:
            print(f'Permutation test for lesions_recall')
            p_values_nnunet_vs_finetuning_lesions_recall = run_permutation_test(nnunet_FROC_metrics_dict,
                                                                                metrics_per_experiment[experiment_list[-1]][0],
                                                                                recall_key='lesions_recall',
                                                                                thresh_list=thresh_list,
                                                                                permutation_nbr=permutation_nbr,
                                                                                patient_selection=patient_list,
                                                                                nbr_of_x_points=nbr_of_x_points
                                                                                )
            p_values_nnunet_vs_finetuning_df = pd.DataFrame([p_values_nnunet_vs_finetuning_lesions_recall], index=['p-value'])
            p_values_nnunet_vs_finetuning_df.to_csv(os.path.join(main_dir, 
                                                                 f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + 
                                                                f'/FROC_lesion_wise/{patient_type}/p_values_{permutation_nbr}_permutations_nnunet_vs_finetuning_tumor_wise_size_{size}_art_{art_contrast}_lesion_overlap_{patient_type}.csv'))
        
        for recall_key in recall_keys:
            print(f'Permutation test for {recall_key}')
            p_values_nnunet_vs_finetuning[recall_key] = run_permutation_test(nnunet_FROC_metrics_dict,
                                            metrics_per_experiment[experiment_list[-1]][0],
                                            recall_key=recall_key,
                                            thresh_list=thresh_list,
                                            permutation_nbr=permutation_nbr,
                                            nbr_of_x_points=nbr_of_x_points,
                                            patient_selection=None
                                            )
        p_values_nnunet_vs_finetuning_df = pd.DataFrame(p_values_nnunet_vs_finetuning, index=['p-value'])
        os.makedirs((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/test/' + os.path.basename(data_dir) + f'/FROC_lesion_wise/normal'), exist_ok=True)
        p_values_nnunet_vs_finetuning_df.to_csv((main_dir + f'/training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/test/' + os.path.basename(data_dir) + 
                                                 f'/FROC_lesion_wise/normal/p_values_{permutation_nbr}_permutations_nnunet_vs_finetuning_tumor_wise_size_{size}_art_{art_contrast}.csv'))
    
    return


if __name__ == "__main__":
    train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'training/tumor_segmentation/unet'))
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))

    # data_dir = os.path.join(main_dir, 'HCC_Surveillance/derivatives/10_test_T1_dataset')
    data_dir = os.path.join(main_dir, 'HCC_pre_ablation/derivatives/6_T1_dataset')

    contrast_selection_json_path = os.path.join(data_dir, 'patient_selection/lesion_wise_art_contrast.json')
    diameter_selection_json_path = os.path.join(data_dir, 'patient_selection/lesion_wise_lesion_diameter.json')

    size = 10
    art_contrast = False
    vein_crit = True
    nbr_of_x_points = 10
    permutation_nbr = 1000
    bootstrap_nbr = 1000

    thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    patient_type = None
    permutation_calculation = True

    run_AUC_test(data_dir, patient_type, contrast_selection_json_path, nbr_of_x_points, size, art_contrast, vein_crit,
                 permutation_nbr, bootstrap_nbr, thresh_list, permutation_calculation, compute_patient_metrics=False)

