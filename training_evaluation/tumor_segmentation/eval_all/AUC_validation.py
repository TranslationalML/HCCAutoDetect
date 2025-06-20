import os
import sys
import yaml
import json
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import ast
import seaborn as sns
from utils import get_patient_metrics, run_permutation_test, AUC_FROC_curve, sort_data_for_AUC, compute_IC_95

def main(config, size, vein_crit, art_contrast, patient_list):
    output_folder_dir = os.path.dirname(os.getcwd()) + ("/unet/" + config["output_dir"] + f'/results')

    if art_contrast:
        FROC_metrics_dict = json.load(open(os.path.join(output_folder_dir, f'FROC_valid_metrics_size_{size}_art_{art_contrast}_art.json'), 'r'))
    else:
        FROC_metrics_dict = json.load(open(os.path.join(output_folder_dir, f'FROC_valid_metrics_size_{size}.json'), 'r'))

    try:
        FROC_metrics_dict['FROC_metrics'] = re.sub(r'\bnan\b', 'None', FROC_metrics_dict['FROC_metrics_per_fold_str'])
        FROC_metrics_per_fold = ast.literal_eval(FROC_metrics_dict['FROC_metrics'])
        
        all_folds_metrics = FROC_metrics_per_fold[0]
        for fold in range(1, len(FROC_metrics_per_fold)):
            for threshold, FROC_list in FROC_metrics_per_fold[fold].items():
                for key, value in FROC_list.items():
                    all_folds_metrics[threshold][key].extend(value)

        FROC_metrics_dict = get_patient_metrics(all_folds_metrics, patient_list)
    except:
        print('No FROC metrics')
        FROC_metrics_dict = {}

    return FROC_metrics_dict

def AUC_validation(data_dir, patient_type, json_path, nbr_of_x_points, size, art_contrast, vein_crit, permutation_nbr, bootstrap_nbr, 
                   thresh_list, permutation_calculation):

    if patient_type is not None:
        with open(json_path, 'r') as f:
            patients_with_diff_lesion_overlap = json.load(f)
        patient_list = patients_with_diff_lesion_overlap[patient_type]
    else:
        patient_list = None

    colors = ['g', 'b', 'gold']
    proxy_list = [Patch(color=colors[0], label='Model 1: nnU-Net', alpha=0.3),
                  Patch(color=colors[1], label='Model 2: U-Net (Tversky)', alpha=0.3),
                  Patch(color=colors[2], label='Model 3: U-Net (Pre-training + Tversky)', alpha=0.3)]
    
    main_saving_path = 'FROC_evaluation'
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

        output_dir = os.path.join("tumor_eval",  experiment[:-5], os.path.basename(data_dir))
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
        FROC_metrics = main(merged_config, size, vein_crit, art_contrast, patient_list)
  

        metrics_per_experiment[experiment] = (FROC_metrics)

    # nnunet
    valid_pred_dir = os.path.join(main_dir, "training/tumor_segmentation/nnunet/work_dir/nnUNet_results/Dataset030_HCC_Surveillance_patients/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/results")
    with open(os.path.join(valid_pred_dir, f'FROC_valid_metrics_size_{size}.json'), 'r') as f:
        nnunet_FROC_metrics_init = json.load(f)
    nnunet_FROC_metrics_init['FROC_metrics'] = re.sub(r'\bnan\b', 'None', nnunet_FROC_metrics_init['FROC_metrics_per_fold_str'])
    FROC_metrics_per_fold = ast.literal_eval(nnunet_FROC_metrics_init['FROC_metrics'])

    all_folds_metrics = FROC_metrics_per_fold[0]
    for fold in range(1, len(FROC_metrics_per_fold)):
        for threshold, FROC_list in FROC_metrics_per_fold[fold].items():
            for key, value in FROC_list.items():
                all_folds_metrics[threshold][key].extend(value)
    nnunet_FROC_metrics_dict = get_patient_metrics(all_folds_metrics, patient_list)
    AUC_per_experiment = {}
    AUC_IC_per_experiment = {}
    recall_per_experiment = {}
    mean_FP_per_experiment = {}

    # ALL lesions plot
    max_TP_list_per_model = {}
    mean_FP_per_threshold = [nnunet_FROC_metrics_dict['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
    recall_per_threshold = [nnunet_FROC_metrics_dict['lesions_recall'][threshold] for threshold in thresh_list]

    recall_per_experiment['0. nnunet'] = recall_per_threshold
    mean_FP_per_experiment['0. nnunet'] = mean_FP_per_threshold

    recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
    (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper) = compute_IC_95(nnunet_FROC_metrics_dict, 'lesions_recall', 
                                                                                            thresh_list, bootstrap_nbr, patient_list, 
                                                                                            nbr_of_x_points)
    
    max_TP_list_per_model['0. nnunet'] = {key: value for key, value in zip(nnunet_FROC_metrics_dict['patient_name'], nnunet_FROC_metrics_dict['TP_per_patient'][thresh_list[0]])}
    AUC_per_experiment['0. nnunet'] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold)
    AUC_IC_per_experiment['0. nnunet'] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
    df_list = [df_nnunet]

    for idx, (experiment, metric) in enumerate(metrics_per_experiment.items()):
        mean_FP_per_threshold = [metric['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        recall_per_threshold = [metric['lesions_recall'][threshold] for threshold in thresh_list]

        recall_per_experiment[str(idx + 1) + '. ' + experiment] = recall_per_threshold
        mean_FP_per_experiment[str(idx + 1) + '. ' + experiment] = mean_FP_per_threshold

        recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
        (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper) = compute_IC_95(metric, 'lesions_recall', thresh_list, 
                                                                                                    bootstrap_nbr, patient_list, nbr_of_x_points)
    
        max_TP_list_per_model[str(idx + 1) + '. ' + experiment] = {key: value for key, value in zip(metric['patient_name'], metric['TP_per_patient'][thresh_list[0]])}
        AUC_per_experiment[str(idx + 1) + '. ' + experiment] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold)
        AUC_IC_per_experiment[str(idx + 1) + '. ' + experiment] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
        df_list.append(pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                                        'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                                        'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[idx+1].get_label()]* len(mean_FP_per_threshold)]}))
    
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
    rounded_AUC = df_AUC_all_lesions.round(2)

    # Create 95% CI table
    CI_dict_all_lesions = {}
    for exp in AUC_IC_per_experiment.keys():
        CI_dict_all_lesions[exp] = str(AUC_per_experiment[exp].round(2)) + ' [95%CI ' + str(AUC_IC_per_experiment[exp][0]) + ', ' + str(AUC_IC_per_experiment[exp][1]) + ']'
    rounded_AUC.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/AUC_all_lesions_tumor_wise_size_{size}.csv'))
    df_CI_all_lesions = pd.DataFrame(CI_dict_all_lesions, index=['All lesions']).T
    df_CI_all_lesions.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/AUC_CI_all_lesions_tumor_wise_size_{size}.csv'))


    # Li-RADS stratified FROC curves
    AUC_LIRADS_per_experiment = {}
    AUC_IC_LIRADS_per_experiment = {}
    recall_LIRADS_per_experiment = {}
    mean_LIRADS_FP_per_experiment = {}
    LR_names = ['recall_HCC', 'recall_LR_5', 'recall_LR_4', 'recall_LR_3', 'recall_LR_TIV', 'recall_LR_M', 'recall_ven_wash', 'recall_del_wash', 'recall_ven_caps', 'recall_del_caps', 'recall_hyper_art']
    for LR_name in LR_names:
        AUC_per_experiment = {}
        AUC_IC_per_experiment = {}
        recall_per_experiment = {}
        mean_FP_per_experiment = {}

        if LR_name == 'recall_HCC':
            mean_FP_per_threshold = [nnunet_FROC_metrics_dict['FP_HCC_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        else:
            mean_FP_per_threshold = [nnunet_FROC_metrics_dict['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]

        recall_per_threshold = [nnunet_FROC_metrics_dict[LR_name][threshold] for threshold in thresh_list]
        recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
        
        (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper, bootstrap_sensitivity) = compute_IC_95(nnunet_FROC_metrics_dict, LR_name, 
                                                                                                    thresh_list, bootstrap_nbr, patient_list, 
                                                                                                    nbr_of_x_points)
        AUC_IC_per_experiment['0. nnunet'] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]
        recall_per_experiment['0. nnunet'] = recall_per_threshold
        mean_FP_per_experiment['0. nnunet'] = mean_FP_per_threshold
        AUC_per_experiment['0. nnunet'] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold)
        df_nnunet = pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                        'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                        'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[0].get_label()]* len(mean_FP_per_threshold)]})
        df_list = [df_nnunet]

        for idx, (experiment, metric) in enumerate(metrics_per_experiment.items()):
            if LR_name == 'recall_HCC':
                mean_FP_per_threshold = [metric['FP_HCC_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
            else:
                mean_FP_per_threshold = [metric['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
            recall_per_threshold = [metric[LR_name][threshold] for threshold in thresh_list]

            recall_per_experiment[str(idx + 1) + '. ' + experiment] = recall_per_threshold
            mean_FP_per_experiment[str(idx + 1) + '. ' + experiment] = mean_FP_per_threshold
            (recall_IC_95_lower, recall_IC_95_upper, AUC_IC_95_lower, AUC_IC_95_upper) = compute_IC_95(metric, LR_name, thresh_list,
                                                                                        bootstrap_nbr, patient_list, nbr_of_x_points)
            recall_per_threshold, mean_FP_per_threshold = sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points)
            AUC_per_experiment[str(idx + 1) + '. ' + experiment] = AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold)
            AUC_IC_per_experiment[str(idx + 1) + '. ' + experiment] = [AUC_IC_95_lower.round(2), AUC_IC_95_upper.round(2)]

            df_list.append(pd.DataFrame({'mean_FP_per_threshold': [mean_FP_ for i in range(bootstrap_nbr) for mean_FP_ in mean_FP_per_threshold], 
                                            'recall_per_threshold': [recall_ for recall_threshold in bootstrap_sensitivity for recall_ in recall_threshold], 
                                            'Experiments': [elem for i in range(bootstrap_nbr) for elem in [proxy_list[idx+1].get_label()]* len(mean_FP_per_threshold)]}))
        AUC_LIRADS_per_experiment[LR_name] = AUC_per_experiment
        recall_LIRADS_per_experiment[LR_name] = recall_per_experiment
        mean_LIRADS_FP_per_experiment[LR_name] = mean_FP_per_experiment
        AUC_IC_LIRADS_per_experiment[LR_name] = AUC_IC_per_experiment

        if LR_name == 'recall_HCC':
            title_name = f'Validation set FROC Curves (HCC)'
        elif LR_name == 'recall_LR_5':
            title_name = f'Validation set FROC Curves (LR-5)'
        elif LR_name == 'recall_LR_4':
            title_name = 'Validation set FROC Curves (LR-4)'
        elif LR_name == 'recall_LR_3':
            title_name = 'Validation set FROC Curves (LR-3)'
        elif LR_name == 'recall_LR_TIV':
            title_name = 'Validation set FROC Curves (LR-TIV)'
        elif LR_name == 'recall_LR_M':
            title_name = 'Validation set FROC Curves (LR-M)'
        elif LR_name == 'recall_ven_wash':
            title_name = 'Validation set FROC Curves (Venous washout)'
        elif LR_name == 'recall_del_wash':
            title_name = 'Validation set FROC Curves (Delayed washout)'
        elif LR_name == 'recall_ven_caps':
            title_name = 'Validation set FROC Curves (Venous caps)'
        elif LR_name == 'recall_del_caps':
            title_name = 'Validation set FROC Curves (Delayed caps)'
        elif LR_name == 'recall_hyper_art':
            title_name = 'Validation set FROC Curves (Hyperenhancing arterial phase)'
        else:
            break

        df_conc = pd.concat(df_list, ignore_index=True)
        sns.lineplot(data=df_conc, x='mean_FP_per_threshold', y='recall_per_threshold', hue='Experiments', palette=colors, markers=True,
                    estimator='mean', ci='sd', marker='o')
        plt.xlabel('Mean False Positive per Patient')
        plt.ylabel('True Positive Rate')
        plt.title(title_name)
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
    AUC_lesion_stratified_table = pd.DataFrame(AUC_LIRADS_per_experiment)
    df_AUC_LIRADS = AUC_lesion_stratified_table.round(2)
    df_AUC_LIRADS.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/AUC_LIRADS_tumor_wise_size_{size}.csv'))

    CI_dict_LIRADS = {}
    for LR_name in LR_names:
        CI_dict_LIRADS[LR_name] = {}
        for exp in AUC_IC_LIRADS_per_experiment[LR_name].keys():
            CI_dict_LIRADS[LR_name][exp] = (str(AUC_LIRADS_per_experiment[LR_name][exp].round(3)) + ' [95%CI ' + 
                                            str(AUC_IC_LIRADS_per_experiment[LR_name][exp][0]) + ', ' + 
                                            str(AUC_IC_LIRADS_per_experiment[LR_name][exp][1]) + ']')
    df_CI_LIRADS = pd.DataFrame(CI_dict_LIRADS)
    df_CI_LIRADS.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/AUC_CI_LIRADS_tumor_wise_size_{size}.csv'))


    df_CI_all_merged = pd.concat([df_CI_all_lesions, df_CI_LIRADS], axis=1)
    if patient_type is not None:
        os.makedirs(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation//FROC_lesion_wise/{patient_type}'), exist_ok=True)
        df_CI_all_merged.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation//FROC_lesion_wise/{patient_type}/AUC_CI_all_lesions_and_LIRADS_tumor_wise_size_{size}.csv'))
    else:
        os.makedirs(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/normal'), exist_ok=True)
        df_CI_all_merged.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/normal/AUC_CI_all_lesions_and_LIRADS_tumor_wise_size_{size}.csv'))

    df_all_lesions_T = df_AUC_all_lesions.T
    df_all_lesions_T = df_all_lesions_T.rename(columns={'AUC': 'All lesions'})
    AUC_df_merged = pd.concat([df_all_lesions_T, df_AUC_LIRADS], axis=1)
    if patient_type is not None:
        AUC_df_merged.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/{patient_type}/AUC_all_lesions_and_LIRADS_tumor_wise_{patient_type}_size_{size}.csv'))
    else:
        AUC_df_merged.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/normal/AUC_all_lesions_and_LIRADS_tumor_wise_size_{size}.csv'))

    if permutation_calculation == True:
        p_values_nnunet_vs_finetuning = {}
        if patient_type is not None:
            recall_keys = ['lesions_recall']
        else:
            recall_keys = ['lesions_recall', 'recall_HCC', 'recall_LR_5', 'recall_LR_4', 'recall_LR_3', 'recall_ven_wash',
                        'recall_del_wash', 'recall_ven_caps', 'recall_del_caps']
        for recall_key in recall_keys:
            print(f'Permutation test for {recall_key}')
            p_values_nnunet_vs_finetuning[recall_key] = run_permutation_test(nnunet_FROC_metrics_dict,
                                            metrics_per_experiment['tumor_segmentation/tumor_segmentation_finetuning/3. Unet_pretrained_Tversky_loss.yaml'],
                                            recall_key=recall_key,
                                            thresh_list=thresh_list,
                                            permutation_nbr=permutation_nbr,
                                            patient_selection=patient_list,
                                            nbr_of_x_points=nbr_of_x_points
                                            )
        p_values_nnunet_vs_finetuning_df = pd.DataFrame(p_values_nnunet_vs_finetuning, index=['p-value'])
        if patient_type is not None:
            p_values_nnunet_vs_finetuning_df.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/{patient_type}/p_values_{permutation_nbr}_permutations_nnunet_vs_finetuning_tumor_wise_size_{size}.csv'))
        else:
            p_values_nnunet_vs_finetuning_df.to_csv(os.path.join(main_dir, f'training_evaluation/tumor_segmentation/eval_all/{main_saving_path}/validation/FROC_lesion_wise/normal/p_values_{permutation_nbr}_permutations_nnunet_vs_finetuning_tumor_wise_size_{size}.csv'))

    return


if __name__ == "__main__":
    train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'training/tumor_segmentation/unet'))
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(train_dir)))
    data_dir = os.path.join(main_dir, 'HCC_Surveillance/derivatives/10_T1_dataset')
  
    contrast_selection_json_path = os.path.join(data_dir, 'patient_selection/lesion_wise_art_contrast.json')
    diameter_selection_json_path = os.path.join(data_dir, 'patient_selection/lesion_wise_lesion_diameter.json')

    x_lim_plot = 10
    nbr_of_x_points = 10
    permutation_nbr = 1000
    bootstrap_nbr = 1000
    thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999]
    permutation_calculation = True
    patient_type = None
    size = 10
    art_contrast = False
    vein_crit = True
    AUC_validation(data_dir, patient_type, contrast_selection_json_path, nbr_of_x_points, x_lim_plot, size, art_contrast, vein_crit, permutation_nbr, bootstrap_nbr,
                   thresh_list,permutation_calculation)
    