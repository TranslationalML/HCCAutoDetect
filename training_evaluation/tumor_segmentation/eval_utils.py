import os
import numpy as np
import matplotlib.pyplot as plt
import json
import SimpleITK as sitk
import sys
import cc3d
import seg_metrics.seg_metrics as sg
from random import randint
from collections import Counter

training_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'training/tumor_segmentation/unet'))
print(training_dir)
sys.path.append(training_dir)


def LIRADSWashoutCapsuleEval(LIRADS_list_per_fold, TP_per_fold, FP_per_fold, ven_caps_per_fold, del_caps_per_fold, ven_wash_per_fold,
                            del_wash_per_fold, hyper_art_per_fold, tumor_size, plot_saving_path, is_test=False):
    """    Evaluate the performance of the model on a list of patients for LIRADS, washout and capsule detection    
    Parameters:
        LIRADS_list_per_fold: list of lists of LIRADS scores per fold
        TP_per_fold: list of lists of true positives per fold
        FP_per_fold: list of lists of false positives per fold
        ven_caps_per_fold: list of lists of venous capsule detection per fold
        del_caps_per_fold: list of lists of delayed capsule detection per fold
        ven_wash_per_fold: list of lists of venous washout detection per fold
        del_wash_per_fold: list of lists of delayed washout detection per fold
        hyper_art_per_fold: list of lists of hyperenhancement arterial detection per fold
        tumor_size: size of the tumor
        plot_saving_path: path to save the plots
        is_test: boolean indicating if the evaluation is for test set or not
    Returns: None, but saves the plots and the performance metrics in a json file
    """
    # Stratified performance
    LIRADS_5_list_per_fold = [[1 if LIRAD == '5' else 0 for LIRAD in LIRADS_fold] for LIRADS_fold in
                              LIRADS_list_per_fold]
    LIRADS_4_list_per_fold = [[1 if LIRAD == '4' else 0 for LIRAD in LIRADS_fold] for LIRADS_fold in
                              LIRADS_list_per_fold]
    LIRADS_3_list_per_fold = [[1 if LIRAD == '3' else 0 for LIRAD in LIRADS_fold] for LIRADS_fold in
                              LIRADS_list_per_fold]
    LIRADS_2_list_per_fold = [[1 if LIRAD == '2' else 0 for LIRAD in LIRADS_fold] for LIRADS_fold in
                              LIRADS_list_per_fold]
    LIRADS_TIV_list_per_fold = [[1 if LIRAD == 'TIV' else 0 for LIRAD in LIRADS_fold] for LIRADS_fold in
                                LIRADS_list_per_fold]
    LIRADS_M_list_per_fold = [[1 if LIRAD == 'M' else 0 for LIRAD in LIRADS_fold] for LIRADS_fold in
                              LIRADS_list_per_fold]
    TP_per_fold = [[TP for TP_list in TP_fold for TP in TP_list] for TP_fold in TP_per_fold]

    def eval_stratified(TP_list_per_fold, LIRADS_list_per_fold, LIRADS_5_list_per_fold, LIRADS_score):
        LIRADS_TP_list_per_fold = [[TP_list[i] for i, LIRAD in enumerate(LIRADS_fold) if LIRAD == LIRADS_score] for
                                   TP_list, LIRADS_fold in zip(TP_list_per_fold, LIRADS_list_per_fold)]
        LIRADS_perentage_deteced = [np.sum(np.sum(TP_list)) / np.sum(np.sum(LIRADS_5_list)) for
                                    TP_list, LIRADS_5_list in
                                    zip(LIRADS_TP_list_per_fold, LIRADS_5_list_per_fold)]
        LIRADS_perentage_deteced = [elem for elem in LIRADS_perentage_deteced if not np.isnan(elem)]
        return np.mean(LIRADS_perentage_deteced), np.std(LIRADS_perentage_deteced)

    perc_lirads_5_mean, perc_lirads_5_std = eval_stratified(TP_per_fold, LIRADS_list_per_fold, LIRADS_5_list_per_fold,
                                                            '5')
    perc_lirads_4_mean, perc_lirads_4_std = eval_stratified(TP_per_fold, LIRADS_list_per_fold, LIRADS_4_list_per_fold,
                                                            '4')
    perc_lirads_3_mean, perc_lirads_3_std = eval_stratified(TP_per_fold, LIRADS_list_per_fold, LIRADS_3_list_per_fold,
                                                            '3')
    perc_lirads_2_mean, perc_lirads_2_std = eval_stratified(TP_per_fold, LIRADS_list_per_fold, LIRADS_2_list_per_fold,
                                                            '2')
    perc_lirads_TIV_mean, perc_lirads_TIV_std = eval_stratified(TP_per_fold, LIRADS_list_per_fold,
                                                                LIRADS_TIV_list_per_fold, 'TIV')
    perc_lirads_M_mean, perc_lirads_M_std = eval_stratified(TP_per_fold, LIRADS_list_per_fold, LIRADS_M_list_per_fold,
                                                            'M')

    # Plot stratified detection
    plt.figure(figsize=(10, 5))
    x = np.arange(6)
    plt.bar(x[0], perc_lirads_5_mean, yerr=perc_lirads_5_std, capsize=5, align='center', label='LIRADS 5')
    plt.bar(x[1], perc_lirads_4_mean, yerr=perc_lirads_4_std, capsize=5, align='center', label='LIRADS 4')
    plt.bar(x[2], perc_lirads_3_mean, yerr=perc_lirads_3_std, capsize=5, align='center', label='LIRADS 3')
    plt.bar(x[3], perc_lirads_2_mean, yerr=perc_lirads_2_std, capsize=5, align='center', label='LIRADS 2')
    plt.bar(x[4], perc_lirads_TIV_mean, yerr=perc_lirads_TIV_std, capsize=5, align='center', label='LIRADS TIV')
    plt.bar(x[5], perc_lirads_M_mean, yerr=perc_lirads_M_std, capsize=5, align='center', label='LIRADS M')
    plt.text(0, 0.01, f'{perc_lirads_5_mean:.2f} ± {perc_lirads_5_std:.2f}', ha='center')
    plt.text(1, 0.01, f'{perc_lirads_4_mean:.2f} ± {perc_lirads_4_std:.2f}', ha='center')
    plt.text(2, 0.01, f'{perc_lirads_3_mean:.2f} ± {perc_lirads_3_std:.2f}', ha='center')
    plt.text(3, 0.01, f'{perc_lirads_2_mean:.2f} ± {perc_lirads_2_std:.2f}', ha='center')
    plt.text(4, 0.01, f'{perc_lirads_TIV_mean:.2f} ± {perc_lirads_TIV_std:.2f}', ha='center')
    plt.text(5, 0.01, f'{perc_lirads_M_mean:.2f} ± {perc_lirads_M_std:.2f}', ha='center')
    plt.xticks(x, ['LIRADS 5', 'LIRADS 4', 'LIRADS 3', 'LIRADS 2', 'LIRADS TIV', 'LIRADS M'])
    plt.ylabel('Percentage of tumors detected')
    plt.ylim(0, 1)
    plt.title('Stratified detection')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if is_test:
        plt.savefig(plot_saving_path + "/LIRADS_stratified_performance_{}_test.png".format(tumor_size))
    else:
        plt.savefig(plot_saving_path + "/LIRADS_stratified_performance_{}.png".format(tumor_size))
    plt.close()

    def eval_caps_wash_art(caps_wash_art_list_per_fold, TP_per_fold):
        caps_wash_art_list = [[capsule for capsule_list in capsule_fold for capsule in capsule_list]
                              for capsule_fold in caps_wash_art_list_per_fold]
        detected_caps_wash_art = [[TP_list[i] for i, caps in enumerate(caps_wash_art_label) if caps == 1] for
                                  TP_list, caps_wash_art_label in zip(TP_per_fold, caps_wash_art_list)]
        perc_caps_wash_art_mean_per_fold = [np.sum(detected_caps_wash_art_list) / np.sum(caps_wash_art_ref) for
                                            detected_caps_wash_art_list, caps_wash_art_ref in
                                            zip(detected_caps_wash_art, caps_wash_art_list)]
        return np.mean(perc_caps_wash_art_mean_per_fold), np.std(perc_caps_wash_art_mean_per_fold)

    (caps_ven_mean, caps_ven_std) = eval_caps_wash_art(ven_caps_per_fold, TP_per_fold)
    (caps_del_mean, caps_del_std) = eval_caps_wash_art(del_caps_per_fold, TP_per_fold)
    (wash_ven_mean, wash_ven_std) = eval_caps_wash_art(ven_wash_per_fold, TP_per_fold)
    (wash_del_mean, wash_del_std) = eval_caps_wash_art(del_wash_per_fold, TP_per_fold)
    (hyper_art_mean, hyper_art_std) = eval_caps_wash_art(hyper_art_per_fold, TP_per_fold)
    hyper_ven_wash = [
        [[1 if art == 1 and wash == 1 else 0 for art, wash in zip(art_list, wash_list)] for art_list, wash_list in
         zip(hyper_art_label, ven_wash_label)]
        for hyper_art_label, ven_wash_label in zip(hyper_art_per_fold, ven_wash_per_fold)]
    hyper_del_wash = [
        [[1 if art == 1 and wash == 1 else 0 for art, wash in zip(art_list, wash_list)] for art_list, wash_list in
         zip(hyper_art_label, del_wash_label)]
        for hyper_art_label, del_wash_label in zip(hyper_art_per_fold, del_wash_per_fold)]
    hyper_ven_caps = [
        [[1 if art == 1 and caps == 1 else 0 for art, caps in zip(art_list, caps_list)] for art_list, caps_list in
         zip(hyper_art_label, ven_caps_label)]
        for hyper_art_label, ven_caps_label in zip(hyper_art_per_fold, ven_caps_per_fold)]
    hyper_del_caps = [
        [[1 if art == 1 and caps == 1 else 0 for art, caps in zip(art_list, caps_list)] for art_list, caps_list in
         zip(hyper_art_label, del_caps_label)]
        for hyper_art_label, del_caps_label in zip(hyper_art_per_fold, del_caps_per_fold)]
    (hyper_ven_wash_mean, hyper_ven_wash_std) = eval_caps_wash_art(hyper_ven_wash, TP_per_fold)
    (hyper_del_wash_mean, hyper_del_wash_std) = eval_caps_wash_art(hyper_del_wash, TP_per_fold)
    (hyper_ven_caps_mean, hyper_ven_caps_std) = eval_caps_wash_art(hyper_ven_caps, TP_per_fold)
    (hyper_del_caps_mean, hyper_del_caps_std) = eval_caps_wash_art(hyper_del_caps, TP_per_fold)

    # Plot capsule and washout detection
    plt.figure(figsize=(10, 5))
    x = np.arange(5)
    plt.bar(x[0], caps_ven_mean, yerr=caps_ven_std, capsize=5, align='center', label='Capsule venous')
    plt.bar(x[1], caps_del_mean, yerr=caps_del_std, capsize=5, align='center', label='Capsule delayed')
    plt.bar(x[2], wash_ven_mean, yerr=wash_ven_std, capsize=5, align='center', label='Washout venous')
    plt.bar(x[3], wash_del_mean, yerr=wash_del_std, capsize=5, align='center', label='Washout delayed')
    plt.bar(x[4], hyper_art_mean, yerr=hyper_art_std, capsize=5, align='center',
            label='Hyperenhancement arterial')
    plt.text(0, 0.01, f'{caps_ven_mean:.2f} ± {caps_ven_std:.2f}', ha='center')
    plt.text(1, 0.01, f'{caps_del_mean:.2f} ± {caps_del_std:.2f}', ha='center')
    plt.text(2, 0.01, f'{wash_ven_mean:.2f} ± {wash_ven_std:.2f}', ha='center')
    plt.text(3, 0.01, f'{wash_del_mean:.2f} ± {wash_del_std:.2f}', ha='center')
    plt.text(4, 0.01, f'{hyper_art_mean:.2f} ± {hyper_art_std:.2f}', ha='center')

    plt.xticks(x, ['Capsule venous', 'Capsule delayed', 'Washout venous', 'Washout delayed',
                   'Hyperenhancement arterial'])
    plt.ylabel('Percentage of tumors detected')
    plt.ylim(0, 1)
    plt.title('Stratified detection')
    plt.legend(loc='center left')
    plt.tight_layout()
    if is_test:
        plt.savefig(plot_saving_path + "/wash_caps_stratified_performance_{}_test.png".format(tumor_size))
    else:
        plt.savefig(plot_saving_path + "/wash_caps_stratified_performance_{}.png".format(tumor_size))
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(4)
    plt.bar(x[0], hyper_ven_wash_mean, yerr=hyper_ven_wash_std, capsize=5, align='center',
            label='Hyperenhancement and venous washout')
    plt.bar(x[1], hyper_del_wash_mean, yerr=hyper_del_wash_std, capsize=5, align='center',
            label='Hyperenhancement and delayed washout')
    plt.bar(x[2], hyper_ven_caps_mean, yerr=hyper_ven_caps_std, capsize=5, align='center',
            label='Hyperenhancement and venous capsule')
    plt.bar(x[3], hyper_del_caps_mean, yerr=hyper_del_caps_std, capsize=5, align='center',
            label='Hyperenhancement and delayed capsule')

    plt.text(0, 0.01, f'{hyper_ven_wash_mean:.2f} ± {hyper_ven_wash_std:.2f}', ha='center')
    plt.text(1, 0.01, f'{hyper_del_wash_mean:.2f} ± {hyper_del_wash_std:.2f}', ha='center')
    plt.text(2, 0.01, f'{hyper_ven_caps_mean:.2f} ± {hyper_ven_caps_std:.2f}', ha='center')
    plt.text(3, 0.01, f'{hyper_del_caps_mean:.2f} ± {hyper_del_caps_std:.2f}', ha='center')

    plt.xticks(x, ['Venous washout', 'Delayed washout', 'Venous capsule', 'Delayed capsule'])
    plt.ylabel('Percentage of tumors detected')
    plt.ylim(0, 1)
    plt.title('Stratified detection')
    plt.legend(loc='center left')
    plt.tight_layout()
    if is_test:
        plt.savefig(plot_saving_path + "/hyper_wash_pattern_stratified_performance_{}_test.png".format(tumor_size))
    else:
        plt.savefig(plot_saving_path + "/hyper_wash_pattern_stratified_performance_{}.png".format(tumor_size))
    plt.close()


    def sensi_speci_malignancy(TP_per_fold, flat_hyper_del_wash):
        TP_hyper_wash = [[1 for TP, wash_art in zip(TP_list, wash_art_label) if wash_art == 1 and TP == 1] for
                                        TP_list, wash_art_label in zip(TP_per_fold, flat_hyper_del_wash)]
        FP_hyper_wash = [[1 for TP, wash_art in zip(TP_list, wash_art_label) if (wash_art == 0 and TP == 1) or TP == 0] for
                                        TP_list, wash_art_label in zip(TP_per_fold, flat_hyper_del_wash)]
        FN_hyper_wash = [[1 for TP, wash_art in zip(TP_list, wash_art_label) if wash_art == 1 and TP == 0] for
                                        TP_list, wash_art_label in zip(TP_per_fold, flat_hyper_del_wash)]
        TN_hyper_wash = [[1 for TP, wash_art in zip(TP_list, wash_art_label) if wash_art == 0 and TP == 0] for
                                        TP_list, wash_art_label in zip(TP_per_fold, flat_hyper_del_wash)]
        sensitivity_hyper_wash = [np.sum(np.sum(TP_list)) / (np.sum(np.sum(TP_list)) + np.sum(np.sum(FN_list))) for
                                        TP_list, FN_list in zip(TP_hyper_wash, FN_hyper_wash)]
        specificity_hyper_wash = [np.sum(np.sum(TN_list)) / (np.sum(np.sum(TN_list)) + np.sum(np.sum(FP_list))) for
                                        TN_list, FP_list in zip(TN_hyper_wash, FP_hyper_wash)]
        F1_hyper_wash = [2 * (sensitivity * specificity) / (sensitivity + specificity) for sensitivity, specificity in
                                        zip(sensitivity_hyper_wash, specificity_hyper_wash)]
        F2_hyper_wash = [5 * (sensitivity * specificity) / (4 * sensitivity + specificity) for sensitivity, specificity in
                                        zip(sensitivity_hyper_wash, specificity_hyper_wash)]
        return sensitivity_hyper_wash, specificity_hyper_wash, F1_hyper_wash, F2_hyper_wash
    flat_hyper_del_wash = [[capsule for capsule_list in capsule_fold for capsule in capsule_list]
                           for capsule_fold in hyper_del_wash]
    flat_hyper_del_caps = [[capsule for capsule_list in capsule_fold for capsule in capsule_list]
                            for capsule_fold in hyper_del_caps]
    sensi_hyper_del_wash, speci_hyper_del_wash, F1_hyper_del_wash, F2_hyper_del_wash = sensi_speci_malignancy(TP_per_fold, flat_hyper_del_wash)
    sensi_hyper_del_caps, speci_hyper_del_caps, F1_hyper_del_caps, F2_hyper_del_caps = sensi_speci_malignancy(TP_per_fold, flat_hyper_del_caps)

    if is_test:
        plot_saving_json_name = plot_saving_path + "/LIRADS_stratified_performance_{}_test.json".format(tumor_size)
    else:
        plot_saving_json_name = plot_saving_path + "/LIRADS_stratified_performance_{}.json".format(tumor_size)
    with open(plot_saving_json_name, 'w') as f:
        json.dump({'LIRADS_5': {'mean': perc_lirads_5_mean, 'std': perc_lirads_5_std},
                     'LIRADS_4': {'mean': perc_lirads_4_mean, 'std': perc_lirads_4_std},
                     'LIRADS_3': {'mean': perc_lirads_3_mean, 'std': perc_lirads_3_std},
                     'LIRADS_2': {'mean': perc_lirads_2_mean, 'std': perc_lirads_2_std},
                     'LIRADS_TIV': {'mean': perc_lirads_TIV_mean, 'std': perc_lirads_TIV_std},
                     'LIRADS_M': {'mean': perc_lirads_M_mean, 'std': perc_lirads_M_std},
                     'Hyper-enhancement + venous washout': {'mean': hyper_ven_wash_mean, 'std': hyper_ven_wash_std},
                     'Hyper-enhancement + delayed washout': {'mean': hyper_del_wash_mean, 'std': hyper_del_wash_std},
                     'Hyper-enhancement + venous capsule': {'mean': hyper_ven_caps_mean, 'std': hyper_ven_caps_std},
                     'Hyper-enhancement + delayed capsule': {'mean': hyper_del_caps_mean, 'std': hyper_del_caps_std},
                        'Hyper-enhancement + delayed washout metrics': {'sensitivity_mean': np.mean(sensi_hyper_del_wash), 'sensitivity_std': np.std(sensi_hyper_del_wash),
                                                                        'specificity_mean': np.mean(speci_hyper_del_wash), 'specificity_std': np.std(speci_hyper_del_wash),
                                                                        'F1_mean': np.mean(F1_hyper_del_wash), 'F1_std': np.std(F1_hyper_del_wash),
                                                                        'F2_mean': np.mean(F2_hyper_del_wash), 'F2_std': np.std(F2_hyper_del_wash)},
                        'Hyper-enhancement + delayed capsule metrics': {'sensitivity_mean': np.mean(sensi_hyper_del_caps), 'sensitivity_std': np.std(sensi_hyper_del_caps),
                                                                        'specificity_mean': np.mean(speci_hyper_del_caps), 'specificity_std': np.std(speci_hyper_del_caps),
                                                                        'F1_mean': np.mean(F1_hyper_del_caps), 'F1_std': np.std(F1_hyper_del_caps),
                                                                        'F2_mean': np.mean(F2_hyper_del_caps), 'F2_std': np.std(F2_hyper_del_caps)}}, f, indent=4)
    return


def eval_perf_fold(prob_map, images, preds, gt, gt_multilabels, liver_labels, vein_labels,
                   min_diameter, config, output_dir, lesions_characteristics, threshold=None, nbr_of_neg_patch_trials=1000):
    """
    Evaluate the performance of the model on a list of patients for a specific threshold
    Parameters:
        prob_map: list of paths to the probability maps
        images: list of paths to the images
        preds: list of paths to the predictions
        gt: list of paths to the ground truth
        gt_multilabels: list of paths to the multilabel ground truth
        liver_labels: list of paths to the liver masks
        vein_labels: list of paths to the vein masks
        tumor_size: size of the tumor
        config: configuration dictionary
        output_dir: output directory
        lesions_characteristics: dataframe containing the characteristics of the lesions
        threshold: threshold to apply on the probability maps

    Returns: dict, containing the performance metrics:
                TP: list of true positives
                FP: list of false positives
                FN: list of false negatives
                TN: list of true negatives
                patient_tumor_metrics_list: list of the metrics per patient calculated with bainary prediction/ground truth masks
                tumor_metrics_list: list of the metrics per tumor
                FN_preds: list of the false negative predictions
                FP_preds: list of the false positive predictions
                patients: list of the patients
                generated_neg_samples: list of the generated negative samples
                LIRADS_list: list of the LIRADS scores
                wash_ven_list: list of the venous washout
                wash_del_list: list of the delayed washout
                caps_ven_list: list of the venous capsule
                caps_del_list: list of the delayed capsule
                hyper_art_list: list of the hyperenhancement arterial
                lesion_type_list: list of the lesion types
    """
    
    patient_tumor_metrics_list = []
    tumor_metrics_list = []
    TN_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    FN_preds = []
    FP_preds = []
    patients = []
    preds_nbr = []
    generated_neg_samples = []
    LIRADS_list = []
    HCC_list = []
    wash_ven_list = []
    wash_del_list = []
    caps_ven_list = []
    caps_del_list = []
    hyper_art_list = []
    lesion_type_list = []
    prob_map_stats_list = []
    for (prob_map_lab, image_4D, pred, label, multi_label, liver_label, vein_label) in (
            zip(prob_map, images, preds, gt, gt_multilabels, liver_labels, vein_labels)):
        TP = []
        FP = []
        FN = []
        TN = []
        LIRADS = []
        HCC_lesions = []
        wash_ven = []
        wash_del = []
        caps_ven = []
        caps_del = []
        hyper_art = []
        lesion_type = []

        print(os.path.basename(multi_label))
        patients.append(os.path.basename(multi_label))
        sub_df = lesions_characteristics[lesions_characteristics['ID'] == os.path.basename(multi_label)[:-7]]
        image_4D_ar = sitk.GetArrayFromImage(sitk.ReadImage(image_4D))
        liver_mask_ar = sitk.GetArrayFromImage(sitk.ReadImage(liver_label))
        gt_mask_ar = sitk.GetArrayFromImage(sitk.ReadImage(label))
        multi_gt_ar = sitk.GetArrayFromImage(sitk.ReadImage(multi_label))
        if config["vein_mask_criterion"]:
            vein_mask_ar = sitk.GetArrayFromImage(sitk.ReadImage(vein_label))
        gt_tumor_idx = Counter(multi_gt_ar.ravel())
        del gt_tumor_idx[0]
        cc_filter = sitk.ConnectedComponentImageFilter()
        if threshold:
            prob_map_image = sitk.ReadImage(prob_map_lab)
            prob_vector = sitk.GetArrayFromImage(prob_map_image).ravel()
            mean_prob = np.sum(prob_vector) / len(prob_vector)
            mean_prob_map = np.mean(prob_vector)
            median_prob_map = np.median(prob_vector)
            prob_vector_more_02 = prob_vector[prob_vector > 0.01]

            vector_prob_map_stats = {'mean': mean_prob_map, 
                               'median': median_prob_map,
                               'P25': np.percentile(prob_vector, 25),
                               'P75': np.percentile(prob_vector, 75),
                               'P90': np.percentile(prob_vector, 90),
                               'P99': np.percentile(prob_vector, 99),
                               'P999': np.percentile(prob_vector, 99.9),
                               'P9999': np.percentile(prob_vector, 99.99),
                               'P99999': np.percentile(prob_vector, 99.999)}

            if not os.path.exists(os.path.dirname(os.path.dirname(prob_map_lab)) + f'/prob_map_hist/{os.path.basename(prob_map_lab)[:-7]}.png'):
                plt.figure()
                plt.hist(prob_vector_more_02, bins=200)
                plt.title('Histogram of the probability map')
                plt.xlabel('Probability')
                plt.ylabel('Frequency')
                plt.ylim(0, 10000)
                os.makedirs(os.path.dirname(os.path.dirname(prob_map_lab)) + '/prob_map_hist', exist_ok=True)
                plt.savefig(os.path.dirname(os.path.dirname(prob_map_lab)) + f'/prob_map_hist/{os.path.basename(prob_map_lab)[:-7]}.png')
                plt.close()
            blurred_map = sitk.SmoothingRecursiveGaussian(prob_map_image)
            prob_map_img = (blurred_map > threshold)
            prob_map_img = cc_filter.Execute(sitk.Cast(prob_map_img, sitk.sitkUInt8))

            lssif = sitk.LabelShapeStatisticsImageFilter()
            lssif.Execute(prob_map_img)
            labels = lssif.GetLabels()
            sitk_4D_image = sitk.ReadImage(image_4D)
            binary_mask_ar = sitk.GetArrayFromImage(prob_map_img)
            final_mask = sitk.Image(prob_map_img.GetSize(), sitk.sitkUInt8)
            final_mask.CopyInformation(prob_map_img)
            for lssif_label in labels:
                volume_mm3 = lssif.GetPhysicalSize(lssif_label)
                diameter = 2 * ((3 * volume_mm3) / (4 * np.pi))**(1/3)
                if diameter < min_diameter:
                    removed_small_components = sitk.BinaryThreshold(prob_map_img,
                                                                    lowerThreshold=lssif_label,
                                                                    upperThreshold=lssif_label,
                                                                    insideValue=1,
                                                                    outsideValue=0)
                    final_mask = sitk.Or(final_mask, removed_small_components)
                    continue
                if config["vein_mask_criterion"] and diameter > min_diameter:
                    binary_mask = binary_mask_ar.copy()
                    binary_mask = np.where(binary_mask == lssif_label, 1, 0)
                    overlap = binary_mask * vein_mask_ar
                    percentage = np.sum(overlap) / np.sum(binary_mask)
                    if percentage > config['vein_mask_threshold']:
                        remove_lesion = sitk.BinaryThreshold(prob_map_img,
                                                            lowerThreshold=lssif_label,
                                                            upperThreshold=lssif_label,
                                                            insideValue=1,
                                                            outsideValue=0)
                        final_mask = sitk.Or(final_mask, remove_lesion)
                    continue
                if diameter > min_diameter:
                    binary_mask = binary_mask_ar.copy()
                    binary_mask = np.where(binary_mask == lssif_label, 1, 0)

                        
            final_mask = sitk.Not(final_mask)
            prob_map_img = sitk.Mask(prob_map_img, final_mask)
            eroded_prob_map = sitk.BinaryErode(prob_map_img, kernelRadius=(1, 1, 1))
            dilated_prob_map = sitk.BinaryDilate(eroded_prob_map, kernelRadius=(1, 1, 1))
            dilated_prob_map.CopyInformation(sitk.ReadImage(liver_label))
            pred_mask_image = dilated_prob_map * sitk.ReadImage(liver_label, sitk.sitkUInt32)
            post_process_img_path = prob_map_lab.replace('/prob_map_tta', '/post_processed_prob_map_tta').replace('/calibrated_prob_map_logit_tta',
                                                                                                                '/post_processed_calibrated_prob_map_logit_tta')
            if threshold == 0.9999:
                os.makedirs(os.path.dirname(post_process_img_path), exist_ok=True)
                sitk.WriteImage(pred_mask_image, post_process_img_path[:-7] + f'_T0_9999_size_' + str(min_diameter) + '.nii.gz')
            pred_mask_image = cc_filter.Execute(dilated_prob_map)

        else:
            pred_mask_image = cc_filter.Execute(sitk.Cast(sitk.ReadImage(pred), sitk.sitkUInt8))

            lssif = sitk.LabelShapeStatisticsImageFilter()
            lssif.Execute(pred_mask_image)
            for lssif_label in lssif.GetLabels():
                volume_mm3 = lssif.GetPhysicalSize(lssif_label)
                diameter = 2 * ((3 * volume_mm3) / (4 * np.pi))**(1/3)
                if diameter < min_diameter:
                    removed_small_components = sitk.BinaryThreshold(pred_mask_image,
                                                                    lowerThreshold=lssif_label,
                                                                    upperThreshold=lssif_label,
                                                                    insideValue=0,
                                                                    outsideValue=1)
                    pred_mask_image = sitk.Mask(pred_mask_image, removed_small_components)

        nbr_of_preds = len(np.unique(sitk.GetArrayFromImage(pred_mask_image))) - 1
        binary_pred_mask_ar = sitk.GetArrayFromImage(pred_mask_image)
        binary_pred_mask_ar[binary_pred_mask_ar > 0] = 1
        pred_mask_ar_cc3d = cc3d.connected_components(binary_pred_mask_ar)


        if threshold:
            patient_tumor_metric = sg.write_metrics(labels=[1],
                                                    gdth_img=sitk.GetArrayFromImage(sitk.ReadImage(label)),
                                                    pred_img=binary_pred_mask_ar,
                                                    metrics='dice',
                                                    verbose=False)[0]['dice']
        else:
            patient_tumor_metric = sg.write_metrics(labels=[1],
                                                    gdth_img=sitk.GetArrayFromImage(sitk.ReadImage(label)),
                                                    pred_img=binary_pred_mask_ar,
                                                    metrics='dice',
                                                    verbose=False)[0]['dice']
        patient_tumor_metrics_list.append(patient_tumor_metric)

        for gt_idx in sorted(list(gt_tumor_idx.keys())):
            tumor_characteristics = sub_df[sub_df['label'] == gt_idx]
            LIRADS.append(tumor_characteristics['LIRADS'].values[0])
            HCC_lesions.append(tumor_characteristics['HCC'].values[0])
            wash_ven.append(tumor_characteristics['Venous washout'].values[0])
            wash_del.append(tumor_characteristics['Delayed washout'].values[0])
            caps_ven.append(tumor_characteristics['Venous capsule'].values[0])
            caps_del.append(tumor_characteristics['Delayed capsule'].values[0])
            hyper_art.append(tumor_characteristics['Arterial'].values[0])
            if 'Lesion_type' in tumor_characteristics.columns:
                lesion_type.append(tumor_characteristics['Lesion_type'].values[0])

            gt_binary_map = (multi_gt_ar == gt_idx).astype(np.uint8)
            touching_pred_label = Counter((gt_binary_map * binary_pred_mask_ar).ravel())
            del touching_pred_label[0]

            if len(touching_pred_label) == 0:
                FN.append(1)
                TP.append(0)
            else:
                FN.append(0)
                TP.append(1)
                pred_touching_label, pred_touching_label_cnt = np.unique((pred_mask_ar_cc3d*gt_binary_map), return_counts=True)
                pred_touching_label_cnt_idx = np.argmax(pred_touching_label_cnt[1:])
                pred_touching_label_mask = (pred_mask_ar_cc3d == pred_touching_label[pred_touching_label_cnt_idx + 1]).astype(np.uint8)
                tumor_wise_dice = sg.write_metrics(labels=[1],
                                                   gdth_img=gt_binary_map,
                                                   pred_img=pred_touching_label_mask,
                                                   metrics='dice',
                                                   verbose=False)[0]['dice']
                tumor_metrics_list.append(tumor_wise_dice)
                pred_mask_ar_cc3d_img = sitk.GetImageFromArray(pred_mask_ar_cc3d)
                pred_mask_ar_cc3d_img.CopyInformation(sitk.ReadImage(pred))
                sitk.WriteImage(pred_mask_ar_cc3d_img, os.getcwd() + '/pred_cc3d.nii.gz')
                gt_binary_map_img = sitk.GetImageFromArray(gt_binary_map)
                gt_binary_map_img.CopyInformation(sitk.ReadImage(label))
                sitk.WriteImage(gt_binary_map_img, os.getcwd() + '/gt_binary_map.nii.gz')


            lssif = sitk.LabelShapeStatisticsImageFilter()
            lssif.Execute(sitk.Cast(sitk.ReadImage(multi_label), sitk.sitkUInt8))
            bbox_coord = lssif.GetBoundingBox(int(gt_idx)) 
            bbox_shape = (bbox_coord[3], bbox_coord[4], bbox_coord[5])
            if bbox_shape[0] * bbox_shape[1] * bbox_shape[2] > 300000:
                bbox_shape = (int(bbox_shape[0] / 2), int(bbox_shape[1] / 2), int(bbox_shape[2] / 2))

            found_negative_sample = 0
            neg_samples = {}
            for neg_patch_trial in range(nbr_of_neg_patch_trials):
                dep = randint(0, gt_mask_ar.shape[0] - bbox_shape[2])
                if bbox_shape[1] > gt_mask_ar.shape[1]:
                    col = gt_mask_ar.shape[1]
                else:
                    col = randint(0, gt_mask_ar.shape[1] - bbox_shape[1])
                if bbox_shape[0] > gt_mask_ar.shape[2]:
                    row = gt_mask_ar.shape[2]
                else:
                    row = randint(0, gt_mask_ar.shape[2] - bbox_shape[0])

                neg_lab_patch = gt_mask_ar[dep:(dep + bbox_shape[2]), col:(col + bbox_shape[1]),
                                row:(row + bbox_shape[0])]
                neg_liver_patch = liver_mask_ar[dep:dep + bbox_shape[2], col:col + bbox_shape[1],
                                  row:row + bbox_shape[0]]
                tumor_count = np.sum(neg_lab_patch)
                liver_ratio = Counter(neg_liver_patch.ravel())
                if tumor_count == 0 and len(liver_ratio) > 1:
                    if liver_ratio[1] / (liver_ratio[0] + liver_ratio[1]) > 0.5:
                        if len(image_4D_ar.shape) == 4:
                            neg_img = image_4D_ar[0, dep:dep + bbox_shape[2], col:col + bbox_shape[1],
                                      row:row + bbox_shape[0]]
                        else:
                            neg_img = image_4D_ar[dep:dep + bbox_shape[2], col:col + bbox_shape[1],
                                      row:row + bbox_shape[0]]
                        neg_samples[dep, col, row] = np.max(neg_img)
                        found_negative_sample += 1

            if len(neg_samples) != 0:
                max_int_sample = max(neg_samples, key=neg_samples.get)
   
                neg_pred = binary_pred_mask_ar[max_int_sample[0]:max_int_sample[0] + bbox_shape[2],
                        max_int_sample[1]:max_int_sample[1] + bbox_shape[1],
                        max_int_sample[2]:max_int_sample[2] + bbox_shape[0]]
                generated_neg_samples.append(len(neg_samples))
            else:
                neg_pred = np.zeros(bbox_shape)
                generated_neg_samples.append(0)

            if np.sum(neg_pred) == 0:
                TN.append(1)

        LIRADS_list.append(LIRADS)
        HCC_list.append(HCC_lesions)
        wash_ven_list.append(wash_ven)
        wash_del_list.append(wash_del)
        caps_ven_list.append(caps_ven)
        caps_del_list.append(caps_del)
        hyper_art_list.append(hyper_art)
        lesion_type_list.append(lesion_type)
        prob_map_stats_list.append(vector_prob_map_stats)
        if len(np.unique(pred_mask_ar_cc3d)) == 1 and np.unique(pred_mask_ar_cc3d)[0] == 1:
            FP_preds = 0
        else:
            FP_preds = nbr_of_preds - (len(np.unique(pred_mask_ar_cc3d * gt_mask_ar)) - 1) # remove bakground label

        if FP_preds < 0:
            sitk.WriteImage(sitk.GetImageFromArray(pred_mask_ar_cc3d), os.getcwd() + '/pred_cc3d.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(gt_mask_ar), os.getcwd() + '/gt_mask.nii.gz')
            raise ValueError('Negative FP preds found: check gt and pred masks save in current directory')

        FP.append(FP_preds)
        TN_list.append(TN)
        TP_list.append(TP)
        FP_list.append(FP_preds)
        FN_list.append(FN)
        preds_nbr.append(nbr_of_preds)
    return {'TP_list': TP_list, 'FP_list': FP_list, 'FN_list': FN_list, 'TN_list': TN_list, 'Number of preds': preds_nbr, 
            'patients': patients, 'patient_tumor_metrics_list': patient_tumor_metrics_list, 'tumor_metrics_list': tumor_metrics_list,
            'generated_neg_samples': generated_neg_samples, 'LIRADS_list': LIRADS_list, 'HCC_list': HCC_list, 'wash_ven_list': wash_ven_list,
            'wash_del_list': wash_del_list, 'caps_ven_list': caps_ven_list, 'caps_del_list': caps_del_list,
            'hyper_art_list': hyper_art_list, 'lesion_type_list': lesion_type_list, 'prob_map_stats_list': prob_map_stats_list}


def contrast_check(image, label):
    """
    Calculate the contrast between the inside and around the tumor for nat, art, ven, and del.
    Parameters:
        image: 4D image with shape (C, D, H, W) where C is the number of channels
        label: 3D label image with shape (D, H, W) where each pixel is labeled as nat, art, ven, or del
    Returns:
        contrast_nat: contrast between nat inside and around the tumor
        contrast_art: contrast between art inside and around the tumor
        contrast_ven: contrast between ven inside and around the tumor
        contrast_del: contrast between del inside and around the tumor
    """
    inside_tumor = sitk.GetArrayFromImage(image) * sitk.GetArrayFromImage(label)
    dilated_tumor_large = sitk.BinaryDilate(label, (10, 10, 10))
    dilated_tumor_small = sitk.BinaryDilate(label, (3, 3, 3))
    around_tumor_large = sitk.GetArrayFromImage(image) * sitk.GetArrayFromImage(dilated_tumor_large)
    around_tumor_small = sitk.GetArrayFromImage(image) * sitk.GetArrayFromImage(dilated_tumor_small)
    around_tumor = around_tumor_large - around_tumor_small

    if sum(inside_tumor[1, :, :, :].ravel()) == 0:
        return 0, 0, 0, 0
    
    nat_perc_25_75_inside = np.percentile(inside_tumor[0, :, :, :][inside_tumor[0, :, :, :] != 0].ravel(), (25, 75))
    nat_perc_25_75_outside = np.percentile(around_tumor[0, :, :, :][around_tumor[0, :, :, :] != 0].ravel(), (25, 75))
    art_perc_25_75_inside = np.percentile(inside_tumor[1, :, :, :][inside_tumor[1, :, :, :] != 0].ravel(), (25, 75))
    art_perc_25_75_outside = np.percentile(around_tumor[1, :, :, :][around_tumor[1, :, :, :] != 0].ravel(), (25, 75))
    ven_perc_25_75_inside = np.percentile(inside_tumor[2, :, :, :][inside_tumor[2, :, :, :] != 0].ravel(), (25, 75))
    ven_perc_25_75_outside = np.percentile(around_tumor[2, :, :, :][around_tumor[2, :, :, :] != 0].ravel(), (25, 75))
    del_perc_25_75_inside = np.percentile(inside_tumor[2, :, :, :][inside_tumor[2, :, :, :] != 0].ravel(), (25, 75))
    del_perc_25_75_outside = np.percentile(around_tumor[2, :, :, :][around_tumor[2, :, :, :] != 0].ravel(), (25, 75))

    nat_inside_mean = np.mean(inside_tumor[0, :, :, :][(inside_tumor[0, :, :, :] > nat_perc_25_75_inside[0]) & (inside_tumor[0, :, :, :] < nat_perc_25_75_inside[1])])
    nat_outside_mean = np.mean(around_tumor[0, :, :, :][(around_tumor[0, :, :, :] > nat_perc_25_75_outside[0]) & (around_tumor[0, :, :, :] < nat_perc_25_75_outside[1])])
    art_inside_mean = np.mean(inside_tumor[1, :, :, :][(inside_tumor[1, :, :, :] > art_perc_25_75_inside[0]) & (inside_tumor[1, :, :, :] < art_perc_25_75_inside[1])])
    art_outside_mean = np.mean(around_tumor[1, :, :, :][(around_tumor[1, :, :, :] > art_perc_25_75_outside[0]) & (around_tumor[1, :, :, :] < art_perc_25_75_outside[1])])
    ven_inside_mean = np.mean(inside_tumor[2, :, :, :][(inside_tumor[2, :, :, :] > ven_perc_25_75_inside[0]) & (inside_tumor[2, :, :, :] < ven_perc_25_75_inside[1])])
    ven_outside_mean = np.mean(around_tumor[2, :, :, :][(around_tumor[2, :, :, :] > ven_perc_25_75_outside[0]) & (around_tumor[2, :, :, :] < ven_perc_25_75_outside[1])])
    del_inside_mean = np.mean(inside_tumor[2, :, :, :][(inside_tumor[2, :, :, :] > del_perc_25_75_inside[0]) & (inside_tumor[2, :, :, :] < del_perc_25_75_inside[1])])
    del_outside_mean = np.mean(around_tumor[2, :, :, :][(around_tumor[2, :, :, :] > del_perc_25_75_outside[0]) & (around_tumor[2, :, :, :] < del_perc_25_75_outside[1])])

    contrast_nat = nat_inside_mean/ nat_outside_mean
    contrast_art = art_inside_mean/ art_outside_mean
    contrast_ven = ven_inside_mean/ ven_outside_mean
    contrast_del = del_inside_mean/ del_outside_mean

    return contrast_nat, contrast_art, contrast_ven, contrast_del
