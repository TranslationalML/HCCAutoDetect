import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from collections import defaultdict

def get_patient_metrics(FROC_metrics_init, patient_selection):
    """
    Function to get the metrics per patient for the FROC metrics.
    Parameters:
        FROC_metrics_init: dict, FROC metrics
            key: float, threshold
            value: dict, patient metrics
                key: str, metric name
                value: list, metric values
                    # examples are for 2 patients having 3 lesions each
                    TP_list: list, TP per patient (e.g. [[1, 0, 1], [0, 1, 0]] for 2 patients)
                    FP_list: list, FP per patient (e.g. [0, 1] for 2 patients)
                    FN_list: list, FN per patient (e.g. [[0, 1], [1, 0]] for 2 patients with each having 1 FN)
                    TN_list: list, TN per patient (e.g. [1, 0] for 2 patients)
                    patients: list, patient names (e.g. ['sub-001.nii.gz', 'sub-002.nii.gz'])
                    patient_tumor_metrics_list: list, patient-wise (can include multiple tumors in a single mask) 
                                                        tumor dice with binary prediction and ground truth masks (e.g. [[0.5], [0.6]] for 2 patients)
                    tumor_metrics_listlist, tumor-wise (include only 1 tumor in the mask at a time) dice metrics (e.g. [[0.5], [0.6], [0.7], [0.8], [0.9], [0.4]]
                                                        for 2 patients having 3 lesions each)
                    generated_neg_sampleslist, number of generated negative samples (e.g. [100, 200] for 2 patients)
                    LIRADS_listlist, LIRADS per lesion (e.g. [['5', '4', '5'], ['4', '5', '4']] for 2 patients having LR 5, 4, 5 and 4, 5, 4)
                    wash_ven_listlist, washin venous per lesion (e.g. [[1, 0, 1], [0, 1, 0]] for 2 patients with 1 washout and 1 non-washout)
                    wash_del_listlist, washin delayed per lesion
                    caps_ven_listlist, capsular venous per lesion (e.g. [[1, 0, 1], [0, 1, 0]] for 2 patients with 1 capsular venous and 1 non-capsular venous)
                    caps_del_listlist, capsular delayed per lesion
                    hyper_art_listlist, hyper arterial per lesion
                    lesion_type_list: list, lesion type per lesion
        patient_selection: dict, selection of patients to consider
    Returns: 
        dict, patient metrics
    """

    if patient_selection is not None:
        if any([isinstance(patient, dict) for patient in list(patient_selection.values())]):
            remaining_patients = [patient for patient in list(patient_selection.keys()) if not all([x is None for x in patient_selection[patient].values()])]
            patient_selection_remaining = {patient: patient_selection[patient] for patient in remaining_patients}
            FROC_metrics_init = {threshold: {metric_name: [metric_val for patient, metric_val in zip(metric_dict_threshold['patients'], metric_dict_threshold[metric_name]) if patient[:-7] in remaining_patients]
                                        for metric_name in metric_dict_threshold.keys()} for threshold, metric_dict_threshold in FROC_metrics_init.items()}

            metrics_not_lesion_wise = ['FP_list', 'generated_neg_samples', 'patients', 'patient_tumor_metrics_list', 'tumor_metrics_list', 'Number of preds']
            FROC_metrics_init = {threshold: {metric_name: [[lesion_metric_val
                                                                for lesion_metric_val, lesion_patient_select_val in zip(patient_metric_val_list, list(patient_select_val.values())) if lesion_patient_select_val is not None] if metric_name not in metrics_not_lesion_wise else patient_metric_val_list
                                                                for patient_metric_val_list, patient_select_val in zip(metric_dict_threshold[metric_name], list(patient_selection_remaining.values()))]
                                                for metric_name in metric_dict_threshold.keys()} 
                                    for threshold, metric_dict_threshold in FROC_metrics_init.items()}
        elif type(patient_selection) == dict:
            FROC_metrics_init = {threshold: {metric_name: [metric_val for patient, metric_val in zip(metric_dict_threshold['patients'], metric_dict_threshold[metric_name]) 
                                                           if patient[:-7] in patient_selection]
                            for metric_name in metric_dict_threshold.keys()} for threshold, metric_dict_threshold in FROC_metrics_init.items()}
        else:
            raise ValueError(f"Patient selection should be a dictionary, not {type(patient_selection)}")

    random_threshold = list(FROC_metrics_init.keys())[0]
    patient_list = FROC_metrics_init[random_threshold]['patients']
    TP_per_patient = {threshold: [sum(TP) for TP in metric_dict['TP_list']] for threshold, metric_dict in FROC_metrics_init.items()}
    FP_per_patient = {threshold: [FP for FP in metric_dict['FP_list']] for threshold, metric_dict in FROC_metrics_init.items()}
    FN_per_patient = {threshold: [sum(FN) for FN in metric_dict['FN_list']] for threshold, metric_dict in FROC_metrics_init.items()}
    
    TP_lesions_per_patient = {threshold: float(sum([np.sum(TP) if np.sum(TP) > 0 else 0 for patient, TP in zip(metric_dict['patients'], metric_dict['TP_list'])])) 
                for threshold, metric_dict in FROC_metrics_init.items()}
    FN_lesions_per_patient = {threshold: float(sum([np.sum(FN) if np.sum(FN) > 0 else 0 for patient, FN in zip(metric_dict['patients'], metric_dict['FN_list'])]))
                    for threshold, metric_dict in FROC_metrics_init.items()}

    lesions_recall = {key_threshold_TP: float(np.divide(float(TP), float(TP) + float(FN), out=np.zeros_like(float(TP)),
                        where=(float(TP) + float(FN)) != 0)) for (key_threshold_TP, TP), (key_threshold_FN, FN) in 
                        zip(TP_lesions_per_patient.items(), FN_lesions_per_patient.items())}
    mean_FP_per_patient_per_threshold = {threshold: np.mean(FP) for threshold, FP in FP_per_patient.items()}
    

    TP_flat_per_threshold = {threshold: [TP_lesion for TP in metric_dict['TP_list'] for TP_lesion in TP] for threshold, metric_dict in FROC_metrics_init.items()}
    TP_per_patient = {threshold: metric_dict['TP_list'] for threshold, metric_dict in FROC_metrics_init.items()}
    FP_per_patient = {threshold: metric_dict['FP_list'] for threshold, metric_dict in FROC_metrics_init.items()}
    HCC_flat_per_threshold = {threshold: [LR_lesion for LR in metric_dict['HCC_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_flat_per_threshold = {threshold: [LR_lesion for LR in metric_dict['LIRADS_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}
    wash_ven_flat_per_threshold = {threshold: [LR_lesion for LR in metric_dict['wash_ven_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}
    wash_del_flat_per_threshold = {threshold: [LR_lesion for LR in metric_dict['wash_del_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}
    caps_ven_flat_per_threshold = {threshold: [LR_lesion for LR in metric_dict['caps_ven_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}
    caps_del_flat_per_threshold = {threshold: [LR_lesion for LR in metric_dict['caps_del_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}
    hyper_art = {threshold: [LR_lesion for LR in metric_dict['hyper_art_list'] for LR_lesion in LR] for threshold, metric_dict in FROC_metrics_init.items()}

    det_wash_ven_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 1 else 0 for TP_lesion, LR_lesion in zip(TP_flat_per_threshold[threshold], wash_ven_flat_per_threshold[threshold])] for threshold in TP_flat_per_threshold.keys()}
    det_wash_del_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 1 else 0 for TP_lesion, LR_lesion in zip(TP_flat_per_threshold[threshold], wash_del_flat_per_threshold[threshold])] for threshold in TP_flat_per_threshold.keys()}
    det_caps_ven_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 1 else 0 for TP_lesion, LR_lesion in zip(TP_flat_per_threshold[threshold], caps_ven_flat_per_threshold[threshold])] for threshold in TP_flat_per_threshold.keys()}
    det_caps_del_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 1 else 0 for TP_lesion, LR_lesion in zip(TP_flat_per_threshold[threshold], caps_del_flat_per_threshold[threshold])] for threshold in TP_flat_per_threshold.keys()}
    det_hyper_art_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 1 else 0 for TP_lesion, LR_lesion in zip(TP_flat_per_threshold[threshold], hyper_art[threshold])] for threshold in TP_flat_per_threshold.keys()}
    wash_ven_tot = np.sum([1 for LR_lesion in wash_ven_flat_per_threshold[random_threshold] if LR_lesion == 1])
    wash_del_tot = np.sum([1 for LR_lesion in wash_del_flat_per_threshold[random_threshold] if LR_lesion == 1])
    caps_ven_tot = np.sum([1 for LR_lesion in caps_ven_flat_per_threshold[random_threshold] if LR_lesion == 1])
    caps_del_tot = np.sum([1 for LR_lesion in caps_del_flat_per_threshold[random_threshold] if LR_lesion == 1])
    hyper_art_tot = np.sum([1 for LR_lesion in hyper_art[random_threshold] if LR_lesion == 1])

    detected_HCC_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 1 else 0 for TP_lesion, LR_lesion 
                                               in zip(TP_flat_per_threshold[threshold], HCC_flat_per_threshold[threshold])] 
                                               for threshold in TP_flat_per_threshold.keys()}
    detected_LR_5_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == '5' else 0 for TP_lesion, LR_lesion 
                                               in zip(TP_flat_per_threshold[threshold], LR_flat_per_threshold[threshold])] 
                                               for threshold in TP_flat_per_threshold.keys()}
    detected_LR_4_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == '4' else 0 for TP_lesion, LR_lesion
                                                  in zip(TP_flat_per_threshold[threshold], LR_flat_per_threshold[threshold])] 
                                                  for threshold in TP_flat_per_threshold.keys()}
    detected_LR_3_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == '3' else 0 for TP_lesion, LR_lesion
                                                    in zip(TP_flat_per_threshold[threshold], LR_flat_per_threshold[threshold])] 
                                                    for threshold in TP_flat_per_threshold.keys()}
    detected_LR_TIV_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 'TIV' else 0 for TP_lesion, LR_lesion
                                                        in zip(TP_flat_per_threshold[threshold], LR_flat_per_threshold[threshold])] 
                                                        for threshold in TP_flat_per_threshold.keys()}
    detected_LR_M_per_threshold = {threshold: [1 if TP_lesion > 0 and LR_lesion == 'M' else 0 for TP_lesion, LR_lesion
                                                  in zip(TP_flat_per_threshold[threshold], LR_flat_per_threshold[threshold])] 
                                                  for threshold in TP_flat_per_threshold.keys()}

    HCC_tot = np.sum([1 for LR_lesion in HCC_flat_per_threshold[random_threshold] if LR_lesion == 1])
    LR_5_tot = np.sum([1 for LR_lesion in LR_flat_per_threshold[random_threshold] if LR_lesion == '5'])
    LR_4_tot = np.sum([1 for LR_lesion in LR_flat_per_threshold[random_threshold] if LR_lesion == '4'])
    LR_3_tot = np.sum([1 for LR_lesion in LR_flat_per_threshold[random_threshold] if LR_lesion == '3'])
    LR_TIV_tot = np.sum([1 for LR_lesion in LR_flat_per_threshold[random_threshold] if LR_lesion == 'TIV'])
    LR_M_tot = np.sum([1 for LR_lesion in LR_flat_per_threshold[random_threshold] if LR_lesion == 'M'])
        
    total_lesions_per_patient = {threshold: [len(TP) for TP in metric_dict['TP_list']] for threshold, metric_dict in FROC_metrics_init.items()}
    recall_per_patient = {threshold: [sum(TP)/total_lesions_per_patient[threshold][patient_idx] if total_lesions_per_patient[threshold][patient_idx] != 0 else None
                            for patient_idx, TP in enumerate(metric_dict['TP_list'])] 
                            for threshold, metric_dict in FROC_metrics_init.items()}
    
    LR_5_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == '5']) for LR in metric_dict['LIRADS_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_4_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == '4']) for LR in metric_dict['LIRADS_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_3_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == '3']) for LR in metric_dict['LIRADS_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_TIV_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 'TIV']) for LR in metric_dict['LIRADS_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_M_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 'M']) for LR in metric_dict['LIRADS_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    wash_ven_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 1]) for LR in metric_dict['wash_ven_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    wash_del_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 1]) for LR in metric_dict['wash_del_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    caps_ven_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 1]) for LR in metric_dict['caps_ven_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    caps_del_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 1]) for LR in metric_dict['caps_del_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}
    hyper_art_tot_per_patient = {threshold: [len([LR_lesion for LR_lesion in LR if LR_lesion == 1]) for LR in metric_dict['hyper_art_list'] ] for threshold, metric_dict in FROC_metrics_init.items()}

    FP_HCC_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion == 1 and LR_lesion != 1]) for TP, LR in zip(metric_dict['TP_list'], metric_dict['HCC_list'])] for threshold, metric_dict in FROC_metrics_init.items()}

    LR_5_detected_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == '5']) for TP, LR in zip(metric_dict['TP_list'], metric_dict['LIRADS_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_4_detected_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == '4']) for TP, LR in zip(metric_dict['TP_list'], metric_dict['LIRADS_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_3_detected_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == '3']) for TP, LR in zip(metric_dict['TP_list'], metric_dict['LIRADS_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_TIV_detected_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 'TIV']) for TP, LR in zip(metric_dict['TP_list'], metric_dict['LIRADS_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    LR_M_detected_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 'M']) for TP, LR in zip(metric_dict['TP_list'], metric_dict['LIRADS_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    det_wash_ven_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 1]) for TP, LR in zip(metric_dict['TP_list'], metric_dict['wash_ven_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    det_wash_del_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 1]) for TP, LR in zip(metric_dict['TP_list'], metric_dict['wash_del_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    det_caps_ven_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 1]) for TP, LR in zip(metric_dict['TP_list'], metric_dict['caps_ven_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    det_caps_del_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 1]) for TP, LR in zip(metric_dict['TP_list'], metric_dict['caps_del_list'])] for threshold, metric_dict in FROC_metrics_init.items()}
    det_hyper_art_per_patient = {threshold: [len([LR_lesion for TP_lesion, LR_lesion in zip(TP, LR) if TP_lesion > 0 and LR_lesion == 1]) for TP, LR in zip(metric_dict['TP_list'], metric_dict['hyper_art_list'])] for threshold, metric_dict in FROC_metrics_init.items()}

    recall_LR_5_per_patient = {LR_5_det_threshold_key: [detected_LR_5/tot_LR_5 if tot_LR_5 != 0 else None for detected_LR_5, tot_LR_5 in zip(LR_5_det_threshold_val, LR_5_tot_threshold_val) ] 
                                for (LR_5_det_threshold_key, LR_5_det_threshold_val), (LR_5_tot_threshold_key, LR_5_tot_threshold_val) in zip(LR_5_detected_per_patient.items(), LR_5_tot_per_patient.items())}
    recall_LR_4_per_patient = {LR_4_det_threshold_key: [detected_LR_4/tot_LR_4 if tot_LR_4 != 0 else None for detected_LR_4, tot_LR_4 in zip(LR_4_det_threshold_val, LR_4_tot_threshold_val) ]
                                    for (LR_4_det_threshold_key, LR_4_det_threshold_val), (LR_4_tot_threshold_key, LR_4_tot_threshold_val) in zip(LR_4_detected_per_patient.items(), LR_4_tot_per_patient.items())}
    recall_LR_3_per_patient = {LR_3_det_threshold_key: [detected_LR_3/tot_LR_3 if tot_LR_3 != 0 else None for detected_LR_3, tot_LR_3 in zip(LR_3_det_threshold_val, LR_3_tot_threshold_val) ]
                                    for (LR_3_det_threshold_key, LR_3_det_threshold_val), (LR_3_tot_threshold_key, LR_3_tot_threshold_val) in zip(LR_3_detected_per_patient.items(), LR_3_tot_per_patient.items())}
    recall_LR_TIV_per_patient = {LR_TIV_det_threshold_key: [detected_LR_TIV/tot_LR_TIV if tot_LR_TIV != 0 else None for detected_LR_TIV, tot_LR_TIV in zip(LR_TIV_det_threshold_val, LR_TIV_tot_threshold_val) ]
                                    for (LR_TIV_det_threshold_key, LR_TIV_det_threshold_val), (LR_TIV_tot_threshold_key, LR_TIV_tot_threshold_val) in zip(LR_TIV_detected_per_patient.items(), LR_TIV_tot_per_patient.items())} 
    recall_LR_M_per_patient = {LR_M_det_threshold_key: [detected_LR_M/tot_LR_M if tot_LR_M != 0 else None for detected_LR_M, tot_LR_M in zip(LR_M_det_threshold_val, LR_M_tot_threshold_val) ]
                                    for (LR_M_det_threshold_key, LR_M_det_threshold_val), (LR_M_tot_threshold_key, LR_M_tot_threshold_val) in zip(LR_M_detected_per_patient.items(), LR_M_tot_per_patient.items())}
    recall_ven_wash_per_patient = {det_wash_ven_threshold_key: [det_wash_ven/tot_wash_ven if tot_wash_ven != 0 else None for det_wash_ven, tot_wash_ven in zip(det_wash_ven_threshold_val, wash_ven_tot_threshold_val) ]
                                    for (det_wash_ven_threshold_key, det_wash_ven_threshold_val), (wash_ven_tot_threshold_key, wash_ven_tot_threshold_val) in zip(det_wash_ven_per_patient.items(), wash_ven_tot_per_patient.items())}
    recall_del_wash_per_patient = {det_wash_del_threshold_key: [det_wash_del/tot_wash_del if tot_wash_del != 0 else None for det_wash_del, tot_wash_del in zip(det_wash_del_threshold_val, wash_del_tot_threshold_val) ]
                                    for (det_wash_del_threshold_key, det_wash_del_threshold_val), (wash_del_tot_threshold_key, wash_del_tot_threshold_val) in zip(det_wash_del_per_patient.items(), wash_del_tot_per_patient.items())}
    recall_ven_caps_per_patient = {det_caps_ven_threshold_key: [det_caps_ven/tot_caps_ven if tot_caps_ven != 0 else None for det_caps_ven, tot_caps_ven in zip(det_caps_ven_threshold_val, caps_ven_tot_threshold_val) ]
                                    for (det_caps_ven_threshold_key, det_caps_ven_threshold_val), (caps_ven_tot_threshold_key, caps_ven_tot_threshold_val) in zip(det_caps_ven_per_patient.items(), caps_ven_tot_per_patient.items())}
    recall_del_caps_per_patient = {det_caps_del_threshold_key: [det_caps_del/tot_caps_del if tot_caps_del != 0 else None for det_caps_del, tot_caps_del in zip(det_caps_del_threshold_val, caps_del_tot_threshold_val) ]
                                    for (det_caps_del_threshold_key, det_caps_del_threshold_val), (caps_del_tot_threshold_key, caps_del_tot_threshold_val) in zip(det_caps_del_per_patient.items(), caps_del_tot_per_patient.items())}
    recall_hyper_art_per_patient = {det_hyper_art_threshold_key: [det_hyper_art/tot_hyper_art if tot_hyper_art != 0 else None for det_hyper_art, tot_hyper_art in zip(det_hyper_art_threshold_val, hyper_art_tot_threshold_val) ]
                                    for (det_hyper_art_threshold_key, det_hyper_art_threshold_val), (hyper_art_tot_threshold_key, hyper_art_tot_threshold_val) in zip(det_hyper_art_per_patient.items(), hyper_art_tot_per_patient.items())}

    recall_HCC_per_threshold = {threshold: np.sum(det_HCC) / HCC_tot for threshold, det_HCC in detected_HCC_per_threshold.items()}
    recall_LR_5_per_threshold = {threshold: np.sum(detected_LR_5) / LR_5_tot for threshold, detected_LR_5 in detected_LR_5_per_threshold.items()}
    recall_LR_4_per_threshold = {threshold: np.sum(detected_LR_4) / LR_4_tot for threshold, detected_LR_4 in detected_LR_4_per_threshold.items()}
    recall_LR_3_per_threshold = {threshold: np.sum(detected_LR_3) / LR_3_tot for threshold, detected_LR_3 in detected_LR_3_per_threshold.items()}
    recall_LR_TIV_per_threshold = {threshold: np.sum(detected_LR_TIV) / LR_TIV_tot for threshold, detected_LR_TIV in detected_LR_TIV_per_threshold.items()}
    recall_LR_M_per_threshold = {threshold: np.sum(detected_LR_M) / LR_M_tot for threshold, detected_LR_M in detected_LR_M_per_threshold.items()}
    recall_ven_wash_per_threshold = {threshold: np.sum(det_wash_ven) / wash_ven_tot for threshold, det_wash_ven in det_wash_ven_per_threshold.items()}
    recall_del_wash_per_threshold = {threshold: np.sum(det_wash_del) / wash_del_tot for threshold, det_wash_del in det_wash_del_per_threshold.items()}
    recall_ven_caps_per_threshold = {threshold: np.sum(det_caps_ven) / caps_ven_tot for threshold, det_caps_ven in det_caps_ven_per_threshold.items()}
    recall_del_caps_per_threshold = {threshold: np.sum(det_caps_del) / caps_del_tot for threshold, det_caps_del in det_caps_del_per_threshold.items()}
    recall_hyper_art_per_threshold = {threshold: np.sum(det_hyper_art) / hyper_art_tot for threshold, det_hyper_art in det_hyper_art_per_threshold.items()}

    total_FP_HCC_wise = {threshold: [FP_HCC_elem + FP_all_elem for FP_HCC_elem, FP_all_elem in zip(FP_HCC, FP_all)
                                    ] for (threshold, FP_HCC), (threshold, FP_all) in zip(FP_HCC_per_patient.items(), FP_per_patient.items())}
    mean_FP_HCC_per_patient_per_threshold = {threshold: np.mean(FP_HCC) for threshold, FP_HCC in total_FP_HCC_wise.items()}

    patient_dice = {threshold: FROC_list['patient_tumor_metrics_list'] for threshold, FROC_list in FROC_metrics_init.items()}
       
    healthy_wrongly_predicted = [[patient for patient, TP in zip(metric_dict['patients'], metric_dict['TP_list']) if np.sum(TP) > 0] 
                                 for threshold, metric_dict in FROC_metrics_init.items()]

    return {'patient_name': patient_list,
            'lesions_recall': lesions_recall,
            'TP_per_patient': TP_per_patient,
            'FP_mean_per_patient_per_threshold': mean_FP_per_patient_per_threshold,
            'recall_per_patient': recall_per_patient,
            'healthy_wrongly_predicted': healthy_wrongly_predicted,
            'recall_HCC': recall_HCC_per_threshold,
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
            'FP_HCC_mean_per_patient_per_threshold': mean_FP_HCC_per_patient_per_threshold,
            'TP_per_patient': TP_per_patient,
            'FP_per_patient': FP_per_patient,
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
            'Dice_per_patient': patient_dice,
            'FROC_metrics_init': FROC_metrics_init,
            } 



def run_permutation_test(model_1, model_2, recall_key, thresh_list, permutation_nbr, patient_selection, nbr_of_x_points):
    """
    Run a permutation test to compare the AUC of two models based on FROC metrics.
    Parameters:
        model_1: dict, FROC metrics for the first model
        model_2: dict, FROC metrics for the second model
        recall_key: str, key to access recall data in the models
        thresh_list: list, list of thresholds to consider
        permutation_nbr: int, number of permutations to run
        patient_selection: dict, selection of patients to consider
        nbr_of_x_points: int, number of x points for sorting data
    Returns:
        p_value: float, p-value from the permutation test
    """
    recall_model_1, mean_FP_model_1 = ([model_1[recall_key][threshold] for threshold in thresh_list], 
                                     [model_1['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list])
    recall_model_2, mean_FP_model_2 = ([model_2[recall_key][threshold] for threshold in thresh_list], 
                                               [model_2['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list])

    recall_model_1_per_threshold, mean_FP_model_1_per_threshold = sort_data_for_AUC(recall_model_1, mean_FP_model_1, nbr_of_x_points)
    recall_model_2_per_threshold, mean_FP_model_2_per_threshold = sort_data_for_AUC(recall_model_2, mean_FP_model_2, nbr_of_x_points)

    auc_model_1 = AUC_FROC_curve(recall_model_1_per_threshold, mean_FP_model_1_per_threshold)
    auc_model_2 = AUC_FROC_curve(recall_model_2_per_threshold, mean_FP_model_2_per_threshold)

    diff_obs_original = auc_model_2 - auc_model_1

    count = 0
    AUC_diff_list = []
    for i in range(permutation_nbr):
        if i % 100 == 0:
            print(f'Permutation ({recall_key}){i}/{permutation_nbr}')

        (FROC_metrics_init_1, FROC_metrics_init_2) = permutation_test_swap(model_1['FROC_metrics_init'], 
                                                                           model_2['FROC_metrics_init'],
                                                                           thresh_list)
        FROC_metrics_dict_1 = get_patient_metrics(FROC_metrics_init_1, patient_selection)
        FROC_metrics_dict_2 = get_patient_metrics(FROC_metrics_init_2, patient_selection)

        model_1_mean_FP_per_patient_swap = [FROC_metrics_dict_1['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        model_1_recall_swap = [FROC_metrics_dict_1[recall_key][threshold] for threshold in thresh_list]

        model_2_mean_FP_per_patient_swap = [FROC_metrics_dict_2['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        model_2_recall_swap = [FROC_metrics_dict_2[recall_key][threshold] for threshold in thresh_list]
        
        model_1_recall_per_threshold, model_1_mean_FP_per_threshold = sort_data_for_AUC(model_1_recall_swap, model_1_mean_FP_per_patient_swap, nbr_of_x_points) 
        model_2_recall_per_threshold, model_2_mean_FP_per_threshold = sort_data_for_AUC(model_2_recall_swap, model_2_mean_FP_per_patient_swap, nbr_of_x_points)

        auc_model_1 = AUC_FROC_curve(model_1_recall_per_threshold, model_1_mean_FP_per_threshold)
        auc_model_2 = AUC_FROC_curve(model_2_recall_per_threshold, model_2_mean_FP_per_threshold)

        diff_permutation = auc_model_2 - auc_model_1
        AUC_diff_list.append(diff_permutation)
        if diff_permutation >= diff_obs_original:
            count += 1
        
    p_value = (count + 1)/(permutation_nbr + 1)
    print(f'Permutation test p-value: {p_value}')
    return p_value


def permutation_test_swap(FROC_metrics_init_1, FROC_metrics_init_2, thresh_list):
    """
    Swap the TP, FP, and FN lists between two sets of FROC metrics based on a random mask.
    Parameters:
        FROC_metrics_init_1: dict, FROC metrics for the first model
        FROC_metrics_init_2: dict, FROC metrics for the second model
        thresh_list: list, list of thresholds to consider
    Returns:
        FROC_metrics_init_1: dict, updated FROC metrics for the first model
        FROC_metrics_init_2: dict, updated FROC metrics for the second model

    """
    FROC_metrics_init_1 = {threshold: values for threshold, values in FROC_metrics_init_1.items() if threshold in thresh_list}
    FROC_metrics_init_2 = {threshold: values for threshold, values in FROC_metrics_init_2.items() if threshold in thresh_list}

    random_threshold = list(FROC_metrics_init_1.keys())[0]
    swap_mask = np.random.rand(len(FROC_metrics_init_1[random_threshold]['TP_list'])) < 0.5

    FROC_metrics_init_swap_1_dict_TP = {threshold: {'TP_list': [FROC_metrics_init_1[threshold]['TP_list'][idx] 
                                                              if not swap else FROC_metrics_init_2[threshold]['TP_list'][idx] 
                                for idx, swap in enumerate(swap_mask)]} for threshold in FROC_metrics_init_1.keys()}
    FROC_metrics_init_swap_1_dict_FP = {threshold: {'FP_list': [FROC_metrics_init_1[threshold]['FP_list'][idx] 
                                                              if not swap else FROC_metrics_init_2[threshold]['FP_list'][idx]
                                for idx, swap in enumerate(swap_mask)]} for threshold in FROC_metrics_init_1.keys()}
    FROC_metrics_init_swap_1_dict_FN = {threshold: {'FN_list': [FROC_metrics_init_1[threshold]['FN_list'][idx] 
                                                              if not swap else FROC_metrics_init_2[threshold]['FN_list'][idx]
                                for idx, swap in enumerate(swap_mask)]} for threshold in FROC_metrics_init_1.keys()}
    
    FROC_metrics_init_swap_2_dict_TP = {threshold: {'TP_list': [FROC_metrics_init_2[threshold]['TP_list'][idx] 
                                                                   if not swap else FROC_metrics_init_1[threshold]['TP_list'][idx]
                                for idx, swap in enumerate(swap_mask)]} for threshold in FROC_metrics_init_1.keys()}
    FROC_metrics_init_swap_2_dict_FP = {threshold: {'FP_list': [FROC_metrics_init_2[threshold]['FP_list'][idx]
                                                                     if not swap else FROC_metrics_init_1[threshold]['FP_list'][idx]
                                for idx, swap in enumerate(swap_mask)]} for threshold in FROC_metrics_init_1.keys()}
    FROC_metrics_init_swap_2_dict_FN = {threshold: {'FN_list': [FROC_metrics_init_2[threshold]['FN_list'][idx]
                                                                        if not swap else FROC_metrics_init_1[threshold]['FN_list'][idx]
                                for idx, swap in enumerate(swap_mask)]} for threshold in FROC_metrics_init_1.keys()}

    for threshold in FROC_metrics_init_1.keys():
        FROC_metrics_init_1[threshold]['TP_list'] = FROC_metrics_init_swap_1_dict_TP[threshold]['TP_list']
        FROC_metrics_init_1[threshold]['FP_list'] = FROC_metrics_init_swap_1_dict_FP[threshold]['FP_list']
        FROC_metrics_init_1[threshold]['FN_list'] = FROC_metrics_init_swap_1_dict_FN[threshold]['FN_list']

        FROC_metrics_init_2[threshold]['TP_list'] = FROC_metrics_init_swap_2_dict_TP[threshold]['TP_list']
        FROC_metrics_init_2[threshold]['FP_list'] = FROC_metrics_init_swap_2_dict_FP[threshold]['FP_list']
        FROC_metrics_init_2[threshold]['FN_list'] = FROC_metrics_init_swap_2_dict_FN[threshold]['FN_list']

    return FROC_metrics_init_1, FROC_metrics_init_2


def sort_data_for_AUC(recall_per_threshold, mean_FP_per_threshold, nbr_of_x_points):
    """
    Sort the recall and mean FP values for AUC calculation.
    Parameters:
        recall_per_threshold: list, recall values from smallest to biggest
        mean_FP_per_threshold: list, mean FP values from smallest to biggest
        nbr_of_x_points: int, the number of x points to use for the AUC
    Returns:
        recall_per_threshold_cut: array, recall values cut to the number of x points
        mean_FP_per_threshold_cut: array, mean FP values cut to the number of x points
    """

    recall_per_threshold = [0] + recall_per_threshold[::-1] + [recall_per_threshold[0]] 
    if mean_FP_per_threshold[0] < nbr_of_x_points:
        mean_FP_per_threshold = [0] + mean_FP_per_threshold[::-1] + [nbr_of_x_points]
    else:
        mean_FP_per_threshold = [0] + mean_FP_per_threshold[::-1] + [mean_FP_per_threshold[0]]

    mean_FP_per_threshold_cut = np.arange(0, nbr_of_x_points + 0.5, 0.5)

    interp_function = interp1d(mean_FP_per_threshold, recall_per_threshold, kind='linear')
    recall_per_threshold_cut = interp_function(mean_FP_per_threshold_cut)
    
    if len(mean_FP_per_threshold_cut) != len(recall_per_threshold_cut):
        raise ValueError('Length of recall and mean FP per threshold should be the same')
    
    if not 0 in mean_FP_per_threshold_cut:
        raise ValueError('Mean FP per threshold should start from 0')
    return recall_per_threshold_cut, mean_FP_per_threshold_cut


def AUC_FROC_curve(recall_per_threshold, mean_FP_per_threshold):
    """
    Compute the AUC of the FROC curve.
    Parameters:
        recall_per_threshold: list of recall values from smallest to biggest
        mean_FP_per_threshold: list of mean FP values from smallest to biggest
    Returns:
        auc: float, area under the FROC curve ranging from 0 to 1
    """
    return integrate.trapz(recall_per_threshold, mean_FP_per_threshold)/np.max(mean_FP_per_threshold)

def compute_IC_95(model, recall_key, thresh_list, permutation_nbr, patient_selection, nbr_of_x_points):
    """
    Compute the 95% confidence interval of the AUC of the FROC curve.
    Parameters:
        model: dict, the FROC metrics dictionary from the model
        recall_key: str, the key for the recall metric to compute the AUC
        thresh_list: list, the list of thresholds
        permutation_nbr: int, the number of permutations to perform
        patient_selection: dict, the patient selection dictionary to filter the lesions
        nbr_of_x_points: int, the number of x points to use for the AUC
    Returns:
        sensitivity_lower_bound: array, the lower bound of the sensitivity for each threshold
        sensitivity_upper_bound: array, the upper bound of the sensitivity for each threshold
        AUC_lower_bound: float, the lower bound of the AUC
        AUC_upper_bound: float, the upper bound of the AUC
        bootstrap_sensitivity: array, the sensitivity for each bootstrap iteration
    """
    rng = np.random.default_rng(seed=42) 
    if patient_selection is None:
        lesion_list = [{'patient': patient[:-7], 'lesion_id': lesion_idx, 'type': model['FROC_metrics_init'][thresh_list[0]]['LIRADS_list'][patient_idx][lesion_idx]} 
                        for patient_idx, patient in enumerate(model['patient_name']) 
                        for lesion_idx in range(0, len(model['FROC_metrics_init'][thresh_list[0]]['TP_list'][model['patient_name'].index(patient)]))]

    else:
        patient_list = list(patient_selection.keys())
        if patient_list[0][-7:] != '.nii.gz':
            patient_list = [patient + '.nii.gz' for patient in patient_list]
        lesion_list = [{'patient': patient[:-7], 'lesion_id': lesion_idx, 'type': model['FROC_metrics_init'][thresh_list[0]]['LIRADS_list'][patient_idx][lesion_idx]} 
                for patient_idx, patient in enumerate(model['patient_name']) 
                for lesion_idx in range(0, len(model['FROC_metrics_init'][thresh_list[0]]['TP_list'][model['patient_name'].index(patient)])) if patient in patient_list]

    from collections import defaultdict
    lesions_by_type = defaultdict(list)
    for lesion in lesion_list:
        lesions_by_type[lesion["type"]].append(lesion)

    bootstrap_AUC = []
    bootstrap_sensitivity = []
    for _ in range(permutation_nbr):
        random_patient_lesion_list = []
        
        for lesion_type, lesions in lesions_by_type.items():
            random_patient_lesion_list.extend(rng.choice(lesions, size=len(lesions), replace=True))
        metrics_not_lesion_wise = ['FP_list', 'TN_list', 'patients', 'patient_tumor_metrics_list', 'tumor_metrics_list', 'generated_neg_samples', 'lesion_type_list', 'Number of preds',]
        
        FROC_dict = {}
        for threshold in thresh_list:
            metrics_dict = {}
            for metric_name in model['FROC_metrics_init'][threshold].keys():
                metric_list = []
                patient_set = set([patient_lesion['patient'] for patient_lesion in random_patient_lesion_list])
                for patient in patient_set:
                    patient_metrics = []
                    lesions_idx = [int(patient_lesion['lesion_id']) for patient_lesion in random_patient_lesion_list if patient_lesion['patient'] == patient]
                    patient_idx = model['FROC_metrics_init'][threshold]['patients'].index(patient + '.nii.gz')
                    if metric_name in ['tumor_metrics_list', 'generated_neg_samples',  'prob_map_stats', 'prob_map_stats_list']:
                            continue

                    if metric_name in metrics_not_lesion_wise:
                        metric_list.append(model['FROC_metrics_init'][threshold][metric_name][patient_idx])
                    else:
                        for lesion_idx in lesions_idx:
                            patient_metrics.append(model['FROC_metrics_init'][threshold][metric_name][patient_idx][lesion_idx])
                        metric_list.append(patient_metrics)

                metrics_dict[metric_name] = metric_list
            FROC_dict[threshold] = metrics_dict
        
        FROC_metrics_dict = get_patient_metrics(FROC_dict, patient_selection)
        model_mean_FP_per_patient = [FROC_metrics_dict['FP_mean_per_patient_per_threshold'][threshold] for threshold in thresh_list]
        model_recall = [FROC_metrics_dict[recall_key][threshold] for threshold in thresh_list]

        model_recall_per_threshold, model_mean_FP_per_threshold = sort_data_for_AUC(model_recall, model_mean_FP_per_patient, nbr_of_x_points)

        if np.isnan(model_recall_per_threshold).any():
            continue
        auc_model = AUC_FROC_curve(model_recall_per_threshold, model_mean_FP_per_threshold)
        bootstrap_AUC.append(auc_model)
        bootstrap_sensitivity.append(model_recall_per_threshold)
    lower_percentile = 2.5
    upper_percentile = 97.5

    bootstrap_metrics_AUC = np.array(bootstrap_AUC)
    AUC_lower_bound = np.percentile(bootstrap_metrics_AUC, lower_percentile)
    AUC_upper_bound = np.percentile(bootstrap_metrics_AUC, upper_percentile)

    bootstrap_sensitivity = np.array(bootstrap_sensitivity)
    sensitivity_lower_bound = np.percentile(bootstrap_sensitivity, lower_percentile, axis=0)
    sensitivity_upper_bound = np.percentile(bootstrap_sensitivity, upper_percentile, axis=0)
    sensitivity_lower_bound[0] = 0
    sensitivity_upper_bound[0] = 0

    if any([np.isnan(item) for item in sensitivity_lower_bound]):
        AUC_lower_bound = 0

    return sensitivity_lower_bound, sensitivity_upper_bound, AUC_lower_bound, AUC_upper_bound, bootstrap_sensitivity



def ROC_patient_wise(nnunet_FROC_metrics_dict, nnunet_thresh_list):
    """
    Compute the ROC curve for each patient based on the nnUNet FROC metrics.
    Parameters:
        nnunet_FROC_metrics_dict: dict, the FROC metrics dictionary from nnUNet
        nnunet_thresh_list: list, the list of thresholds used in nnUNet
    Returns:
        TPR: dict, True Positive Rate for each threshold
        FPR: dict, False Positive Rate for each threshold
    """
    TP_patient_wise = {threshold: np.sum([1 if nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['Number of preds'][patient_idx] > 0 and 
                       len(nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['LIRADS_list'][patient_idx]) != 0 else 0 
                       for patient_idx in range(len(nnunet_FROC_metrics_dict['patient_name']) - 1)])
                        for threshold in nnunet_thresh_list}
    TN_patient_wise = {threshold: np.sum([1 if nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['Number of preds'][patient_idx] <= 0 and
                       len(nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['LIRADS_list'][patient_idx]) == 0 else 0
                       for patient_idx in range(len(nnunet_FROC_metrics_dict['patient_name']) - 1)])
                        for threshold in nnunet_thresh_list}
    FP_patient_wise = {threshold: np.sum([1 if nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['Number of preds'][patient_idx] > 0 and
                       len(nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['LIRADS_list'][patient_idx]) == 0 else 0
                       for patient_idx in range(len(nnunet_FROC_metrics_dict['patient_name']) - 1)])
                        for threshold in nnunet_thresh_list}
    FN_patient_wise = {threshold: np.sum([1 if nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['Number of preds'][patient_idx] <= 0 and
                       len(nnunet_FROC_metrics_dict['FROC_metrics_init'][threshold]['LIRADS_list'][patient_idx]) != 0 else 0
                       for patient_idx in range(len(nnunet_FROC_metrics_dict['patient_name']) - 1)])
                        for threshold in nnunet_thresh_list}
    TPR = {threshold: (TP_patient_wise[threshold]) / ((TP_patient_wise[threshold]) + (FN_patient_wise[threshold])) for threshold in nnunet_thresh_list}
    FPR = {threshold: (FP_patient_wise[threshold]) / ((FP_patient_wise[threshold]) + (TN_patient_wise[threshold])) for threshold in nnunet_thresh_list}
    return TPR, FPR


def compute_IC_95_patientwise(model, thresh_list, permutation_nbr, patient_selection):
    """
    Compute the 95% confidence interval of the AUC of the FROC curve.
    Parameters:
        model: dict, the FROC metrics dictionary from the model
        thresh_list: list, the list of thresholds
        permutation_nbr: int, the number of permutations to perform
        patient_selection: dict, the patient selection dictionary to filter the lesions
    Returns:
        TPR_lower_bound: array, the lower bound of the TPR for each threshold
        TPR_upper_bound: array, the upper bound of the TPR for each threshold
        interp_points: array, the x points for the TPR
        mean_AUC: float, the mean AUC
        AUC_lower_bound: float, the lower bound of the AUC
        AUC_upper_bound: float, the upper bound of the AUC
        bootstrap_TPR: array, the TPR for each bootstrap iteration
    """
    rng = np.random.default_rng(seed=42)
    interp_points = np.linspace(0, 1, 20)

    patient_list = [{'patient': patient[:-7], 'type': 1 if len(model['FROC_metrics_init'][thresh_list[0]]['HCC_list'][patient_idx]) != 0 else 0}
                    for patient_idx, patient in enumerate(model['patient_name'])
                    ]
    patients_by_type = defaultdict(list)
    for patient in patient_list:
        patients_by_type[patient["type"]].append(patient)

    bootstrap_AUC = []
    bootstrap_TPR = []
    for _ in range(permutation_nbr):
        random_patient_list = []
        for patient_type, patients in patients_by_type.items():
            random_patient_list.extend(rng.choice(patients, size=len(patients), replace=True))
        random_patient_list = [patient['patient'] for patient in random_patient_list]
        FROC_dict = {}
        for threshold in thresh_list:
            metrics_dict = {}
            for metric_name in model['FROC_metrics_init'][threshold].keys():
                metric_list = []
                metrics_not_patient_wise = ['generated_neg_samples', 'tumor_metrics_list']
                if metric_name in metrics_not_patient_wise:
                    continue
                for patient in random_patient_list:
                    patient_idx = model['FROC_metrics_init'][threshold]['patients'].index(patient + '.nii.gz')
                    metric_list.append(model['FROC_metrics_init'][threshold][metric_name][patient_idx])
                metrics_dict[metric_name] = metric_list
            FROC_dict[threshold] = metrics_dict
        FROC_metrics_dict = get_patient_metrics(FROC_dict, patient_selection)
        TPR, FPR = ROC_patient_wise(FROC_metrics_dict, thresh_list)
        TPR = [0] + list(TPR.values())[::-1]
        FPR = [0] + list(FPR.values())[::-1]
        interp_tpr = np.interp(interp_points, FPR, TPR)
        auc_model = integrate.trapz(TPR + [1], FPR + [1])
        bootstrap_AUC.append(auc_model)
        bootstrap_TPR.append(interp_tpr)

    lower_percentile = 2.5
    upper_percentile = 97.5
    bootstrap_metrics_AUC = np.array(bootstrap_AUC)
    AUC_lower_bound = np.percentile(bootstrap_metrics_AUC, lower_percentile)
    AUC_upper_bound = np.percentile(bootstrap_metrics_AUC, upper_percentile)
    bootstrap_TPR = np.array(bootstrap_TPR)
    TPR_lower_bound = np.percentile(bootstrap_TPR, lower_percentile, axis=0)
    TPR_upper_bound = np.percentile(bootstrap_TPR, upper_percentile, axis=0)
    mean_AUC = np.mean(bootstrap_metrics_AUC)
    if any([np.isnan(item) for item in TPR_lower_bound]):
        AUC_lower_bound = 0

    return TPR_lower_bound, TPR_upper_bound, interp_points, mean_AUC, AUC_lower_bound, AUC_upper_bound, bootstrap_TPR