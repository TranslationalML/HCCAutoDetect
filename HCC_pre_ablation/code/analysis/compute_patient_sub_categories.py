import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import sys
from scipy.spatial.distance import pdist
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from utils import contrast_check


def main(data_path, exclude_patients):
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    images = os.path.join(data_path, "images_4D")
    multi_labels = os.path.join(data_path, "multi_labels")

    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t")

    patient_HCC_size = {}
    HCC_patients_LR_grade = {}
    non_HCC_patients_LR_grade = {}
    lesion_size = {}
    nat_contrasts = {}
    art_contrasts = {}
    ven_contrasts = {}
    del_contrasts = {}
    lesion_characteristics = pd.read_csv(os.path.join(data_path, "tumors_characteristics.csv"))
    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        if patient_ID in exclude_patients:
            continue
        if os.path.exists(os.path.join(images, patient_ID + '.nii.gz')) == False:
            continue
        image = sitk.ReadImage(os.path.join(images, patient_ID + '.nii.gz'))
        multi_label = sitk.ReadImage(os.path.join(multi_labels, patient_ID + '.nii.gz'))
        labels = np.unique(sitk.GetArrayFromImage(multi_label))
        labels = labels[labels != 0]

        HCC_size = {}
        HCC_LR_grade = {}
        non_HCC_LR_grade = {}
        lesions_size_in_patient_dict = {}
        nat_contrasts_in_patient_dict = {}
        art_contrasts_in_patient_dict = {}
        ven_contrasts_in_patient_dict = {}
        del_contrasts_in_patient_dict = {}
        for label_index in labels:
            binary_mask_1_lesion = (multi_label == label_index)
            binary_mask_1_lesion = sitk.Cast(binary_mask_1_lesion, sitk.sitkUInt8)

            voxel_spacing = binary_mask_1_lesion.GetSpacing()
            label_ar = sitk.GetArrayFromImage(binary_mask_1_lesion)
            distances_3D = []
            for plane in range(label_ar.shape[0]):
                coordinates = np.argwhere(label_ar[plane, :, :] > 0)
                if len(coordinates) > 1:
                    distances = pdist(coordinates, 'euclidean')
                    max_distance = np.max(distances)
                    distances_3D.append(max_distance)
            if len(distances_3D) == 0:
                lesions_size_in_patient_dict[str(label_index)] = 0
                continue
            diameter = np.max(distances_3D) * voxel_spacing[0]
            lesions_size_in_patient_dict[str(label_index)] = diameter
            contrast_nat, contrast_art, contrast_ven, contrast_del = contrast_check(image, binary_mask_1_lesion)

            nat_contrasts_in_patient_dict[str(label_index)] = contrast_nat
            art_contrasts_in_patient_dict[str(label_index)] = contrast_art
            ven_contrasts_in_patient_dict[str(label_index)] = contrast_ven
            del_contrasts_in_patient_dict[str(label_index)] = contrast_del

            is_HCC = lesion_characteristics.loc[(lesion_characteristics['ID'] == patient_ID) & (lesion_characteristics['label'] == label_index), 'HCC'].values
            if is_HCC[0] > 0:
                HCC_size[str(label_index)] = diameter
                HCC_LR_grade[str(label_index)] = lesion_characteristics.loc[(lesion_characteristics['ID'] == patient_ID) & (lesion_characteristics['label'] == label_index), 'LIRADS'].values[0]
            if is_HCC[0] == 0:
                non_HCC_LR_grade[str(label_index)] = lesion_characteristics.loc[(lesion_characteristics['ID'] == patient_ID) & (lesion_characteristics['label'] == label_index), 'LIRADS'].values[0]


        lesion_size[patient_ID] = lesions_size_in_patient_dict
        nat_contrasts[patient_ID] = nat_contrasts_in_patient_dict
        art_contrasts[patient_ID] = art_contrasts_in_patient_dict
        ven_contrasts[patient_ID] = ven_contrasts_in_patient_dict
        del_contrasts[patient_ID] = del_contrasts_in_patient_dict
        patient_HCC_size[patient_ID] = HCC_size
        HCC_patients_LR_grade[patient_ID] = HCC_LR_grade
        non_HCC_patients_LR_grade[patient_ID] = non_HCC_LR_grade

    excluded_patients_art_contrast_1 = [k_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion <= 1]
    excluded_patients_art_contrast_1_1 = [k_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion <= 1.1]
    excluded_patients_art_contrast_1_2 = [k_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion <= 1.2]
    excluded_patients_art_contrast_1_3 = [k_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion <= 1.3]
    excluded_patients_art_contrast_outside_1_to_1_15_range = [k_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion < 1 or v_lesion >= 1.15]
    all_patients = [k for k, v in art_contrasts.items() for k_lesion, v_lesion in v.items()]

    patients_art_contrast_hypo = [patient for patient in all_patients if patient in excluded_patients_art_contrast_1]
    patients_art_contrast_1 = [patient for patient in all_patients if patient not in excluded_patients_art_contrast_1]
    patients_art_contrast_1_1 = [patient for patient in all_patients if patient not in excluded_patients_art_contrast_1_1]
    patients_art_contrast_1_2 = [patient for patient in all_patients if patient not in excluded_patients_art_contrast_1_2]
    patients_art_contrast_1_3 = [patient for patient in all_patients if patient not in excluded_patients_art_contrast_1_3]
    patients_art_contrast_outside_1_to_1_15_range = [patient for patient in all_patients if patient not in excluded_patients_art_contrast_outside_1_to_1_15_range]

    lesion_art_contrast_hypo = {k_patient:v_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion <= 1 and k_patient in patients_art_contrast_hypo}
    lesions_art_contrast_1 = {k_patient:v_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion > 1 and k_patient in patients_art_contrast_1}
    lesions_art_contrast_1_1= {k_patient:v_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion > 1.1 and k_patient in patients_art_contrast_1_1}
    lesions_art_contrast_1_2 = {k_patient:v_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion > 1.2 and k_patient in patients_art_contrast_1_2}
    lesions_art_contrast_1_3 = {k_patient:v_patient for k_patient, v_patient in art_contrasts.items() for k_lesion, v_lesion in v_patient.items() if v_lesion > 1.3 and k_patient in patients_art_contrast_1_3}


    patients_less_70mm = {k_patient:v_patient for k_patient, v_patient in lesion_size.items() for k_lesion, v_lesion in v_patient.items() if v_lesion < 70}
    patients_less_50mm = {k_patient:v_patient for k_patient, v_patient in lesion_size.items() for k_lesion, v_lesion in v_patient.items() if v_lesion < 50}
    patients_less_20mm = {k_patient:v_patient for k_patient, v_patient in lesion_size.items() for k_lesion, v_lesion in v_patient.items() if v_lesion < 20}
    patients_less_10mm = {k_patient:v_patient for k_patient, v_patient in lesion_size.items() for k_lesion, v_lesion in v_patient.items() if v_lesion < 10}

    
    mean_lesion_size = np.mean([v_lesion for v_patient in lesion_size.values() for v_lesion in v_patient.values()])
    std_lesion_size = np.std([v_lesion for v_patient in lesion_size.values() for v_lesion in v_patient.values()])
    print(mean_lesion_size, ' +- ', std_lesion_size)

    lesions_per_patient = {k: len(v) for k, v in lesion_size.items() if len(v) > 0}
    mean_nbr_lesions_per_patient = np.mean(list(lesions_per_patient.values()))
    std_nbr_lesions_per_patient = np.std(list(lesions_per_patient.values()))



    os.makedirs(data_path + "/patient_selection", exist_ok=True)
    with open(os.path.join(data_path, "patient_selection/HCC_lesion_sizes.json"), 'w') as f:
        json.dump(patient_HCC_size, f, indent=4)

    with open(os.path.join(data_path, "patient_selection/HCC_patients_LR_grade.json"), 'w') as f:
        json.dump({'LR-5': int(np.sum([1 for k, v in HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '5' or v_lesion == '5 (growth)'])),
                    'LR-4': int(np.sum([1 for k, v in HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '4'])),
                    'LR-3': int(np.sum([1 for k, v in HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '3'])),
                    'LR-2': int(np.sum([1 for k, v in HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '2'])),
                    'LR-M': int(np.sum([1 for k, v in HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == 'M'])),
                    'LR-TIV': int(np.sum([1 for k, v in HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == 'TIV'])),
                    'HCC mean size': np.mean([v_lesion for k, v in lesion_size.items() for k_lesion, v_lesion in v.items() if k_lesion in HCC_patients_LR_grade[k].keys()]),
                    'HCC std size': np.std([v_lesion for k, v in lesion_size.items() for k_lesion, v_lesion in v.items() if k_lesion in HCC_patients_LR_grade[k].keys()]),
                   }, f, indent=4)
    with open(os.path.join(data_path, "patient_selection/non_HCC_patients_LR_grade.json"), 'w') as f:
        json.dump({'LR-5': int(np.sum([1 for k, v in non_HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '5' or v_lesion == '5 (growth)'])),
                    'LR-4': int(np.sum([1 for k, v in non_HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '4'])),
                    'LR-3': int(np.sum([1 for k, v in non_HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '3'])),
                    'LR-2': int(np.sum([1 for k, v in non_HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == '2'])),
                    'LR-M': int(np.sum([1 for k, v in non_HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == 'M'])),
                    'LR-TIV': int(np.sum([1 for k, v in non_HCC_patients_LR_grade.items() for k_lesion, v_lesion in v.items() if v_lesion == 'TIV'])),
                    'non-HCC mean size': np.mean([v_lesion for k, v in lesion_size.items() for k_lesion, v_lesion in v.items() if k_lesion in non_HCC_patients_LR_grade[k].keys()]),
                    'non-HCC std size': np.std([v_lesion for k, v in lesion_size.items() for k_lesion, v_lesion in v.items() if k_lesion in non_HCC_patients_LR_grade[k].keys()]),
                   }, f, indent=4)
    
    with open(os.path.join(data_path, "patient_selection/patient_wise_art_contrast.json"), 'w') as f:
        json.dump({'contrast <= 1': sorted(list(set(patients_art_contrast_hypo))),
                    'contrast > 1': sorted(list(set(patients_art_contrast_1))),
                   'contrast > 1.1': sorted(list(set(patients_art_contrast_1_1))),
                   'contrast > 1.2': sorted(list(set(patients_art_contrast_1_2))),
                   'contrast > 1.3': sorted(list(set(patients_art_contrast_1_3))),
                   'contrast outside [1:1.15] range': sorted(list(set(patients_art_contrast_outside_1_to_1_15_range)))},
                    f)
    
    with open(os.path.join(data_path, "patient_selection/patient_wise_lesion_diameter.json"), 'w') as f:
        json.dump({'all_lesions': lesion_size,
                   'lesion_diameter_less_70mm': patients_less_70mm,
                   'lesion_diameter_less_50mm': patients_less_50mm,
                   'lesion_diameter_less_20mm': patients_less_20mm,
                   'lesion_diameter_less_10mm': patients_less_10mm
                   }, 
                   f)

    lesions_per_contrast_category = {k: len(v) for k, v in {'contrast <= 1': lesion_art_contrast_hypo,
                                                            'contrast > 1': lesions_art_contrast_1,
                                                            'contrast > 1.1': lesions_art_contrast_1_1,
                                                            'contrast > 1.2': lesions_art_contrast_1_2,
                                                            'contrast > 1.3': lesions_art_contrast_1_3}.items()}
    patients_per_contrast_category = {k: len(v) for k, v in {
                                                            'contrast <= 1': list(set(patients_art_contrast_hypo)), 
                                                            'contrast > 1': list(set(patients_art_contrast_1)),
                                                            'contrast > 1.1': list(set(patients_art_contrast_1_1)),
                                                            'contrast > 1.2': list(set(patients_art_contrast_1_2)),
                                                            'contrast > 1.3':list(set(patients_art_contrast_1_3))}.items()}
    lesions_per_size_category = {k: len(v) for k, v in {'lesion_diameter_less_70mm': patients_less_70mm,
                                                        'lesion_diameter_less_50mm': patients_less_50mm,
                                                        'lesion_diameter_less_25mm': patients_less_20mm}.items()}
    
    patients_per_size_category = {k: len(v) for k, v in {'lesion_diameter_less_70mm': list(set([k for k in patients_less_70mm.keys()])),
                                                        'lesion_diameter_less_50mm': list(set([k for k in patients_less_50mm.keys()])),
                                                        'lesion_diameter_less_25mm': list(set([k for k in patients_less_20mm.keys()]))}.items()}
    

    # Lesion-wise selection
    lesions_art_contrast_hypo = {k_patient: {k_lesion: float(v_lesion) if v_lesion <= 1 else None for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in art_contrasts.items()}
    lesions_art_contrast_1 = {k_patient: {k_lesion: float(v_lesion)  if v_lesion > 1 else None for k_lesion, v_lesion in v_patient.items() if v_lesion > 1} for k_patient, v_patient in art_contrasts.items()}
    lesions_art_contrast_1_1 = {k_patient: {k_lesion: float(v_lesion) if v_lesion > 1.1 else None for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in art_contrasts.items()}
    lesions_art_contrast_1_2 = {k_patient: {k_lesion: float(v_lesion) if v_lesion > 1.2 else None for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in art_contrasts.items()}
    lesions_art_contrast_1_3 = {k_patient: {k_lesion: float(v_lesion) if v_lesion > 1.3 else None for k_lesion, v_lesion in v_patient.items() } for k_patient, v_patient in art_contrasts.items()}
    lesions_art_contrast_hypo_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_art_contrast_hypo.items() if v_patient is not None}
    lesions_art_contrast_1_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_art_contrast_1.items() if v_patient is not None}
    lesions_art_contrast_1_1_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_art_contrast_1_1.items() if v_patient is not None}
    lesions_art_contrast_1_2_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_art_contrast_1_2.items() if v_patient is not None}
    lesions_art_contrast_1_3_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_art_contrast_1_3.items() if v_patient is not None}
    lesions_art_contrast_hypo_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_art_contrast_hypo_no_nan.items() if len(v_patient) > 0}
    lesions_art_contrast_1_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_art_contrast_1_no_nan.items() if len(v_patient) > 0}
    lesions_art_contrast_1_1_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_art_contrast_1_1_no_nan.items() if len(v_patient) > 0}
    lesions_art_contrast_1_2_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_art_contrast_1_2_no_nan.items() if len(v_patient) > 0}
    lesions_art_contrast_1_3_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_art_contrast_1_3_no_nan.items() if len(v_patient) > 0}


    lesions_selec_lesions_less_70mm = {k_patient: {k_lesion: v_lesion if v_lesion < 70 else None for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in lesion_size.items()}
    lesions_selec_lesions_less_50mm = {k_patient: {k_lesion: v_lesion if v_lesion < 50 else None for k_lesion, v_lesion in v_patient.items()}  for k_patient, v_patient in lesion_size.items()}
    lesions_selec_lesions_less_20mm = {k_patient: {k_lesion: v_lesion if v_lesion < 20 else None for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in lesion_size.items()}
    lesions_selec_lesions_less_10mm = {k_patient: {k_lesion: v_lesion if v_lesion < 10 else None for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in lesion_size.items()}

    lesions_selec_lesions_less_70mm_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_selec_lesions_less_70mm.items()}
    lesions_selec_lesions_less_50mm_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_selec_lesions_less_50mm.items() if v_patient is not None}
    lesions_selec_lesions_less_20mm_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_selec_lesions_less_20mm.items() if v_patient is not None}
    lesions_selec_lesions_less_10mm_no_nan = {k_patient: {k_lesion: v_lesion for k_lesion, v_lesion in v_patient.items() if v_lesion is not None} for k_patient, v_patient in lesions_selec_lesions_less_10mm.items() if v_patient is not None}
    lesions_selec_lesions_less_70mm_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_selec_lesions_less_70mm_no_nan.items() if len(v_patient) > 0}
    lesions_selec_lesions_less_50mm_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_selec_lesions_less_50mm_no_nan.items() if len(v_patient) > 0}
    lesions_selec_lesions_less_20mm_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_selec_lesions_less_20mm_no_nan.items() if len(v_patient) > 0}
    lesions_selec_lesions_less_10mm_no_nan = {k_patient: v_patient for k_patient, v_patient in lesions_selec_lesions_less_10mm_no_nan.items() if len(v_patient) > 0}
    

    lesion_selection_lesions_per_contrast_category = {k: len(v) for k, v in {'contrast <= 1': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_art_contrast_hypo_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'contrast > 1': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_art_contrast_1_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'contrast > 1.1': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_art_contrast_1_1_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'contrast > 1.2': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_art_contrast_1_2_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'contrast > 1.3': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_art_contrast_1_3_no_nan.items() for k_lesion, v_lesion in v_patient.items()]
                                                            }.items()}
    lesion_selection_patients_per_diameter_category = {k: len(v) for k, v in {
                                                            'All lesions': [k_patient + '_' + k_lesion for k_patient, v_patient in lesion_size.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'lesion_diameter_less_70mm': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_selec_lesions_less_70mm_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'lesion_diameter_less_50mm': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_selec_lesions_less_50mm_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'lesion_diameter_less_20mm': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_selec_lesions_less_20mm_no_nan.items() for k_lesion, v_lesion in v_patient.items()],
                                                            'lesion_diameter_less_10mm': [k_patient + '_' + k_lesion for k_patient, v_patient in lesions_selec_lesions_less_10mm_no_nan.items() for k_lesion, v_lesion in v_patient.items()]
                                                            }.items()}
    
    with open(os.path.join(data_path, "patient_selection/lesion_wise_art_contrast.json"), 'w') as f:
        json.dump({'contrast <= 1': lesions_art_contrast_hypo,
                    'contrast > 1': lesions_art_contrast_1,
                    'contrast > 1.1': lesions_art_contrast_1_1,
                    'contrast > 1.2': lesions_art_contrast_1_2,
                    'contrast > 1.3': lesions_art_contrast_1_3}, f, indent=4)
        
    with open(os.path.join(data_path, "patient_selection/lesion_wise_lesion_diameter.json"), 'w') as f:
        json.dump({'lesion_diameter_less_70mm': lesions_selec_lesions_less_70mm,
                   'lesion_diameter_less_50mm': lesions_selec_lesions_less_50mm,
                   'lesion_diameter_less_20mm': lesions_selec_lesions_less_20mm,
                   'lesion_diameter_less_10mm': lesions_selec_lesions_less_10mm}, f, indent=4)

    with open(os.path.join(data_path, "patient_selection/lesions_information.json"), 'w') as f:
        json.dump({'patient_wise_selection:_lesions_per_contrast_category': lesions_per_contrast_category,
                   'patient_wise_selection:_patients_per_contrast_category': patients_per_contrast_category,
                   'patient_wise_selection:_lesions_per_size_category': lesions_per_size_category,
                   'patient_wise_selection:_patients_per_size_category': patients_per_size_category,
                   'lesion_wise_selection:_lesions_per_contrast_category': lesion_selection_lesions_per_contrast_category,
                   'lesion_wise_selection:_lesions_per_size_category': lesion_selection_patients_per_diameter_category,
                   'total lesions': len(lesion_size), 
                   'mean lesion diameter': mean_lesion_size, 
                   'std lesion diameter': std_lesion_size,
                   'mean number of lesions per patient': mean_nbr_lesions_per_patient,
                     'std number of lesions per patient': std_nbr_lesions_per_patient,
                     'mean HCC size': np.mean([v_lesion for k_patient, v_patient in patient_HCC_size.items() for k_lesion, v_lesion in v_patient.items()]),
                     'std HCC size': np.std([v_lesion for k_patient, v_patient in patient_HCC_size.items() for k_lesion, v_lesion in v_patient.items()]),
                   }, f, indent=4)


    art_contrasts_float = {k_patient: {k_lesion: float(v_lesion) for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in art_contrasts.items()}
    with open(os.path.join(data_path, "patient_selection/lesion_contrast_list.json"), 'w') as f:
        json.dump(art_contrasts_float, f, indent=4)

    with open(os.path.join(data_path, "patient_selection/lesion_diameter_list.json"), 'w') as f:
        json.dump(lesion_size, f, indent=4)

    # other contrast
    nat_contrasts = {k_patient: {k_lesion: float(v_lesion) for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in nat_contrasts.items() if v_patient != {}}
    ven_contrasts = {k_patient: {k_lesion: float(v_lesion) for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in ven_contrasts.items() if v_patient != {}}
    del_contrasts = {k_patient: {k_lesion: float(v_lesion) for k_lesion, v_lesion in v_patient.items()} for k_patient, v_patient in del_contrasts.items() if v_patient != {}}

    with open(os.path.join(data_path, "patient_selection/lesion_contrast_list_nat.json"), 'w') as f:
        json.dump(nat_contrasts, f, indent=4)
    with open(os.path.join(data_path, "patient_selection/lesion_contrast_list_ven.json"), 'w') as f:
        json.dump(ven_contrasts, f, indent=4)
    with open(os.path.join(data_path, "patient_selection/lesion_contrast_list_del.json"), 'w') as f:
        json.dump(del_contrasts, f, indent=4)

    return


if __name__ == "__main__":
    cwd = os.getcwd()
    data_path = os.path.dirname(os.path.dirname(cwd)) + "/derivatives/6_T1_dataset"
    exclude_patients = []
    main(data_path, exclude_patients)

    
