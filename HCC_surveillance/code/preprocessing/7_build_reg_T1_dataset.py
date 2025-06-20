import SimpleITK as sitk
import numpy as np
import os
from collections import Counter
import json
import cc3d
import pandas as pd
from matplotlib import pyplot as plt


def main(output_derivative_path, data_dir, tumor_seg_dir, liver_seg_dir, total_seg_dir, margin, save_as_4D, subtract):
    os.makedirs(output_derivative_path, exist_ok=True)
    output_img_path = os.path.join(output_derivative_path, 'images')
    os.makedirs(output_img_path, exist_ok=True)
    output_img_sub_path = os.path.join(output_derivative_path, 'image_subtractions')
    os.makedirs(output_img_sub_path, exist_ok=True)
    output_img_4D_path = os.path.join(output_derivative_path, 'images_4D')
    os.makedirs(output_img_4D_path, exist_ok=True)

    output_label_path = os.path.join(output_derivative_path, 'labels')
    os.makedirs(output_label_path, exist_ok=True)
    output_rb_label_path = os.path.join(output_derivative_path, 'region_based_labels')
    os.makedirs(output_rb_label_path, exist_ok=True)
    output_liver_label_path = os.path.join(output_derivative_path, 'liver_labels')
    os.makedirs(output_liver_label_path, exist_ok=True)
    output_multilabel_path = os.path.join(output_derivative_path, 'multi_labels')
    os.makedirs(output_multilabel_path, exist_ok=True)

    output_total_segmentator_path = os.path.join(output_derivative_path, 'total_segmentator')
    os.makedirs(output_total_segmentator_path, exist_ok=True)



    number_of_tumors_file = os.path.join(os.path.dirname(os.getcwd()), 'analysis/excel_files/tumors_per_patient_all_phases.json')
    with open(number_of_tumors_file, 'r') as file:
        number_of_tumors = json.load(file)
    path = os.path.join(main_dir, "sourcedata/Lésions_VF.xlsx")
    df = pd.read_excel(path, dtype=str, header=[0, 1])
    last_index = 234
    df = df.iloc[:last_index+1]


    # match patient names
    match_patient_names_HCC_pos = os.path.join(main_dir, "code/pipeline/match_patient_names_HCC_pos.json")
    with open(match_patient_names_HCC_pos, 'r') as file:
        match_names_HCC_pos = json.load(file)
    match_patient_names_HCC_neg = os.path.join(main_dir, "code/pipeline/match_patient_names_HCC_neg.json")
    with open(match_patient_names_HCC_neg, 'r') as file:
        match_names_HCC_neg = json.load(file)
    match_names = {**match_names_HCC_neg, **match_names_HCC_pos}


    inv_match_names = {v[4:]: k for k, v in match_names.items()}
    df['IPP_fill'] = df['IPP']['Unnamed: 2_level_1'].fillna(method='ffill')
    df['ID'] = df['IPP_fill'].replace(inv_match_names)
    df['tumor_arterial'] = df['Artériel']['Phases :\n1=AP1\n2=AP2\n3=AP3\n4=Une seule phase']

    liver_labels = sorted([os.path.join(dir_, file) for dir_, subdirs, files in os.walk(liver_seg_dir) for file in files
                if 'venous' in file])
    lesion_labels = sorted([os.path.join(dir_, file) for dir_, subdirs, files in os.walk(tumor_seg_dir) for file in files])

    test_patients = [x for x in [f'sub-{str(i).zfill(3)}' for i in range(97, 119 + 1)]]

    patient_dropped_files = {}
    patient_with_copy_annotations = {}
    annotations_count = {}
    removed_patient = {}
    patient_with_multiple_TTC = {}
    lesions_per_phase = {}
    lesions_size = {}
    lesion_per_patient_dict = {}
    spacing = {}
    dimensions = {}
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')
    all_tumors_df_list = []
    for patient, sub_id in participants[['Patient', 'Sub-ID']].values:
        patient_dyn_files_to_select = os.path.join(os.path.dirname(output_derivative_path) + '/6_select_dyn_phases', sub_id)
        print(sub_id)
        if sub_id in test_patients:
            removed_patient[sub_id] = "Patient excluded, in test set"
            continue
        if sub_id not in df['ID'].values:
            removed_patient[sub_id] = "Patient excluded due to bad quality imaging or missing sequences"
            continue
        
        if sub_id in ['sub-044']: #Problem of annotation, missing Liver
            removed_patient[sub_id] = "Bad arterial phase"
            continue

        if sub_id in ['sub-079', 'sub-082', 'sub-039', 'sub-033', 'sub-019', 'sub-030']:  # too many unnanotated lesions
            removed_patient[sub_id] = "Patient excluded, too many unnanotated lesions"
            continue

        if sub_id in ['sub-012', 'sub-016', 'sub-017', 'sub-018', 'sub-035', 'sub-076', 'sub-090', 'sub-111', 'sub-119']:
            removed_patient[sub_id] = "Patient excluded due to too large lesion(s)"
            continue

        try:
            liver_lab = [file for file in liver_labels if sub_id in file][0]
        except:
            print('No Liver label for patient ', sub_id)
            removed_patient[sub_id] = "No Liver segmentation"
            continue

        sub_df = df.loc[df['ID'] == sub_id]
        sub_df_ID = sub_df['ID']
        sub_df_art = sub_df['Artériel']['Si visible : 1=Hyper\n2=Iso ou hypo\n3=Périph']
        sub_df_art.name = 'Arterial'
        sub_df_ven_washout = sub_df['Veineux']['Si visible:\nWash-out:\n0=Non\n1=Oui']
        sub_df_ven_washout.name = 'Venous washout'
        sub_df_ven_capsule = sub_df['Veineux']['Si visible:\n0=Pas de capsule\n1=Capsule réhaussante\n2=Capsule non réhaussante']
        sub_df_ven_capsule.name = 'Venous capsule'
        sub_df_del_capsule = sub_df['Tardif']['Si visible:\n0=Pas de capsule\n1=Capsule réhaussante\n2=Capsule non réhaussante']
        sub_df_del_capsule.name = 'Delayed capsule'
        sub_df_del_washout = sub_df['Tardif']['Si visible:\nWash-out:\n0=Non\n1=Oui']
        sub_df_del_washout.name = 'Delayed washout'
        sub_df_LIRADS = sub_df['LIRADS']['Unnamed: 44_level_1']
        sub_df_LIRADS.name = 'LIRADS'
        sub_df_size = sub_df['Taille lésion']['Unnamed: 18_level_1']
        sub_df_size.name = 'Lesion diameter'
        sub_df_loc = sub_df['Segment(s)']['Unnamed: 19_level_1']
        sub_df_loc.name = 'Location'
        sub_df_HCC = sub_df['Type de lésion']['0=CHC\n1=Autres']
        sub_df_HCC.name = 'HCC'
        sub_df_conc = pd.concat([sub_df_ID, sub_df_art, sub_df_ven_washout, sub_df_ven_capsule,
                                 sub_df_del_washout, sub_df_del_capsule, sub_df_LIRADS, sub_df_loc, sub_df_size, sub_df_HCC], axis=1)
        sub_df_conc['HCC'] = sub_df_conc['HCC'].replace({'0': '1', '1': '0'})
        sub_df_conc = sub_df_conc.reset_index(drop=True)
        
        sub_df_conc.insert(0, 'label', sub_df_conc.index + 1)
        sub_df_conc.fillna(0, inplace=True)

        # Select unwanted lesions
        benin_lesions = []
        for idx, row in sub_df_conc.iterrows():
            if row['LIRADS'] not in ['3', '4', '5', 'TIV', 'M']:
                sub_df_conc = sub_df_conc.drop(index=idx)
                removed_patient[sub_id] = "Dropped lesion " + str(idx) + " as it doesn't follow criterion"
                benin_lesions.append(idx)

        #img
        patient_dyn_path = os.path.join(data_dir, sub_id + '/dyn')
        img_list = [os.path.join(patient_dyn_path, file_name) for file_name in os.listdir(patient_dyn_path)]

        all_TTC_lesion_lab = [file for file in lesion_labels if sub_id in file and
                              all([any(x in file for x in ['dixon_w', 'caipi']),
                                   any(x in file for x in ['native', 'arterial', 'venous', 'delayed']),
                                   any(x in file for x in ['L1', 'L2', 'L3', 'L4', 'L5', 'Copy', 'Lesion'])])]
        annot_count = Counter([1 if 'L1' in file else 2 if 'L2' in file else 3 if 'L3' in file else 4 if 'L4' in file
                       else 5 if 'L5' in file else 0 for file in all_TTC_lesion_lab])
        annotations_count[sub_id] = len(annot_count)


        if len(benin_lesions) != 0:
            benin_lesion_names = ['L' + str(idx+1) for idx in benin_lesions]
            all_TTC_lesion_lab = [file for file in all_TTC_lesion_lab if not any(x in file for x in benin_lesion_names)]

        if len(all_TTC_lesion_lab) != 0:
            liver_lab = [file for file in liver_labels if sub_id in file and 'venous' in file][0]
            if any([x in file for file in all_TTC_lesion_lab for x in ['Copy', 'copy', 'lesion']]):
                copy_annotations = True
                L1 = [file for file in all_TTC_lesion_lab if any(x in file for x in ['L1', 'Lesion 1', 'Lesion-1'])]
                L2 = [file for file in all_TTC_lesion_lab if any(x in file for x in ['L2', 'Lesion 2', 'Lesion-2'])]
                L3 = [file for file in all_TTC_lesion_lab if any(x in file for x in ['L3', 'Lesion 3', 'Lesion-3'])]
                L4 = [file for file in all_TTC_lesion_lab if any(x in file for x in ['L4', 'Lesion 4', 'Lesion-4'])]
                L5 = [file for file in all_TTC_lesion_lab if any(x in file for x in ['L5', 'Lesion 5', 'Lesion-5'])]

                F01 = 'F' + L1[0].partition('_F')[2][:2] if len(L1) != 0 else None
                F02 = 'F' + L2[0].partition('_F')[2][:2] if len(L2) != 0 else None
                F03 = 'F' + L3[0].partition('_F')[2][:2] if len(L3) != 0 else None
                F04 = 'F' + L4[0].partition('_F')[2][:2] if len(L4) != 0 else None
                F05 = 'F' + L5[0].partition('_F')[2][:2] if len(L5) != 0 else None
                F_annots = [F_ for F_ in [F01, F02, F03, F04, F05] if F_ is not None]


                lesion_name_list = ['L1', 'L2', 'L3', 'L4', 'L5',
                                    'Lesion-1', 'Lesion-2', 'Lesion-3', 'Lesion-4', 'Lesion-5'] + F_annots

            else:
                copy_annotations = False
                lesion_name_list = ['L1', 'L2', 'L3', 'L4', 'L5']

            dyn_files_path = (patient_dyn_files_to_select + '/dyn/phases_to_keep.json')
            with open(dyn_files_path, 'r') as f:
                name_list = json.load(f)
                files_to_keep = [file_seg for file_seg in all_TTC_lesion_lab if any([os.path.basename(file)[:-7] in file_seg for file in name_list])]
                dropped_files = [file_seg for file_seg in all_TTC_lesion_lab if file_seg not in files_to_keep]
                patient_dropped_files[sub_id] = dropped_files
                all_TTC_lesion_lab = files_to_keep


            is_TTC = any([x in file for file in all_TTC_lesion_lab for x in ['TTC',]])

            TTC_with_most_tumors = Counter([(file.partition('TTC_')[2][0]) for file in all_TTC_lesion_lab if 'TTC' in file]).most_common(1)[0][0] if is_TTC else None

            if TTC_with_most_tumors is not None:
                all_TTC_lesion_lab = [file for file in all_TTC_lesion_lab if 'TTC_' not in file or 'TTC_' + TTC_with_most_tumors in file]

            native_lesion_lab = [file for file in all_TTC_lesion_lab if sub_id in file and 'native' in file
                                 and any(x in file for x in lesion_name_list)]
            arterial_lesion_lab = [file for file in all_TTC_lesion_lab if sub_id in file and
                                   any([x in file for x in ['arterial']]) and
                                   any(x in file for x in lesion_name_list)]
            venous_lesion_lab = [file for file in all_TTC_lesion_lab if sub_id in file and 'venous' in file
                                 and any(x in file for x in lesion_name_list)]
            delayed_lesion_lab = [file for file in all_TTC_lesion_lab if sub_id in file and 'delayed' in file
                                  and any(x in file for x in lesion_name_list)]


            # Phase images
            img_path = os.path.join(data_dir, sub_id + '/dyn')
            img_list = sorted(os.listdir(img_path))
            img_list = [file for file in img_list if file[-7:] == '.nii.gz']
            if is_TTC:
                img_list = [file for file in img_list if
                            any([x in file for x in ['native', 'TTC_' + TTC_with_most_tumors, 'venous', 'delayed']])]
            else:
                img_list = [file for file in img_list if
                            any([x in file for x in ['native', 'art', 'venous', 'delayed']])]
            if len(img_list) != 4:
                print('4D image volume problem')

            if save_as_4D == True:
                run_files = [file for file in img_list if 'run' in file]
                if run_files:
                    img_list = [
                        file.replace('run-01_', '').replace('run-02_', '').replace('run-03_', '').replace('run-04_', '')
                        for file in img_list]

                img_list = sorted(img_list)
                img_list = [file for file in img_list if not any([file in run_file for run_file in run_files])]
                for run_file in run_files:
                    img_list = [run_file if os.path.basename(file)[16:] in run_file else file for file in img_list]
            img_list_path = [os.path.join(img_path, img_dir) for img_dir in img_list]

            # Liver labels
            original_liver_mask = sitk.ReadImage(liver_lab)
            spacing[sub_id] = original_liver_mask.GetSpacing()
            mask_ar = sitk.GetArrayFromImage(original_liver_mask)
            cc_mask = cc3d.connected_components(mask_ar, delta=0.1)
            counter = Counter(cc_mask.ravel())
            del counter[0]
            biggest = counter.most_common(1)
            itensity_to_keep = biggest[0][0]
            cc_mask_liver = np.where(cc_mask == itensity_to_keep, cc_mask, 0)

            liver_mask = sitk.GetImageFromArray(cc_mask_liver)
            liver_mask.SetSpacing(original_liver_mask.GetSpacing())
            liver_mask.SetDirection(original_liver_mask.GetDirection())
            liver_mask.SetOrigin(original_liver_mask.GetOrigin())
            liver_mask = sitk.Cast(liver_mask, sitk.sitkUInt8)
            liver_mask = sitk.Clamp(liver_mask, upperBound=1)

            if margin:
                radius = margin
                bd_filter = sitk.BinaryDilateImageFilter()
                bd_filter.SetForegroundValue(1)
                bd_filter.SetKernelRadius(radius)
                liver_mask = bd_filter.Execute(liver_mask)

            lsif = sitk.LabelShapeStatisticsImageFilter()
            lsif.Execute(liver_mask)
            liver_bb = lsif.GetBoundingBox(1)
            roi = sitk.RegionOfInterestImageFilter()
            roi.SetRegionOfInterest(liver_bb)

            liver_mask_roi = roi.Execute(liver_mask)
            phase_list = [[], arterial_lesion_lab, [], []]


            if copy_annotations:
                annots = [['L1', 'Lesion-1', F01], ['L2', 'Lesion-2', F02], ['L3', 'Lesion-3', F03],
                            ['L4', 'Lesion-4', F04], ['L5', 'Lesion-5', F05]]
                annots = [annot if annot[2] is not None else annot[:2] for annot in annots ]

                if sub_id == 'sub-097':
                    annots = [['L1', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06'],]

                seg_mask_per_tumor = []
                for idx, tumor_idx in enumerate(annots):
                    phases_with_tumor = [file for file in all_TTC_lesion_lab if any(x in file for x in tumor_idx)]
                    if len(phases_with_tumor) != 0:
                        art_phase = [file for file in phases_with_tumor if 'arterial' in file]
                        mask_tumor = sitk.ReadImage(phases_with_tumor[0])
                        for file in phases_with_tumor:
                            lesion_mask = sitk.ReadImage(file)
                            mask_tumor += lesion_mask
                        bounded_mask_tumor = sitk.Clamp(mask_tumor, upperBound=1)
                        bounded_mask_tumor = bounded_mask_tumor * (idx + 1)
                        seg_mask_per_tumor.append(bounded_mask_tumor)

            else:
                tumors_idx_str_list = sorted(set(['L' + file.partition('-L')[2][0] for file in all_TTC_lesion_lab]))
                seg_mask_per_tumor = []
                for idx, tumor_idx in enumerate(tumors_idx_str_list):
                    phases_with_tumor = [file for file in all_TTC_lesion_lab if tumor_idx in file]
                    if len(phases_with_tumor) != 0:
                        art_phase = [file for file in phases_with_tumor if 'arterial' in file]
                        mask_tumor = sitk.ReadImage(phases_with_tumor[0])
                        for file in phases_with_tumor:
                            lesion_mask = sitk.ReadImage(file)
                            mask_tumor += lesion_mask
                        bounded_mask_tumor = sitk.Clamp(mask_tumor, upperBound=1)
                        bounded_mask_tumor = bounded_mask_tumor * (idx+1)
                        seg_mask_per_tumor.append(bounded_mask_tumor)

            multilab_mask = sitk.Image(seg_mask_per_tumor[0].GetSize(), sitk.sitkUInt8)
            multilab_mask.SetSpacing(seg_mask_per_tumor[0].GetSpacing())
            multilab_mask.SetOrigin(seg_mask_per_tumor[0].GetOrigin())
            multilab_mask.SetDirection(seg_mask_per_tumor[0].GetDirection())
            for tumor_mask in seg_mask_per_tumor:
                tumor_mask_ar = sitk.GetArrayFromImage(tumor_mask)
                multilab_mask_ar = sitk.GetArrayFromImage(multilab_mask)
                multilab_mask_ar[(tumor_mask_ar != 0) & (multilab_mask_ar == 0)] = np.max(sitk.GetArrayFromImage(tumor_mask).ravel())
                multilab_mask = sitk.GetImageFromArray(multilab_mask_ar)
                multilab_mask.CopyInformation(tumor_mask)


            component_sizes = sitk.LabelShapeStatisticsImageFilter()
            component_sizes.Execute(sitk.Cast(multilab_mask, sitk.sitkUInt8))

            for lssif_label in component_sizes.GetLabels():
                size = component_sizes.GetPhysicalSize(lssif_label)
                if size < 100:
                    removed_small_components = sitk.BinaryThreshold(multilab_mask,
                                                                    lowerThreshold=lssif_label,
                                                                    upperThreshold=lssif_label,
                                                                    insideValue=0,
                                                                    outsideValue=1)
                    multilab_mask = sitk.Mask(multilab_mask, removed_small_components)
            component_sizes.Execute(sitk.Cast(multilab_mask, sitk.sitkUInt8))
            comp_size_values = {label: int(component_sizes.GetPhysicalSize(label)) for label in
                                component_sizes.GetLabels()}
            lesions_size[sub_id] = comp_size_values

            comp_size_for_df = []
            for df_tumor_idx in range(len(sub_df_conc)):
                try:
                    comp_size_for_df.append(comp_size_values[df_tumor_idx + 1])
                except:
                    comp_size_for_df.append('Tumor not found in files')
            sub_df_conc['Connected components size'] = comp_size_for_df

            if len(comp_size_for_df) != len(sub_df_conc):
                print('Difference between number of lesions in excel and in files')
                break

            if copy_annotations:
                patient_with_copy_annotations[sub_id] = F_annots
                lesion_per_patient_dict[sub_id] = len(F_annots)
            else:
                lesion_per_patient_dict[sub_id] = len(tumors_idx_str_list)
            
            seg_mask_per_phase = []
            for phases in phase_list:
                if len(phases) != 0:
                    lesion_mask = sitk.ReadImage(phases[0])
                    for file in phases[1:]:
                        seg_image = sitk.ReadImage(file)
                        lesion_mask += seg_image
                    seg_mask_per_phase.append(lesion_mask)

            if len(seg_mask_per_phase) == 0:
                print('No mask for patient', sub_id)
                removed_patient[sub_id] = "No tumor annotation"
                continue

            all_phases_mask = seg_mask_per_phase[0]
            if len(seg_mask_per_phase) > 1:
                for phase_mask in seg_mask_per_phase:
                    all_phases_mask += phase_mask
            final_mask = sitk.Clamp(all_phases_mask, upperBound=1)

        else:
            print('No annotations for patient', sub_id)
            if sub_id not in removed_patient.keys():
                removed_patient[sub_id] = "No tumor annotation"
            continue

        final_mask_roi = roi.Execute(final_mask)
        multilab_mask_roi = roi.Execute(multilab_mask)

        # region-based labels
        rb_array = np.zeros_like(sitk.GetArrayFromImage(final_mask_roi))
        rb_array[sitk.GetArrayFromImage(liver_mask_roi) != 0] = 1
        rb_array[sitk.GetArrayFromImage(final_mask_roi) != 0] = 2

        rb_mask = sitk.GetImageFromArray(rb_array)
        rb_mask.SetOrigin(final_mask_roi.GetOrigin())
        rb_mask.SetSpacing(final_mask_roi.GetSpacing())
        rb_mask.SetDirection(final_mask_roi.GetDirection())
        sitk.WriteImage(rb_mask, output_rb_label_path + '/' + sub_id + '.nii.gz')
        sitk.WriteImage(final_mask_roi, output_label_path + '/' + sub_id + '.nii.gz')
        sitk.WriteImage(liver_mask_roi, output_liver_label_path + '/' + sub_id + '.nii.gz')
        sitk.WriteImage(multilab_mask_roi, output_multilabel_path + '/' + sub_id + '.nii.gz')


        img_path = os.path.join(data_dir, sub_id + '/dyn')
        img_list = sorted(os.listdir(img_path))
        img_list = [file for file in img_list if file[-7:] == '.nii.gz']
        if is_TTC:
            img_list = [file for file in img_list if
                        any([x in file for x in ['native', 'TTC_' + TTC_with_most_tumors, 'venous', 'delayed']])]
        else:
            img_list = [file for file in img_list if any([x in file for x in ['native', 'art', 'venous', 'delayed']])]
        if len(img_list) != 4:
            print('4D image volume problem')

        if subtract:
            img_subtract_list = [(img_list[0], img_list[1]), (img_list[1], img_list[2]), (img_list[2], img_list[3])]
            img_subtract_list = img_subtract_list[:subtract]

        if save_as_4D == True:
            run_files = [file for file in img_list if 'run' in file]
            if run_files:
                img_list = [file.replace('run-01_', '').replace('run-02_', '').replace('run-03_', '').replace('run-04_', '')
                            for file in img_list]

            img_list = sorted(img_list)
            img_list = [file for file in img_list if not any([file in run_file for run_file in run_files])]
            for run_file in run_files:
                img_list = [run_file if os.path.basename(file)[16:] in run_file else file for file in img_list]

            total_seg_patient_dir = os.path.join(total_seg_dir, sub_id, 'dyn')
            total_seg_files = [os.path.join(total_seg_patient_dir, file) for file in os.listdir(total_seg_patient_dir) if 'venous' in file]
            total_segmentator_mask = sitk.ReadImage(total_seg_files[0])

            # save total segmentator mask
            total_segmentator_mask = roi.Execute(total_segmentator_mask)
            # remove liver masks by replacing 5 labels by 0
            total_segmentator_mask_ar = sitk.GetArrayFromImage(total_segmentator_mask)
            total_segmentator_mask_ar[total_segmentator_mask_ar == 1] = 0 # spleen to zero
            # remove all but portal_vein_and_splenic_vein 64, superior_vena_cava 62, inferior_vena_cava 63, kidneys 2, gallbladder 4, stomach 6, heart 51
            total_segmentator_mask_ar[total_segmentator_mask_ar == 64] = 1
            total_segmentator_mask_ar[total_segmentator_mask_ar == 63] = 1
            total_segmentator_mask_ar[total_segmentator_mask_ar == 2] = 1
            total_segmentator_mask_ar[total_segmentator_mask_ar != 1] = 0

            
            total_segmentator_mask_final = sitk.GetImageFromArray(total_segmentator_mask_ar)
            total_segmentator_mask_final.SetSpacing(total_segmentator_mask.GetSpacing())
            total_segmentator_mask_final.SetOrigin(total_segmentator_mask.GetOrigin())
            total_segmentator_mask_final.SetDirection(total_segmentator_mask.GetDirection())
            total_segmentator_mask_final = sitk.Cast(total_segmentator_mask_final, sitk.sitkUInt8)
            sitk.WriteImage(total_segmentator_mask_final, output_total_segmentator_path + '/' + sub_id + '.nii.gz')

            vectorOfImages = sitk.VectorOfImage()
            for img_dir in img_list:
                image = sitk.ReadImage(os.path.join(img_path, img_dir))
                img_roi = roi.Execute(image)
                img_roi = img_roi * sitk.Cast(liver_mask_roi, img_roi.GetPixelID())
                vectorOfImages.push_back(img_roi)

            # Compute subtracted images
            if subtract:
                for img_time_1, img_time_2 in img_subtract_list:
                    image_1 = sitk.ReadImage(os.path.join(img_path, img_time_1))
                    image_2 = sitk.ReadImage(os.path.join(img_path, img_time_2))
                    image_difference = image_2 - image_1
                    img_roi = roi.Execute(image_difference)
                    img_roi = img_roi * sitk.Cast(liver_mask_roi, img_roi.GetPixelID())
                    vectorOfImages.push_back(img_roi)

            image = sitk.JoinSeries(vectorOfImages)
            save_img_path = os.path.join(output_img_4D_path, sub_id + '.nii.gz')
            sitk.WriteImage(image, save_img_path)
            dimensions[sub_id] = image.GetSize()


        for img_dir in img_list:
            image = sitk.ReadImage(os.path.join(img_path, img_dir))
            img_roi = roi.Execute(image)
            img_roi = img_roi * sitk.Cast(liver_mask_roi, img_roi.GetPixelID())
            save_img_path = os.path.join(output_img_path, img_dir)
            sitk.WriteImage(img_roi, save_img_path)
            dimensions[sub_id] = img_roi.GetSize()


        save_img_sub_path = os.path.join(output_img_sub_path, sub_id)
        if subtract:
            names = ['art-nat', 'ven-art', 'del-ven']
            names = names[:subtract]
            for (img_time_1, img_time_2), name in zip(img_subtract_list, names):
                img_1 = sitk.ReadImage(os.path.join(img_path, img_time_1))
                img_2 = sitk.ReadImage(os.path.join(img_path, img_time_2))

                img_diff = img_2 - img_1
                img_diff_roi = roi.Execute(img_diff)
                img_diff_roi = img_diff_roi * sitk.Cast(liver_mask_roi, img_diff_roi.GetPixelID())
                save_img_path = save_img_sub_path + '_' +  name
                sitk.WriteImage(img_diff_roi, save_img_path + '.nii.gz')
        all_tumors_df_list.append(sub_df_conc)

    all_tumor_df = pd.concat(all_tumors_df_list)
    all_tumor_df.to_csv(os.path.join(output_derivative_path, 'tumors_characteristics.csv'))
    with open(os.path.dirname(output_img_path) + '/lesions_per_patient.txt', 'w') as file:
        for key, value in lesion_per_patient_dict.items():
            try:
                file.write(f"{key}: {value}, sizes: {lesions_size[key]}\n")
            except:
                continue
        file.write("\n")
        file.write(f"Total of {sum(lesion_per_patient_dict.values())} tumors")

    lesion_per_patient_dict = {key: len(value) for key, value in lesions_size.items()}
    LR_cnt = all_tumor_df['LIRADS'].value_counts()
    mean_diameter = np.mean(all_tumor_df['Lesion diameter'].astype(int))
    std_diameter = np.std(all_tumor_df['Lesion diameter'].astype(int))
    venous_washout = all_tumor_df['Venous washout'].astype(int).sum()
    arterial_hyper = all_tumor_df['Arterial'].astype(int).sum()
    delayed_washout = all_tumor_df['Delayed washout'].astype(int).sum()
    delayed_capsule = all_tumor_df['Delayed capsule'].astype(int).sum()
    venous_capsule = all_tumor_df['Venous capsule'].astype(int).sum()
    location = all_tumor_df['Location'].value_counts()
    total_number_of_tumors = sum(lesion_per_patient_dict.values())
    mean_lesions_per_patient = np.mean(list(lesion_per_patient_dict.values()))
    std_lesions_per_patient = np.std(list(lesion_per_patient_dict.values()))
    with open(os.path.dirname(output_img_path) + '/HCC_pos_characteristics.json', 'w') as file:
        json.dump({'mean_diameter': mean_diameter, 'std_diameter': std_diameter,
                    'venous_washout': int(venous_washout), 'arterial_hyper': int(arterial_hyper),
                    'delayed_washout': int(delayed_washout), 'delayed_capsule': int(delayed_capsule),
                    'venous_capsule': int(venous_capsule), 'location': location.to_dict(),
                    'total_number_of_tumors': total_number_of_tumors, 'mean_lesions_per_patient': mean_lesions_per_patient,
                    'std_lesions_per_patient': std_lesions_per_patient, 'LIRADS': LR_cnt.to_dict()
                     }, file, indent=4)


    with open(os.path.dirname(output_img_path) + '/removed_patients_and_lesions.txt', 'w') as file:
        for key, value in removed_patient.items():
            file.write(f"{key}: {value}\n")

    with open(os.path.dirname(output_img_path) + '/patients_with_multiple_TTC.txt', 'w') as file:
        for key, value in patient_with_multiple_TTC.items():
            file.write(f"{key}: {value}\n")
    with open(os.path.dirname(output_img_path) + '/patients_with_copy_annotations.txt', 'w') as file:
        for key, value in patient_with_copy_annotations.items():
            file.write(f"{key}: {value}\n")
    with open(os.path.dirname(output_img_path) + '/patients_with_dropped_annotations.txt', 'w') as file:
        for key, value in patient_dropped_files.items():
            file.write(f"{key}: {value}\n")


    plt.figure(figsize=(20, 7))
    plt.title('Lesions per patient based on connected components (total of {}/{} tumors)'.format(sum(lesion_per_patient_dict.values()),
                                                                   sum(number_of_tumors.values())))
    plt.ylabel('Number of lesions')
    plt.bar(number_of_tumors.keys(), number_of_tumors.values(), alpha=0.5)
    plt.bar(lesion_per_patient_dict.keys(), lesion_per_patient_dict.values(), alpha=0.5)
    plt.xticks(rotation='vertical')
    plt.ylim(0, np.max(list(lesion_per_patient_dict.values())))
    plt.legend(['Ground truth', 'Tumors in the dataset'])
    # plt.show(block=True)
    plt.savefig(os.path.dirname(output_img_path) + '/lesions_per_patient_connect_comp.png')
    plt.close()


    plt.figure(figsize=(20, 7))
    plt.title('Lesions per patient based on annotation file names (total of {}/{} tumors)'.format(sum(annotations_count.values()),
                                                                   sum(number_of_tumors.values())))
    plt.ylabel('Number of lesions')
    plt.bar(number_of_tumors.keys(), number_of_tumors.values(), alpha=0.5)
    plt.bar(annotations_count.keys(), annotations_count.values(), alpha=0.5)
    plt.xticks(rotation='vertical')
    plt.ylim(0, np.max(list(lesion_per_patient_dict.values())))
    plt.legend(['Ground truth', 'Tumors in the dataset'])
    # plt.show(block=True)
    plt.savefig(os.path.dirname(output_img_path) + '/lesions_per_patient_file_name.png')
    plt.close()

    all_lesions = list([list(item.values()) for item in lesions_size.values()])
    all_lesions_flat = [sub_lesions for lesion in all_lesions for sub_lesions in lesion]

    plt.figure(figsize=(10, 7))
    plt.title('Lesion size per patient')
    plt.ylabel('Lesion volume (mm³)')
    plt.bar(np.arange(len(all_lesions_flat)), all_lesions_flat)
    plt.xticks(rotation='vertical')
    plt.savefig(os.path.dirname(output_img_path) + '/lesion_size_per_patient.png')
    plt.close()

    with open(os.path.dirname(output_img_path) + '/lesions_per_phase.txt', 'w') as file:
        for key, value in lesion_per_patient_dict.items():
            try:
                file.write(f"{key}: {value}, lesions per phase: {lesions_per_phase[key]}\n")
            except:
                continue
        file.write("\n")
        file.write(f"Total of {sum(lesion_per_patient_dict.values())} tumors")

    with open(os.path.dirname(output_img_path) + '/spacing.txt', 'w') as file:
        for key, value in spacing.items():
            file.write(f"{key}: {value}\n")

    data = [[value[0] for key, value in spacing.items()],
            [value[1] for key, value in spacing.items()],
            [value[2] for key, value in spacing.items()]]

    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data)
    ax.set_xticklabels(['X-axis', 'Y-axis', 'Z-axis'])
    ax.set_title('Voxels spacing')
    ax.set_xlabel('Axis')
    ax.set_ylabel('Spacing')
    medians = [item.get_ydata().mean() for item in boxplot['medians']]
    ax.text(1+0.35, medians[0], f'{medians[0]:.2f}', horizontalalignment='center')
    ax.text(2+0.35, medians[1], f'{medians[1]:.2f}', horizontalalignment='center')
    ax.text(3+0.35, medians[2], f'{medians[2]:.2f}', horizontalalignment='center')
    plt.savefig(os.path.dirname(output_img_path) + '/voxel_spacing.png')
    plt.close()

    with open(os.path.dirname(output_img_path) + '/dimensions.txt', 'w') as file:
        for key, value in dimensions.items():
            file.write(f"{key}: {value}\n")

    data = [[value[0] for key, value in dimensions.items()],
            [value[1] for key, value in dimensions.items()],
            [value[2] for key, value in dimensions.items()]]

    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data)
    ax.set_xticklabels(['X-axis', 'Y-axis', 'Z-axis'])
    ax.set_title('Images dimensions')
    ax.set_xlabel('Axis')
    ax.set_ylabel('Dimensions')
    medians = [item.get_ydata().mean() for item in boxplot['medians']]
    ax.text(1+0.35, medians[0]-5, f'{medians[0]:.2f}', horizontalalignment='center')
    ax.text(2+0.35, medians[1]-5, f'{medians[1]:.2f}', horizontalalignment='center')
    ax.text(3+0.35, medians[2]-5, f'{medians[2]:.2f}', horizontalalignment='center')
    plt.savefig(os.path.dirname(output_img_path) + '/image_dimensions.png')
    plt.close()


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    data_dir = os.path.join(main_dir, "derivatives/6_T1_registration_groupwise")
    seg_dir = os.path.join(main_dir, "derivatives/9_3_T1_liver_tumor_segmentation_transformed_full")
    liver_seg_dir = os.path.join(main_dir, "derivatives/9_3_T1_liver_segmentation_transformed_nnunet")
    total_seg_dir = os.path.join(main_dir, "derivatives/9_total_segmentator")
    output_derivative_path = os.path.join(main_dir, 'derivatives/10_T1_dataset')
    main(output_derivative_path, data_dir, seg_dir, liver_seg_dir, total_seg_dir, margin=0, save_as_4D=True, subtract=False)
