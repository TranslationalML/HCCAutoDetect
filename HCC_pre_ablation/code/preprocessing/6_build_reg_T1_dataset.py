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

    path = os.path.join(main_dir, "sourcedata/annotations.xlsx")
    df = pd.read_excel(path, dtype=str, header=[0], sheet_name='Feuil3')

    # match patient names
    match_patient_names = os.path.join(main_dir, "code/quality_control/match_patient_names.csv")
    df_match_names = pd.read_csv(match_patient_names)


    liver_labels = sorted([os.path.join(dir_, file) for dir_, subdirs, files in os.walk(liver_seg_dir) for file in files])
    lesion_labels = sorted([os.path.join(dir_, file) for dir_, subdirs, files in os.walk(tumor_seg_dir) for file in files])

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
    annot_found_vs_gt = []
    IPP_list = []

    for patient, sub_id in participants[['Patient', 'Sub-ID']].values:
        patient_dyn_files_to_select = os.path.join(os.path.dirname(output_derivative_path) + '/3_select_dyn_phases', sub_id)
        print(sub_id)
        IPP = df_match_names.loc[df_match_names['Sub-ID'] == sub_id]['Original Name'].values[0]
        IPP = IPP[4:]

        if IPP not in df['IPP'].values:
            removed_patient[sub_id] = "Patient " + IPP + " excluded"
            continue
        try:
            liver_lab = [file for file in liver_labels if sub_id in file][0]
        except:
            print('No Liver label for patient ', sub_id)
            removed_patient[sub_id] = "No Liver segmentation"
            lesion_per_patient_dict[sub_id] = 0
            lesions_per_phase[sub_id] = 0
            continue

        sub_df = df.loc[df['IPP'] == IPP]
        sub_df['sub-ID'] = sub_id

        lr_scores = ['L1 Lr score', 'L2 Lr score', 'L3 Lr score', 'L4 Lr score', 'L5 Lr score']
        lesions_annot_gt = sub_df[lr_scores]
        lesion_annot_gt_names = lesions_annot_gt.columns[~lesions_annot_gt.iloc[0].isna()].to_list()
        lesion_annot_gt_names = [elem[:2] for elem in lesion_annot_gt_names]

        #img
        patient_dyn_path = os.path.join(data_dir, sub_id + '/dyn')

        img_list = [os.path.join(patient_dyn_path, file_name) for file_name in os.listdir(patient_dyn_path)]

        all_lesion_lab = [file for file in lesion_labels if sub_id in file and any([lesion_name in file 
                                                                                    for lesion_name in ['L1', 'L2', 'L3', 'L4', 'L5']])]

        annot_count = ['L1' if 'L1' in file 
                       else 'L2' if 'L2' in file 
                       else 'L3' if 'L3' in file 
                       else 'L4' if 'L4' in file 
                       else 'L5' if 'L5' in file 
                       else 0 for file in all_lesion_lab]
        annotations_count[sub_id] = len(annot_count)

        annotated_lesions = set(annot_count)
        columns_to_keep = []
        columns_to_keep.append('sub-ID')
        for lesion in annotated_lesions:
            L1_score = lesion + ' Lr score'
            L1_size = lesion + ' taille'
            L1_loc = lesion + ' segment'
            columns_to_keep.extend([L1_score, L1_size, L1_loc])
        sub_df = sub_df[columns_to_keep]

        if len(all_lesion_lab) == 0:
            removed_patient[sub_id] = "No lesion segmentation"
            lesion_per_patient_dict[sub_id] = 0
            lesions_per_phase[sub_id] = 0
            continue
        else:
            liver_lab = [file for file in liver_labels if sub_id in file and 'venous' in file][0]

            lesion_name_list = ['L1', 'L2', 'L3', 'L4', 'L5']

            dyn_files_path = (patient_dyn_files_to_select + '/dyn/phases_to_keep.json')

            with open(dyn_files_path, 'r') as f:
                name_list = json.load(f)
                files_to_keep = [file_seg for file_seg in all_lesion_lab if any([os.path.basename(file)[:-7] in file_seg for file in name_list])]
                dropped_files = [file_seg for file_seg in all_lesion_lab if file_seg not in files_to_keep]
                patient_dropped_files[sub_id] = dropped_files
                all_lesion_lab = files_to_keep


            native_lesion_lab = [file for file in all_lesion_lab if sub_id in file and 'native' in file
                                 and any(x in file for x in lesion_name_list)]
            arterial_lesion_lab = [file for file in all_lesion_lab if sub_id in file and
                                   any([x in file for x in ['arterial']]) and
                                   any(x in file for x in lesion_name_list)]
            venous_lesion_lab = [file for file in all_lesion_lab if sub_id in file and 'venous' in file
                                 and any(x in file for x in lesion_name_list)]
            delayed_lesion_lab = [file for file in all_lesion_lab if sub_id in file and 'delayed' in file
                                  and any(x in file for x in lesion_name_list)]

            # Phase images
            img_path = os.path.join(data_dir, sub_id + '/dyn')
            img_list = sorted(os.listdir(img_path))
            img_list = [file for file in img_list if file[-7:] == '.nii.gz']

            img_list = [file for file in img_list if
                        any([x in file for x in ['native', 'art', 'venous', 'delayed']])]
            img_list = sorted(img_list, key=lambda s: s.split("_phase_")[1])
            if len(img_list) != 4:
                removed_patient[sub_id] = "Not all phases found"
                print('4D image volume problem')

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

            phase_list = [native_lesion_lab, arterial_lesion_lab, venous_lesion_lab, delayed_lesion_lab]

            tumors_idx_str_list = sorted(set(['L' + file.partition('_L')[2][0] for file in all_lesion_lab]))
            seg_mask_per_tumor = []
            for idx, tumor_idx in enumerate(tumors_idx_str_list):
                phases_with_tumor = [file for file in all_lesion_lab if tumor_idx in file]
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

            component_sizes.Execute(sitk.Cast(multilab_mask, sitk.sitkUInt8))
            comp_size_values = {label: int(component_sizes.GetPhysicalSize(label)) for label in
                                component_sizes.GetLabels()}
            lesions_size[sub_id] = comp_size_values

            comp_size_for_df = []
            for df_tumor_idx in range(len(annotated_lesions)):
                try:
                    comp_size_for_df.append(comp_size_values[df_tumor_idx + 1])
                except:
                    comp_size_for_df.append('Tumor not found in files')

            df_list = []
            for idx, lesion_name in enumerate(sorted(annotated_lesions)):
                df_new = pd.DataFrame([idx+1, sub_df['sub-ID'].values[0], sub_df[lesion_name + ' Lr score'].values[0], sub_df[lesion_name + ' taille'].values[0], sub_df[lesion_name + ' segment'].values[0],
                                       lesion_name, df.loc[df['IPP'] == IPP][lesion_name + ' HCC y/n'].values[0], comp_size_for_df[idx]], 
                                       index=['label', 'ID', 'LIRADS', 'Lesion diameter', 'location', 'lesion', 'HCC', 'Connected components size']).T
                df_new['HCC'] = df_new['HCC'].replace({np.nan: 0})
                df_list.append(df_new)
            df_conc = pd.concat(df_list, axis=0)

            if not all([gt_lesion_name in list(df_conc['lesion'].values) for gt_lesion_name in lesion_annot_gt_names]):
                removed_patient[sub_id] = "Missing lesion annotations"
                print('Missing lesion annotations for patient', sub_id)

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
                lesion_per_patient_dict[sub_id] = 0
                removed_patient[sub_id] = "No lesion segmentation"
                continue

            all_phases_mask = seg_mask_per_phase[0]
            for phase_mask in seg_mask_per_phase:
                all_phases_mask += phase_mask
            final_mask = sitk.Clamp(all_phases_mask, upperBound=1)

            all_tumors_df_list.append(df_conc)
            annot_found_vs_gt.append({sub_id: {'found': len(annotated_lesions), 'gt': len(lesion_annot_gt_names)}})
            

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

        img_list = [file for file in img_list if any([x in file for x in ['native', 'art', 'venous', 'delayed']])]
        img_list = sorted(img_list, key=lambda s: s.split("_phase_")[1])

        if (len(img_list) != 4 or not 'native' in img_list[0] 
            or not 'art' in img_list[1] or not 'venous' in img_list[2] 
            or not 'delayed' in img_list[3]):
            removed_patient[sub_id] = "Not all phases found"
            print('4D image volume problem')

        if subtract:
            img_subtract_list = [(img_list[0], img_list[1]), (img_list[1], img_list[2]), (img_list[2], img_list[3])]
            img_subtract_list = img_subtract_list[:subtract]

        if save_as_4D == True:

            total_seg_patient_dir = os.path.join(total_seg_dir, sub_id, 'dyn')
            total_seg_files = [os.path.join(total_seg_patient_dir, file) for file in os.listdir(total_seg_patient_dir) if 'venous' in file]
            total_segmentator_mask = sitk.ReadImage(total_seg_files[0])
            # save total segmentator mask
            total_segmentator_mask = roi.Execute(total_segmentator_mask)
            total_segmentator_mask_ar = sitk.GetArrayFromImage(total_segmentator_mask)
            total_segmentator_mask_ar[total_segmentator_mask_ar == 1] = 0 # spleen to zero
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
                image.CopyInformation(liver_mask)
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
            image.CopyInformation(liver_mask)
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
        IPP_list.append(IPP)
    patient_in_df_not_included = [sub_id for sub_id in df['IPP'].values if sub_id not in IPP_list]
    all_tumor_df = pd.concat(all_tumors_df_list)
    match_columns = ['Arterial', 'Venous washout', 'Venous capsule', 'Delayed washout', 'Delayed capsule']
    for column in match_columns:
        all_tumor_df[column] = np.nan

    all_tumor_df.to_csv(os.path.join(output_derivative_path, 'tumors_characteristics.csv'), index=False)
    all_tumor_df.to_csv(os.path.join(output_derivative_path, 'tumors_characteristics_semi_col.csv'), index=False, sep=';')
    
    patient_in_df_not_included = [sub_id for sub_id in df.keys() if sub_id not in lesion_per_patient_dict.keys()]

    with open(os.path.dirname(output_img_path) + '/lesions_per_patient.txt', 'w') as file:
        for key, value in lesion_per_patient_dict.items():
            try:
                file.write(f"{key}: {value}, sizes: {lesions_size[key]}\n")
            except:
                continue
        file.write("\n")
        file.write(f"Total of {sum(lesion_per_patient_dict.values())} tumors")


    with open(os.path.dirname(output_img_path) + '/annot_found_vs_gt.txt', 'w') as file:
        for item in annot_found_vs_gt:
            file.write(f"{item}\n")

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
    data_dir = os.path.join(main_dir, "derivatives/5_T1_registration_groupwise")
    seg_dir = os.path.join(main_dir, "derivatives/5_tumor_segmentations_transformed")
    liver_seg_dir = os.path.join(main_dir, "derivatives/5_liver_masks_registered")
    total_seg_dir = os.path.join(main_dir, "derivatives/5_total_segmentator")
    output_derivative_path = os.path.join(main_dir, 'derivatives/6_T1_dataset')
    main(output_derivative_path, data_dir, seg_dir, liver_seg_dir, total_seg_dir, margin=0, save_as_4D=True, subtract=False)
