import SimpleITK as sitk
import numpy as np
import os
from collections import Counter
import cc3d
import pandas as pd
from matplotlib import pyplot as plt

def main(output_derivative_path, data_dir, tumor_seg_dir, liver_seg_dir, margin, save_as_4D, subtract):
    os.makedirs(output_derivative_path, exist_ok=True)

    patient_cnt = 0
    for sub_dir in os.listdir(data_dir):
        files = os.listdir(os.path.join(data_dir, sub_dir, 'dyn'))
        if len(files) != 0:
            patient_cnt += 1


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

    liver_labels = sorted([os.path.join(dir_, file) for dir_, subdirs, files in os.walk(liver_seg_dir) for file in files
                if 'venous' in file])

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
    participants = pd.read_csv((main_dir + '/participants_more_infos.tsv'), sep='\t', index_col=0)
    all_tumors_df_list = []
    for patient_id, row in participants.iterrows():
        df = pd.DataFrame([[1, patient_id, 0, 0, 0, 0, 0, 5, 0, 0, 0]], columns=['label',
                                                                                'ID',
                                                                                'Arterial',
                                                                                'Venous washout',
                                                                                'Venous capsule',
                                                                                'Delayed washout',
                                                                                'Delayed capsule',
                                                                                'LIRADS',
                                                                                'Location',
                                                                                'Lesion diameter',
                                                                                'Connected components size'])
        df['Lesion_type'] = row['C+A_lesions'][2:-2]
        print(patient_id)
        print(row['C+A_lesions'])

        patient_liver_seg_path = os.path.join(liver_seg_dir, patient_id, 'dyn')
        if os.path.exists(patient_liver_seg_path) == False:
            print('No liver segmentation for patient', patient_id)
            removed_patient[patient_id] = "No Liver segmentation"
            lesion_per_patient_dict[patient_id] = 0
            lesions_per_phase[patient_id] = 0
            continue
        patient_liver_lab = [os.path.join(patient_liver_seg_path, file) for file in os.listdir(patient_liver_seg_path) if patient_id in file]

        if len(patient_liver_lab) == 0:
            print('No Liver label for patient ', patient_id)
            removed_patient[patient_id] = "No Liver segmentation"
            lesion_per_patient_dict[patient_id] = 0
            lesions_per_phase[patient_id] = 0
            continue

        #img
        patient_dyn_path = os.path.join(data_dir, patient_id, 'dyn')
        img_list = [os.path.join(patient_dyn_path, file_name) for file_name in os.listdir(patient_dyn_path) if not 'transforms' in file_name]


        patient_lesions_path = os.path.join(tumor_seg_dir, patient_id, 'dyn')
        if os.path.exists(patient_lesions_path) == False:
            print('No annotations for patient', patient_id)
            removed_patient[patient_id] = "No tumor annotation"
            lesion_per_patient_dict[patient_id] = 0
            lesions_per_phase[patient_id] = 0
            continue
        patient_lesions_lab = [os.path.join(patient_lesions_path, file) for file in os.listdir(patient_lesions_path) if patient_id in file]

        if len(patient_lesions_lab) == 4:
            liver_labels = [sitk.ReadImage(liver_lab) for liver_lab in patient_liver_lab]
            #merge maps
            final_liver_lab = (liver_labels[0] + liver_labels[1] + liver_labels[2] + liver_labels[3])
            final_liver_lab = sitk.Cast(final_liver_lab, sitk.sitkUInt8)
            final_liver_lab = sitk.Clamp(final_liver_lab, lowerBound=0, upperBound=1)


            # Phase images
            img_path = os.path.join(data_dir, patient_id + '/dyn')
            img_list = sorted(os.listdir(img_path))
            img_list = [file for file in img_list if file[-7:] == '.nii.gz']
            img_list = [file for file in img_list if any([x in file for x in ['native', 'art', 'venous', 'delayed']])]

            if len(img_list) != 4:
                print('4D image volume problem')
            if save_as_4D == True:
                img_list = sorted(img_list)

            # Liver labels
            original_liver_mask = final_liver_lab
            spacing[patient_id] = original_liver_mask.GetSpacing()
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


            multilab_mask = sitk.ReadImage(sorted(patient_lesions_lab)[2])
            component_sizes = sitk.LabelShapeStatisticsImageFilter()
            component_sizes.Execute(sitk.Cast(multilab_mask, sitk.sitkUInt8))

            comp_size_values = {label: int(component_sizes.GetPhysicalSize(label)) for label in
                                component_sizes.GetLabels()}
            lesions_size[patient_id] = comp_size_values
            df['Connected components size'] = comp_size_values[1]
            all_tumors_df_list.append(df)

            final_mask = sitk.ReadImage(sorted(patient_lesions_lab)[2])

        else:
            print('No annotations for patient', patient_id)
            if patient_id not in removed_patient.keys():
                removed_patient[patient_id] = "No tumor annotation"
            lesion_per_patient_dict[patient_id] = 0
            lesions_per_phase[patient_id] = 0
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
        sitk.WriteImage(rb_mask, output_rb_label_path + '/' + patient_id + '.nii.gz')
        sitk.WriteImage(final_mask_roi, output_label_path + '/' + patient_id + '.nii.gz')
        sitk.WriteImage(liver_mask_roi, output_liver_label_path + '/' + patient_id + '.nii.gz')
        sitk.WriteImage(multilab_mask_roi, output_multilabel_path + '/' + patient_id + '.nii.gz')


        img_path = os.path.join(data_dir, patient_id + '/dyn')
        img_list = sorted(os.listdir(img_path))
        img_list = [file for file in img_list if file[-7:] == '.nii.gz']

        img_list = [file for file in img_list if any([x in file for x in ['native', 'art', 'venous', 'delayed']])]
        if len(img_list) != 4:
            print('4D image volume problem')

        if subtract:
            img_subtract_list = [(img_list[0], img_list[1]), (img_list[1], img_list[2]), (img_list[2], img_list[3])]
            img_subtract_list = img_subtract_list[:subtract]

        if save_as_4D == True:
            img_list = sorted(img_list)

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
            save_img_path = os.path.join(output_img_4D_path, patient_id + '.nii.gz')
            sitk.WriteImage(image, save_img_path)
            dimensions[patient_id] = image.GetSize()

        img_1 = sitk.ReadImage(os.path.join(img_path, img_list[0]))
        img_1_roi = roi.Execute(img_1)
        img_1_roi = img_1_roi * sitk.Cast(liver_mask_roi, img_roi.GetPixelID())
        for img_dir in img_list:
            image = sitk.ReadImage(os.path.join(img_path, img_dir))
            img_roi = roi.Execute(image)
            img_roi = img_roi * sitk.Cast(liver_mask_roi, img_roi.GetPixelID())
            img_roi.CopyInformation(img_1_roi)
            save_img_path = os.path.join(output_img_path, img_dir)
            sitk.WriteImage(img_roi, save_img_path)
            dimensions[patient_id] = img_roi.GetSize()

        save_img_sub_path = os.path.join(output_img_sub_path, patient_id)
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
    data_dir = os.path.join(main_dir, "derivatives/6_2_T1_registration_groupwise")
    seg_dir = os.path.join(main_dir, "derivatives/7_tumor_segmentation_transformed")
    liver_seg_dir = os.path.join(main_dir, "derivatives/8_liver_masks_corrected/liver_masks_no_margin_original_threshold")
    output_derivative_path = os.path.join(main_dir, 'derivatives/9_T1_dataset_all_lesions')
    main(output_derivative_path, data_dir, seg_dir, liver_seg_dir, margin=0, save_as_4D=True, subtract=False)
