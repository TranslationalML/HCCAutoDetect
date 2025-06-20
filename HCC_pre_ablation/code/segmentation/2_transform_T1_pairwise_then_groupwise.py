import os
import pandas as pd
import SimpleITK as sitk


def main(seg_path, orig_dyn_select_path, pairwise_reg_path, groupwise_reg_path, output_derivative_path):
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')
    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:
        print(patient_id)
        patient_pairwise_reg_path = os.path.join(pairwise_reg_path, patient_id)
        patient_groupwise_reg_path = os.path.join(groupwise_reg_path, patient_id)
        patient_seg_path = os.path.join(seg_path, patient_id)
        patient_files = os.path.join(orig_dyn_select_path, patient_id)
        pairwise_reg_dyn_path = os.path.join(patient_pairwise_reg_path, 'dyn/transforms')
        groupwise_reg_dyn_path = os.path.join(patient_groupwise_reg_path, 'dyn/transforms')

        try:
            pairwise_files = os.listdir(pairwise_reg_dyn_path)
        except:
            continue

        if os.path.exists(groupwise_reg_dyn_path):
            groupwise_files = os.listdir(groupwise_reg_dyn_path)
            groupwise_transform = [os.path.join(groupwise_reg_dyn_path, file)
                                   for file in groupwise_files if 'Transform' in file][0]
        else:
            groupwise_transform = None

        seg_dyn_path = os.path.join(patient_seg_path, 'dyn')
        if not os.path.exists(seg_dyn_path) and all(any(x in y for y in seg_dyn_path) for x in ['native', 'arterial', 'venous', 'delayed']):
            continue

        patient_derivative_path = os.path.join(output_derivative_path, patient_id)
        os.makedirs(patient_derivative_path, exist_ok=True)
        registration_output_path = os.path.join(patient_derivative_path, 'dyn')
        os.makedirs(registration_output_path, exist_ok=True)

        pairwise_transforms = sorted([file for file in pairwise_files if 'Transform' in file])
        if len(pairwise_transforms) == 0:
            continue

        try:
            seg_file_list = os.listdir(seg_dyn_path)
            seg_file_list = [file for file in seg_file_list if 'nii.gz' in file]
            patient_dyn_path = os.path.join(patient_files, 'dyn')
        except:
            print('No segmentation files found for ', patient_id)
            continue

        nat_art_0 = [os.path.join(pairwise_reg_dyn_path, file) for file in pairwise_transforms if
                        '0_nat_to_art.txt' in file][0]
        nat_art_1 = [os.path.join(pairwise_reg_dyn_path, file) for file in pairwise_transforms if
                        '1_nat_to_art.txt' in file][0]
        art_ven_0 = [os.path.join(pairwise_reg_dyn_path, file) for file in pairwise_transforms if
                        '0_art_to_venous.txt' in file][0]
        art_ven_1 = [os.path.join(pairwise_reg_dyn_path, file) for file in pairwise_transforms if
                        '1_art_to_venous.txt' in file][0]
        del_ven_0 = [os.path.join(pairwise_reg_dyn_path, file) for file in pairwise_transforms if
                        '0_delayed_to_venous.txt' in file][0]
        del_ven_1 = [os.path.join(pairwise_reg_dyn_path, file) for file in pairwise_transforms if
                        '1_delayed_to_venous.txt' in file][0]

        # pairwise registration
        # caipi
        nat_art_11 = sitk.ReadParameterFile(nat_art_0)
        nat_art_12 = sitk.ReadParameterFile(nat_art_1)
        art_ven_41 = sitk.ReadParameterFile(art_ven_0)
        art_ven_42 = sitk.ReadParameterFile(art_ven_1)
        del_ven_51 = sitk.ReadParameterFile(del_ven_0)
        del_ven_52 = sitk.ReadParameterFile(del_ven_1)
        nat_art_11['FinalBSplineInterpolationOrder'] = ('0',)
        nat_art_12['FinalBSplineInterpolationOrder'] = ('0',)
        art_ven_41['FinalBSplineInterpolationOrder'] = ('0',)
        art_ven_42['FinalBSplineInterpolationOrder'] = ('0',)
        del_ven_51['FinalBSplineInterpolationOrder'] = ('0',)
        del_ven_52['FinalBSplineInterpolationOrder'] = ('0',)

        for seg_file in seg_file_list:
            liver_lesion_seg_path = os.path.join(seg_dyn_path, seg_file)

            if 'native' in seg_file:
                # align native to ven
                transformix_image_filter = sitk.TransformixImageFilter()
                transformix_image_filter.SetMovingImage(sitk.ReadImage(liver_lesion_seg_path))
                transformix_image_filter.SetTransformParameterMap(nat_art_11)
                transformix_image_filter.AddTransformParameterMap(nat_art_12)
                transformix_image_filter.AddTransformParameterMap(art_ven_41)
                transformix_image_filter.AddTransformParameterMap(art_ven_42)
                transformix_image_filter.Execute()

            if 'art' in seg_file:
                # align TTC1 to ven
                transformix_image_filter = sitk.TransformixImageFilter()
                transformix_image_filter.SetMovingImage(sitk.ReadImage(liver_lesion_seg_path))
                transformix_image_filter.AddTransformParameterMap(art_ven_41)
                transformix_image_filter.AddTransformParameterMap(art_ven_42)
                transformix_image_filter.Execute()

            if 'delayed' in seg_file:
                # align delayed to ven
                transformix_image_filter = sitk.TransformixImageFilter()
                transformix_image_filter.SetMovingImage(sitk.ReadImage(liver_lesion_seg_path))
                transformix_image_filter.AddTransformParameterMap(del_ven_51)
                transformix_image_filter.AddTransformParameterMap(del_ven_52)
                transformix_image_filter.Execute()


            if 'venous' in seg_file:
                res_img = sitk.ReadImage(liver_lesion_seg_path)
            else:
                res_img = transformix_image_filter.GetResultImage()

            saving_name = seg_file
            if groupwise_transform:
                if 'native' in liver_lesion_seg_path:
                    position = 0
                elif 'arterial' in liver_lesion_seg_path:
                    position = 1
                elif 'venous' in liver_lesion_seg_path:
                    position = 2
                elif 'delayed' in liver_lesion_seg_path:
                    position = 3
                else:
                    print('No phase position found')
                    break

                population = [liver_lesion_seg_path] * 4

                vectorOfImages = sitk.VectorOfImage()
                for filename in population:
                    vectorOfImages.push_back(res_img)
                image = sitk.JoinSeries(vectorOfImages)

                transformixImageFilter = sitk.TransformixImageFilter()
                parametermap = transformixImageFilter.ReadParameterFile(groupwise_transform)
                parametermap['FinalBSplineInterpolationOrder'] = ('0',)
                transformixImageFilter.SetTransformParameterMap(parametermap)
                transformixImageFilter.SetMovingImage(image)
                transformixImageFilter.Execute()

                transformed_img = transformixImageFilter.GetResultImage()

                sitk.WriteImage(transformed_img[:, :, :, position], os.path.join(registration_output_path, saving_name))
            else:
                venous_img = [os.path.join(patient_dyn_path, file) for file in os.listdir(patient_dyn_path)
                                if ('venous' in file) and 'nii.gz' in file][0]
                ven_sitk_img = sitk.ReadImage(venous_img)
                res_img.SetOrigin(ven_sitk_img.GetOrigin())
                res_img.SetDirection(ven_sitk_img.GetDirection())
                res_img.SetSpacing(ven_sitk_img.GetSpacing())
                sitk.WriteImage(res_img, os.path.join(registration_output_path, saving_name))


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))

    orig_dyn_select_path = os.path.join(main_dir, 'derivatives/3_select_dyn_phases_with_mask')
    seg_path = os.path.join(main_dir, 'derivatives/tumor_segmentations')
    registration_path_1 = os.path.join(main_dir, 'derivatives/5_1_T1_registration_pairwise')
    registration_path_2 = os.path.join(main_dir, 'derivatives/5_2_T1_registration_groupwise')
    output_path = os.path.join(main_dir, 'derivatives/5_tumor_segmentations_transformed')
    os.makedirs(output_path, exist_ok=True)
    main(seg_path, orig_dyn_select_path, registration_path_1, registration_path_2, output_path)

