import os
import pandas as pd
import SimpleITK as sitk

def main(seg_path, orig_dyn_select_path, pairwise_reg_path, groupwise_reg_path, output_derivative_path):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t', index_col=0)
    for patient_id, row in participants.iterrows():
        print(patient_id)
        try:
            patient_pairwise_reg_path = os.path.join(pairwise_reg_path, patient_id)
            patient_groupwise_reg_path = os.path.join(groupwise_reg_path, patient_id)
            seg_dyn_path = os.path.join(seg_path, patient_id, 'dyn')
            patient_files = os.path.join(orig_dyn_select_path, patient_id)
            pairwise_reg_dyn_path = os.path.join(patient_pairwise_reg_path, 'dyn/transforms')
            groupwise_reg_dyn_path = os.path.join(patient_groupwise_reg_path, 'dyn/transforms')

            try:
                pairwise_files = os.listdir(pairwise_reg_dyn_path)
            except:
                with open('missing_pairwise_reg.txt', 'a') as f:
                    f.write(patient_id + '\n')
                print('Pairwise registration missing for patient: ', patient_id)
                continue

            if os.path.exists(groupwise_reg_dyn_path):
                if len(os.listdir(groupwise_reg_dyn_path)) > 1:
                    groupwise_files = os.listdir(groupwise_reg_dyn_path)
                    groupwise_transform = [os.path.join(groupwise_reg_dyn_path, file)
                                        for file in groupwise_files if 'Transform' in file][0]
                else:
                    with open('missing_groupwise_reg.txt', 'a') as f:
                        f.write(patient_id + '\n')
                    print('Groupwise registration missing for patient: ', patient_id)
                    continue
            else:
                with open('missing_groupwise_reg.txt', 'a') as f:
                    f.write(patient_id + '\n')
                print('Groupwise registration missing for patient: ', patient_id)
                continue

            if not os.path.exists(seg_dyn_path):
                with open('missing_segmentation.txt', 'a') as f:
                    f.write(patient_id + '\n')
                print('Segmentation missing for ', patient_id)
                continue


            registration_output_path = os.path.join(output_derivative_path, patient_id, 'dyn')
            os.makedirs(registration_output_path, exist_ok=True)
            if len(os.listdir(registration_output_path)) > 1:
                continue
            
            pairwise_transforms = sorted([file for file in pairwise_files if 'Transform' in file])
            if len(pairwise_transforms) != 6:
                with open('missing_pairwise_reg.txt', 'a') as f:
                    f.write(patient_id + '\n')
                print('Pairwise registration missing for patient: ', patient_id)
                continue
            seg_file_list = os.listdir(seg_dyn_path)
            seg_file_list = [file for file in seg_file_list if 'nii.gz' in file]
            patient_dyn_path = os.path.join(patient_files, 'dyn')

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
                    original_img = sitk.ReadImage([os.path.join(patient_dyn_path, file) for file in os.listdir(patient_dyn_path) if 'native' in file][0], sitk.sitkUInt8)
                    seg = sitk.ReadImage(liver_lesion_seg_path, sitk.sitkUInt8)
                    seg.CopyInformation(original_img)

                    transformix_image_filter = sitk.TransformixImageFilter()
                    transformix_image_filter.SetMovingImage(seg)
                    transformix_image_filter.SetTransformParameterMap(nat_art_11)
                    transformix_image_filter.AddTransformParameterMap(nat_art_12)
                    transformix_image_filter.AddTransformParameterMap(art_ven_41)
                    transformix_image_filter.AddTransformParameterMap(art_ven_42)
                    transformix_image_filter.Execute()
                    res_img = transformix_image_filter.GetResultImage()

                elif 'art' in seg_file:
                    # align TTC1 to ven
                    original_img = sitk.ReadImage([os.path.join(patient_dyn_path, file) for file in os.listdir(patient_dyn_path) if 'arterial' in file][0], sitk.sitkUInt8)
                    seg = sitk.ReadImage(liver_lesion_seg_path, sitk.sitkUInt8)
                    seg.SetOrigin(original_img.GetOrigin())
                    seg.SetSpacing(original_img.GetSpacing())
                    seg.SetDirection(original_img.GetDirection())

                    transformix_image_filter = sitk.TransformixImageFilter()
                    transformix_image_filter.SetMovingImage(seg)
                    transformix_image_filter.AddTransformParameterMap(art_ven_41)
                    transformix_image_filter.AddTransformParameterMap(art_ven_42)
                    transformix_image_filter.Execute()
                    res_img = transformix_image_filter.GetResultImage()

                elif 'delayed' in seg_file:
                    # align delayed to ven
                    original_img = sitk.ReadImage([os.path.join(patient_dyn_path, file) for file in os.listdir(patient_dyn_path) if 'delayed' in file][0], sitk.sitkUInt8)
                    seg = sitk.ReadImage(liver_lesion_seg_path, sitk.sitkUInt8)
                    seg.CopyInformation(original_img)
                    transformix_image_filter = sitk.TransformixImageFilter()
                    transformix_image_filter.SetMovingImage(seg)
                    transformix_image_filter.AddTransformParameterMap(del_ven_51)
                    transformix_image_filter.AddTransformParameterMap(del_ven_52)
                    transformix_image_filter.Execute()
                    res_img = transformix_image_filter.GetResultImage()
                
                elif 'venous' in seg_file:
                    original_img = sitk.ReadImage([os.path.join(patient_dyn_path, file) for file in os.listdir(patient_dyn_path) if 'venous' in file][0], sitk.sitkUInt8)
                    res_img = sitk.ReadImage(liver_lesion_seg_path, sitk.sitkUInt8)
                    res_img.CopyInformation(original_img)

                else:
                    continue

                res_img = sitk.Cast(res_img, sitk.sitkUInt8)
                saving_name = seg_file
            
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

                transformixImageFilter_grpwise = sitk.TransformixImageFilter()
                parametermap = transformixImageFilter_grpwise.ReadParameterFile(groupwise_transform)
                parametermap['FinalBSplineInterpolationOrder'] = ('0',)
                transformixImageFilter_grpwise.SetMovingImage(image)
                transformixImageFilter_grpwise.AddTransformParameterMap(parametermap)
                try:
                    transformixImageFilter_grpwise.Execute()
                    transformed_img = transformixImageFilter_grpwise.GetResultImage()
                    transformed_img = sitk.Cast(transformed_img, sitk.sitkUInt8)
                    sitk.WriteImage(transformed_img[:, :, :, position], os.path.join(registration_output_path, saving_name))

                except:
                    with open('error_groupwise_reg.txt', 'a') as f:
                        f.write(patient_id + '\n')
                    print('Error in groupwise registration for patient: ', patient_id)
                    continue
        except:
            with open('error_unkown.txt', 'a') as f:
                f.write(patient_id + '\n')
            print('Error in groupwise registration for patient: ', patient_id)
            continue


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    orig_dyn_select_path = os.path.join(main_dir, 'derivatives/2_bias_field_correction')
    seg_path = os.path.join(main_dir, 'derivatives/reoriented_segmentations')
    registration_path_1 = os.path.join(main_dir, 'derivatives/5_1_T1_registration_pairwise')
    registration_path_2 = os.path.join(main_dir, 'derivatives/6_2_T1_registration_groupwise')
    output_path = os.path.join(main_dir, 'derivatives/7_tumor_segmentation_transformed')
    os.makedirs(output_path, exist_ok=True)
    main(seg_path, orig_dyn_select_path, registration_path_1, registration_path_2, output_path)
