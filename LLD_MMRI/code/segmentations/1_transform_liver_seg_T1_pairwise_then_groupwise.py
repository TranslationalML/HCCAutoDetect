import os
import pandas as pd
import SimpleITK as sitk

def main(seg_path, orig_dyn_select_path, groupwise_reg_path, output_derivative_path):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t', index_col=0)
    for patient_id, row in participants.iterrows():
        print(patient_id)
        try:
            patient_groupwise_reg_path = os.path.join(groupwise_reg_path, patient_id)
            seg_dyn_path = os.path.join(seg_path, patient_id, 'dyn')
            patient_files = os.path.join(orig_dyn_select_path, patient_id)
            groupwise_reg_dyn_path = os.path.join(patient_groupwise_reg_path, 'dyn/transforms')

            if os.path.exists(groupwise_reg_dyn_path):
                if len(os.listdir(groupwise_reg_dyn_path)) > 1:
                    groupwise_files = os.listdir(groupwise_reg_dyn_path)
                    groupwise_transform = [os.path.join(groupwise_reg_dyn_path, file)
                                        for file in groupwise_files if 'Transform' in file][0]
                else:
                    with open('liver_seg_missing_groupwise_reg.txt', 'a') as f:
                        f.write(patient_id + '\n')
                    print('Groupwise registration missing for patient: ', patient_id)
                    continue
            else:
                with open('liver_seg_missing_groupwise_reg.txt', 'a') as f:
                    f.write(patient_id + '\n')
                print('Groupwise registration missing for patient: ', patient_id)
                continue

            if not os.path.exists(seg_dyn_path):
                with open('liver_seg_missing_segmentation.txt', 'a') as f:
                    f.write(patient_id + '\n')
                print('Segmentation missing for ', patient_id)
                continue


            registration_output_path = os.path.join(output_derivative_path, patient_id, 'dyn')
            os.makedirs(registration_output_path, exist_ok=True)
            if len(os.listdir(registration_output_path)) > 1:
                continue
            
            seg_file_list = os.listdir(seg_dyn_path)
            seg_file_list = [file for file in seg_file_list if 'nii.gz' in file]
            patient_dyn_path = os.path.join(patient_files, 'dyn')

    
            for seg_file in seg_file_list:
                liver_lesion_seg_path = os.path.join(seg_dyn_path, seg_file)
                if 'venous' in seg_file:
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
                    with open('liver_seg_error_groupwise_reg.txt', 'a') as f:
                        f.write(patient_id + '\n')
                    print('Error in groupwise registration for patient: ', patient_id)
                    continue
        except:
            with open('liver_seg_error_unkown.txt', 'a') as f:
                f.write(patient_id + '\n')
            print('Error in groupwise registration for patient: ', patient_id)
            continue


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    orig_dyn_select_path = os.path.join(main_dir, 'derivatives/2_bias_field_correction')
    seg_path = os.path.join(main_dir, 'derivatives/4_liver_masks_corrected/liver_masks_margin_10')
    registration_path = os.path.join(main_dir, 'derivatives/6_2_T1_registration_groupwise')
    output_path = os.path.join(main_dir, 'derivatives/7_liver_segmentation_transformed')
    os.makedirs(output_path, exist_ok=True)
    main(seg_path, orig_dyn_select_path, registration_path, output_path)
