import os
import SimpleITK as sitk
import pandas as pd
import shutil
from utils import rigid_transform
import json


def main(input_patients_path, output_derivative_path, liver_mask_dataset, config_rigid, config_bspline):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')
    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:
        print(patient_id)
        patient_dyn_path = os.path.join(input_patients_path, patient_id + '/dyn')
        files_to_keep = patient_dyn_path + '/phases_to_keep.json'
        if not os.path.exists(files_to_keep):
            continue
        with open(files_to_keep, 'r') as file:
            dyn_list = json.load(file)

        patient_derivative_path = os.path.join(output_derivative_path, patient_id)
        os.makedirs(patient_derivative_path, exist_ok=True)
        patient_path = os.path.join(input_patients_path, patient_id)
        patient_cat_path = os.path.join(patient_path, 'dyn')
        patient_output_path = os.path.join(output_derivative_path, patient_id)

        registration_output_path = os.path.join(patient_output_path, 'dyn')
        os.makedirs(registration_output_path, exist_ok=True)
        os.makedirs(registration_output_path + '/transforms', exist_ok=True)

        if not dyn_list or not all(any(x in y for y in dyn_list) for x in ['native', 'arterial', 'venous', 'delayed']):
            continue

        dyn_list = sorted(dyn_list)
        population = dyn_list

        T1_dixon_w_liver_seg = [file.replace('2_bias_field_correction', liver_mask_dataset) for file in dyn_list]
        if any(not os.path.exists(file) for file in T1_dixon_w_liver_seg):
            print('Liver mask missing for ', patient_id)
            continue

        # nat->art-TTC1
        nat_img = [file for file in population if 'native' in file][0]
        art_TTC1_img = [file for file in population if 'arterial' in file][0]
        nat_mask = nat_img.replace('2_bias_field_correction', liver_mask_dataset)
        art_TTC1_mask = art_TTC1_img.replace('2_bias_field_correction', liver_mask_dataset)
        rigid_transform(config_rigid, config_bspline, art_TTC1_img, art_TTC1_mask, nat_img, nat_mask,
                        registration_output_path, '_nat_to_art')

        mov_img = [file for file in population if 'arterial' in file][0]
        fix_img = [file for file in population if 'venous' in file][0]
        mov_mask = mov_img.replace('2_bias_field_correction', liver_mask_dataset)
        fix_mask = fix_img.replace('2_bias_field_correction', liver_mask_dataset)
        rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask,
                        registration_output_path, '_art_to_venous')

        mov_img = [file for file in population if 'delayed' in file][0]
        fix_img = [file for file in population if 'venous' in file][0]
        mov_mask = mov_img.replace('2_bias_field_correction', liver_mask_dataset)
        fix_mask = fix_img.replace('2_bias_field_correction', liver_mask_dataset)
        rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask, registration_output_path, '_delayed_to_venous')

        transforms = os.listdir(registration_output_path + '/transforms')
        transforms = [file for file in transforms if 'TransformParameters' in file]


        # transform
        nat_art = sorted([os.path.join(registration_output_path + '/transforms', file) for file in transforms if
                            '_nat_to_art' in file])
        art_venous = sorted(
            [os.path.join(registration_output_path + '/transforms', file) for file in transforms if
                '_art_to_venous' in file])
        delayed_venous = sorted(
            [os.path.join(registration_output_path + '/transforms', file) for file in transforms if
                '_delayed_to_venous' in file])

        transform_11 = sitk.ReadParameterFile(nat_art[0])
        transform_12 = sitk.ReadParameterFile(nat_art[1])
        transform_21 = sitk.ReadParameterFile(art_venous[0])
        transform_22 = sitk.ReadParameterFile(art_venous[1])
        transform_31 = sitk.ReadParameterFile(delayed_venous[0])
        transform_32 = sitk.ReadParameterFile(delayed_venous[1])

        # align native to ven
        nat_img = [file for file in population if 'native' in file][0]
        transformix_image_filter = sitk.TransformixImageFilter()
        transformix_image_filter.SetMovingImage(sitk.ReadImage(nat_img))
        transformix_image_filter.SetTransformParameterMap(transform_11)
        transformix_image_filter.AddTransformParameterMap(transform_12)
        transformix_image_filter.AddTransformParameterMap(transform_21)
        transformix_image_filter.AddTransformParameterMap(transform_22)
        transformix_image_filter.Execute()
        res_img = transformix_image_filter.GetResultImage()
        sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(nat_img)))

        # align art to ven
        art_img = [file for file in population if 'arterial' in file][0]
        transformix_image_filter = sitk.TransformixImageFilter()
        transformix_image_filter.SetMovingImage(sitk.ReadImage(art_img))
        transformix_image_filter.AddTransformParameterMap(transform_21)
        transformix_image_filter.AddTransformParameterMap(transform_22)
        transformix_image_filter.Execute()
        res_img = transformix_image_filter.GetResultImage()
        sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(art_img)))

        # align delayed to ven
        delayed_img = [file for file in population if 'delayed' in file][0]
        transformix_image_filter = sitk.TransformixImageFilter()
        transformix_image_filter.SetMovingImage(sitk.ReadImage(delayed_img))
        transformix_image_filter.AddTransformParameterMap(transform_31)
        transformix_image_filter.AddTransformParameterMap(transform_32)
        transformix_image_filter.Execute()
        res_img = transformix_image_filter.GetResultImage()
        sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(delayed_img)))

        venous = [file for file in population if 'venous' in file][0]
        shutil.copyfile(venous, os.path.join(registration_output_path, os.path.basename(venous)))

        if os.path.isdir(registration_output_path + '/temp'):
            shutil.rmtree(registration_output_path + '/temp')
 
if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))

    input_path = os.path.join(main_dir, 'derivatives/3_select_dyn_phases')
    output_path = os.path.join(main_dir, 'derivatives/4_1_T1_registration_pairwise')
    config_rigid = os.path.join(main_dir, 'code/preprocessing/registration_config/REG__elastixConfig_rigid.txt')
    config_bspline = os.path.join(main_dir, 'code/preprocessing/registration_config/REG__elastixConfig_bspline.txt')
    liver_mask_dataset = '3_select_dyn_phases/predictions'

    input_path = input_path
    output_path = output_path
    main(input_path, output_path, liver_mask_dataset, config_rigid, config_bspline)
