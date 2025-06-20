import os
import SimpleITK as sitk
import pandas as pd
import shutil
from utils import resampleTTC, updatePopulation, rigid_transform
import json

def main(input_patients_path, output_derivative_path, liver_mask_dataset, config_rigid, config_bspline):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')
    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:

        patient_dyn_path = os.path.join(input_patients_path, patient_id + '/dyn')
        files_to_keep = patient_dyn_path + '/phases_to_keep.json'
        if not os.path.exists(files_to_keep):
            continue 
        with open(files_to_keep, 'r') as file:
            dyn_list = json.load(file)
        dyn_list = [file for file in dyn_list]

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

        T1_dixon_w = [file for file in dyn_list if '_dixon_w_' in file]
        T1_dixon_w = sorted(T1_dixon_w)
        T1_caipi = [file for file in dyn_list if '_caipi_' in file]

        if len(T1_dixon_w) != 0:
            try:
                T1_TTC = [file for file in T1_dixon_w if 'TTC' in file]
                venous = [file for file in T1_dixon_w if 'venous' in file][0]
            except:
                print('No TTC or venous phase for ', patient_id)
                continue
            venous_img = sitk.ReadImage(venous)
            T1_list_arterial_w_TTC_path = resampleTTC(T1_TTC, venous_img, registration_output_path, '.nii.gz')
            population = updatePopulation(T1_dixon_w, T1_list_arterial_w_TTC_path)

            T1_TTC_mask = [file.replace('3_bias_field_correction', liver_mask_dataset) for file in T1_TTC]
            venous_mask = venous.replace('3_bias_field_correction', liver_mask_dataset)
            venous_mask_sitk = sitk.ReadImage(venous_mask)
            T1_list_arterial_w_TTC_mask_path = resampleTTC(T1_TTC_mask, venous_mask_sitk, registration_output_path, '_mask.nii.gz')

        else:
            population = T1_caipi

        T1_liver_seg = [file.replace('3_bias_field_correction', liver_mask_dataset) for file in population]
        if not any(os.path.exists(file) for file in T1_liver_seg):
            print('Liver mask missing for ', patient_id)
            continue

        # nat->art-TTC1
        if len(T1_dixon_w) == 6:
            nat_img = [file for file in population if 'native' in file][0]
            art_TTC1_img = [file for file in population if 'TTC_1' in file][0]

            nat_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
            art_TTC1_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)
            rigid_transform(config_rigid, config_bspline, art_TTC1_img, art_TTC1_mask, nat_img, nat_mask, registration_output_path, '_nat_to_art_TTC_1')

            mov_img = [file for file in population if 'TTC_1' in file][0]
            fix_img = [file for file in population if 'TTC_2' in file][0]
            mov_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
            fix_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)
            rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask, registration_output_path, '_TTC_1_to_art_TTC_2')

            mov_img = [file for file in population if 'TTC_2' in file][0]
            fix_img = [file for file in population if 'TTC_3' in file][0]
            mov_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
            fix_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)
            rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask, registration_output_path, '_TTC_2_to_art_TTC_3')

            mov_img = [file for file in population if 'TTC_3' in file][0]
            fix_img = [file for file in population if 'venous' in file][0]
            mov_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
            fix_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)
            rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask, registration_output_path, '_TTC_3_to_venous')

        else:
            nat_img = [file for file in population if 'native' in file][0]
            art_TTC1_img = [file for file in population if 'arterial' in file][0]
            if sitk.ReadImage(nat_img).GetSize() != sitk.ReadImage(art_TTC1_img).GetSize():
                art_TTC1_img = resampleTTC([art_TTC1_img], sitk.ReadImage(nat_img), registration_output_path, '.nii.gz')
                population = [file if not 'art' in file else art_TTC1_img[0] for file in population]
            nat_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
            art_TTC1_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)
            rigid_transform(config_rigid, config_bspline, art_TTC1_img, art_TTC1_mask, nat_img, nat_mask,
                            registration_output_path, '_nat_to_art')

            mov_img = [file for file in population if 'arterial' in file][0]
            fix_img = [file for file in population if 'venous' in file][0]
            mov_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
            fix_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)

            rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask,
                            registration_output_path, '_art_to_venous')

        mov_img = [file for file in population if 'delayed' in file][0]
        fix_img = [file for file in population if 'venous' in file][0]
        mov_mask = nat_img.replace('3_bias_field_correction', liver_mask_dataset)
        fix_mask = art_TTC1_img.replace('3_bias_field_correction', liver_mask_dataset)

        rigid_transform(config_rigid, config_bspline, fix_img, fix_mask, mov_img, mov_mask, registration_output_path, '_delayed_to_venous')

        transforms = os.listdir(registration_output_path + '/transforms')
        transforms = [file for file in transforms if 'TransformParameters' in file]


        # transform
        if len(T1_dixon_w) == 6:
            nat_TTC_1 = sorted([os.path.join(registration_output_path + '/transforms', file) for file in transforms if '_nat_to_art_TTC_1' in file])
            TTC_1_TTC_2 = sorted([os.path.join(registration_output_path + '/transforms', file) for file in transforms if '_TTC_1_to_art_TTC_2' in file])
            TTC_2_TTC_3 = sorted([os.path.join(registration_output_path + '/transforms', file) for file in transforms if '_TTC_2_to_art_TTC_3' in file])
            TTC_3_venous = sorted([os.path.join(registration_output_path + '/transforms', file) for file in transforms if '_TTC_3_to_venous' in file])
            delayed_venous = sorted([os.path.join(registration_output_path + '/transforms', file) for file in transforms if '_delayed_to_venous' in file])

            transform_11 = sitk.ReadParameterFile(nat_TTC_1[0])
            transform_12 = sitk.ReadParameterFile(nat_TTC_1[1])
            transform_21 = sitk.ReadParameterFile(TTC_1_TTC_2[0])
            transform_22 = sitk.ReadParameterFile(TTC_1_TTC_2[1])
            transform_31 = sitk.ReadParameterFile(TTC_2_TTC_3[0])
            transform_32 = sitk.ReadParameterFile(TTC_2_TTC_3[1])
            transform_41 = sitk.ReadParameterFile(TTC_3_venous[0])
            transform_42 = sitk.ReadParameterFile(TTC_3_venous[1])
            transform_51 = sitk.ReadParameterFile(delayed_venous[0])
            transform_52 = sitk.ReadParameterFile(delayed_venous[1])

            # align native to ven
            nat_img = [file for file in population if 'native' in file][0]
            transformix_image_filter = sitk.TransformixImageFilter()
            transformix_image_filter.SetMovingImage(sitk.ReadImage(nat_img))
            transformix_image_filter.SetTransformParameterMap(transform_11)
            transformix_image_filter.AddTransformParameterMap(transform_12)
            transformix_image_filter.AddTransformParameterMap(transform_21)
            transformix_image_filter.AddTransformParameterMap(transform_22)
            transformix_image_filter.SetTransformParameterMap(transform_31)
            transformix_image_filter.AddTransformParameterMap(transform_32)
            transformix_image_filter.AddTransformParameterMap(transform_41)
            transformix_image_filter.AddTransformParameterMap(transform_42)
            transformix_image_filter.Execute()
            res_img = transformix_image_filter.GetResultImage()
            sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(nat_img)))


            #align TTC1 to ven
            TTC_1_img = [file for file in population if 'TTC_1' in file][0]
            transformix_image_filter = sitk.TransformixImageFilter()
            transformix_image_filter.SetMovingImage(sitk.ReadImage(TTC_1_img))
            transformix_image_filter.AddTransformParameterMap(transform_21)
            transformix_image_filter.AddTransformParameterMap(transform_22)
            transformix_image_filter.AddTransformParameterMap(transform_31)
            transformix_image_filter.AddTransformParameterMap(transform_32)
            transformix_image_filter.AddTransformParameterMap(transform_41)
            transformix_image_filter.AddTransformParameterMap(transform_42)
            transformix_image_filter.Execute()
            res_img = transformix_image_filter.GetResultImage()
            sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(TTC_1_img)))

            # align TTC2 to ven
            TTC_2_img = [file for file in population if 'TTC_2' in file][0]
            transformix_image_filter = sitk.TransformixImageFilter()
            transformix_image_filter.SetMovingImage(sitk.ReadImage(TTC_2_img))
            transformix_image_filter.AddTransformParameterMap(transform_31)
            transformix_image_filter.AddTransformParameterMap(transform_32)
            transformix_image_filter.AddTransformParameterMap(transform_41)
            transformix_image_filter.AddTransformParameterMap(transform_42)
            transformix_image_filter.Execute()
            res_img = transformix_image_filter.GetResultImage()
            sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(TTC_2_img)))

            # align TTC3 to ven
            TTC_3_img = [file for file in population if 'TTC_3' in file][0]
            transformix_image_filter = sitk.TransformixImageFilter()
            transformix_image_filter.SetMovingImage(sitk.ReadImage(TTC_3_img))
            transformix_image_filter.AddTransformParameterMap(transform_41)
            transformix_image_filter.AddTransformParameterMap(transform_42)
            transformix_image_filter.Execute()
            res_img = transformix_image_filter.GetResultImage()
            sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(TTC_3_img)))

            # align delayed to ven
            del_img = [file for file in population if 'delayed' in file][0]
            transformix_image_filter = sitk.TransformixImageFilter()
            transformix_image_filter.SetMovingImage(sitk.ReadImage(del_img))
            transformix_image_filter.AddTransformParameterMap(transform_51)
            transformix_image_filter.AddTransformParameterMap(transform_52)
            transformix_image_filter.Execute()
            res_img = transformix_image_filter.GetResultImage()
            sitk.WriteImage(res_img, os.path.join(registration_output_path, os.path.basename(del_img)))

        else:
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
    output_path = os.path.join(main_dir, 'derivatives/5_T1_registration_pairwise')
    config_rigid = os.path.join(main_dir, 'code/preprocessing/registration_config/REG__elastixConfig_rigid.txt')
    config_bspline = os.path.join(main_dir, 'code/preprocessing/registration_config/REG__elastixConfig_bspline.txt')
    liver_mask_datasets = '7_T1_water_images' 
    input_path = input_path
    output_path = output_path
    main(input_path, output_path, liver_mask_datasets, config_rigid, config_bspline)
