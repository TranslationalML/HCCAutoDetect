import json
import SimpleITK as sitk
import os
import pandas as pd
from utils import slice_by_slice_decorator
import shutil


def main(reoriented_patients_path, output_derivative_path):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')

    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:
        print(patient_id)

        patient_derivative_path = os.path.join(output_derivative_path, patient_id)
        os.makedirs(patient_derivative_path, exist_ok=True)

        patient_path = os.path.join(reoriented_patients_path, patient_id)
        for dir_, subdirs, files in os.walk(patient_path):
            for file in files:
                if file[-6:] == 'nii.gz' and 'T1' in file:
                    if os.path.exists(os.path.join(patient_derivative_path, file)):
                        continue
                    dcm_path = os.path.join(dir_, file)
                    img = sitk.ReadImage(dcm_path, sitk.sitkFloat64)
                    img_category = os.path.basename(dir_)
                    patient_category_path = os.path.join(patient_derivative_path, img_category)
                    os.makedirs(patient_category_path, exist_ok=True)

                    if 'TRACE' in file:
                        json_file = file[:-9] + '.json'
                    else:
                        json_file = file[:-7] + '.json'

                    with open(os.path.join(dir_, json_file)) as f:
                        meta_data = json.load(f)
                    acquistion_type = meta_data['MRAcquisitionType']

                    shrink_factor = 4
                    saving_name = os.path.join(patient_category_path, file)
                    if os.path.exists(saving_name):
                        continue

                    if acquistion_type == '3D':
                        shrinked_image = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
                        corrector = sitk.N4BiasFieldCorrectionImageFilter()
                        number_of_fitting_levels = 4
                        corrected_img = corrector.Execute(shrinked_image)

                        log_bias_field = sitk.Cast(corrector.GetLogBiasFieldAsImage(img), sitk.sitkFloat64)
                        corrected_image_full_resolution = img / sitk.Exp(log_bias_field)
                        sitk.WriteImage(corrected_image_full_resolution, saving_name)
                        if 'dwi' in file:
                            json_file = file.replace('.nii.gz', '.json')
                            json_saving_name = saving_name.replace('.nii.gz', '.json')
                            for idx in range(10):
                                json_file = json_file.replace('dwi_{}'.format(idx), 'dwi')
                                json_saving_name = json_saving_name.replace('dwi_{}'.format(idx), 'dwi')
                            shutil.copyfile(os.path.join(dir_, json_file), json_saving_name)
                        else:
                            json_file = file.replace('.nii.gz', '.json')
                            json_saving_name = saving_name.replace('.nii.gz', '.json')
                            shutil.copyfile(os.path.join(dir_, json_file), json_saving_name)

                    elif acquistion_type == '2D':
                        N4_2d = slice_by_slice_decorator(sitk.N4BiasFieldCorrectionImageFilter())
                        N4_2d(img, shrink_factor)
                        sitk.WriteImage(img, saving_name)
                        if 'dwi' in file:
                            json_file = file.replace('.nii.gz', '.json')
                            json_saving_name = saving_name.replace('.nii.gz', '.json')
                            for idx in range(10):
                                json_file = json_file.replace('dwi_{}'.format(idx), 'dwi')
                                json_saving_name = json_saving_name.replace('dwi_{}'.format(idx), 'dwi')
                            shutil.copyfile(os.path.join(dir_, json_file), json_saving_name)
                        else:
                            json_file = file.replace('.nii.gz', '.json')
                            json_saving_name = saving_name.replace('.nii.gz', '.json')
                            shutil.copyfile(os.path.join(dir_, json_file), json_saving_name)


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    input_path = os.path.join(main_dir, 'derivatives/1_reoriented_images')
    output_path = os.path.join(main_dir, 'derivatives/2_bias_field_correction')

    main(input_path, output_path)
