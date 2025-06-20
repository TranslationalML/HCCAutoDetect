import SimpleITK as sitk
import os
import pandas as pd


def main(reoriented_patients_path, output_derivative_path):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t', index_col=0)
    for patient_id, row in participants.iterrows():
        print(patient_id)
        patient_derivative_path = os.path.join(output_derivative_path, patient_id)
        os.makedirs(patient_derivative_path, exist_ok=True)

        patient_path = os.path.join(reoriented_patients_path, patient_id)
        for dir_, subdirs, files in os.walk(patient_path):
            for file in files:
                dcm_path = os.path.join(dir_, file)
                img = sitk.ReadImage(dcm_path, sitk.sitkFloat64)
                img_category = os.path.basename(dir_)
                patient_category_path = os.path.join(patient_derivative_path, img_category)
                os.makedirs(patient_category_path, exist_ok=True)

                shrink_factor = 4
                saving_name = os.path.join(patient_category_path, file)

                shrinked_image = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                number_of_fitting_levels = 4
                corrected_img = corrector.Execute(shrinked_image)

                log_bias_field = sitk.Cast(corrector.GetLogBiasFieldAsImage(img), sitk.sitkFloat64)
                corrected_image_full_resolution = img / sitk.Exp(log_bias_field)
                sitk.WriteImage(corrected_image_full_resolution, saving_name)



if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    input_path = os.path.join(main_dir, 'derivatives/1_reoriented_images')
    output_path = os.path.join(main_dir, 'derivatives/2_bias_field_correction')

    main(input_path, output_path)
