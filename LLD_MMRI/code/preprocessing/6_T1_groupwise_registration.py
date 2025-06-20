import os
import SimpleITK as sitk
import pandas as pd
import shutil

def main(input_patients_path, output_derivative_path, config):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t', index_col=0)
    for patient_id, row in participants.iterrows():
        patient_derivative_path = os.path.join(output_derivative_path, patient_id)
        os.makedirs(patient_derivative_path, exist_ok=True)
        patient_cat_path = os.path.join(input_patients_path, patient_id, 'dyn')

        registration_output_path = os.path.join(output_derivative_path, patient_id, 'dyn')
        os.makedirs(registration_output_path, exist_ok=True)

        population = sorted([os.path.join(patient_cat_path, file) for file in os.listdir(patient_cat_path) if not any(x in file for x in ['_in', '_out', 'transforms'])])
        if len(population) == 0:
            continue

        if len(os.listdir(registration_output_path)) != 0:
            continue

        vectorOfImages = sitk.VectorOfImage()
        for filename in population:
            image = sitk.ReadImage(filename)
            image = sitk.Cast(image, sitk.sitkFloat64)
            vectorOfImages.push_back(image)

        image = sitk.JoinSeries(vectorOfImages)
        parameter_map = sitk.ReadParameterFile(config)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(image)
        elastixImageFilter.SetMovingImage(image)
        elastixImageFilter.SetParameterMap(parameter_map)
        elastixImageFilter.Execute()

        res_img = elastixImageFilter.GetResultImage()

        for idx, img_path in enumerate(population):
            saving_name = os.path.basename(img_path)
            sitk.WriteImage(res_img[:, :, :, idx], os.path.join(registration_output_path, saving_name))

        os.makedirs(os.path.join(registration_output_path, 'transforms'), exist_ok=True)
        for txt_file in ['IterationInfo.0.R0.txt', 'IterationInfo.0.R1.txt', 'IterationInfo.0.R2.txt',
                        'IterationInfo.0.R3.txt', 'TransformParameters.0.txt']:
            shutil.copyfile(os.path.join(cwd, txt_file), os.path.join(registration_output_path, 'transforms/' + txt_file))
            os.remove(os.path.join(cwd, txt_file))

if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))

    input_path = os.path.join(main_dir, 'derivatives/5_1_T1_registration_pairwise')
    output_path = os.path.join(main_dir, 'derivatives/6_2_T1_registration_groupwise')
    os.makedirs(output_path, exist_ok=True)
    config = os.path.join(main_dir, 'code/preprocessing/registration_config/REG__elastixConfig_groupwise.txt')
    main(input_path, output_path, config)
