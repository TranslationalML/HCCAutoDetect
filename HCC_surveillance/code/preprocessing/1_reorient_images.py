import os
import pandas as pd
import SimpleITK as sitk
import shutil

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t")
    output_derivative_path = os.path.join(main_dir, 'derivatives/1_reoriented_images')
    output_derivative_path = output_derivative_path

    main_dir = main_dir
    os.makedirs(output_derivative_path, exist_ok=True)
    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        print(patient_ID)
        patient_path = os.path.join(main_dir, patient_ID)
        patient_output_path = os.path.join(output_derivative_path, patient_ID)
        os.makedirs(patient_output_path, exist_ok=True)

        dyn_output_path = os.path.join(patient_output_path, 'dyn')
        os.makedirs(dyn_output_path, exist_ok=True)

        for dir_, subdir, files in os.walk(patient_path):
            for file in files:
                img_category = os.path.basename(dir_)
                img_output_path = os.path.join(patient_output_path, img_category)
                if os.path.exists(os.path.join(img_output_path, file)):
                    continue
                if 'nii.gz' in file: 
                    os.makedirs(img_output_path, exist_ok=True)
                    img = sitk.ReadImage(os.path.join(dir_, file))
                    img_RAS = sitk.DICOMOrient(img, 'RAS')
                    sitk.WriteImage(img_RAS, os.path.join(img_output_path, file))

                if any([x in file for x in ['.bvec', '.bval', '.json']]):

                    os.makedirs(img_output_path, exist_ok=True)
                    if os.path.isfile(os.path.join(dir_, file)):
                        shutil.copyfile(os.path.join(dir_, file), os.path.join(img_output_path, file))
                        shutil.copyfile(os.path.join(dir_, file), os.path.join(img_output_path, file))


if __name__ == "__main__":
    main()
