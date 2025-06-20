import os
import pandas as pd
import SimpleITK as sitk

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t", index_col=0)

    dyn_selection_path = main_dir
    output_derivative_path = os.path.join(main_dir, 'derivatives/1_reoriented_images')
    output_derivative_path = output_derivative_path
    os.makedirs(output_derivative_path, exist_ok=True)
    for patient_ID, row in participants.iterrows():
        print(patient_ID)
        patient_path = os.path.join(dyn_selection_path, patient_ID)
        patient_dyn_path = os.path.join(patient_path, 'dyn')
        patient_dwi_path = os.path.join(patient_path, 'dwi')
        patient_anat_path = os.path.join(patient_path, 'anat')
        patient_output_path = os.path.join(output_derivative_path, patient_ID)
        os.makedirs(patient_output_path, exist_ok=True)

        dyn_output_path = os.path.join(patient_output_path, 'dyn')
        os.makedirs(dyn_output_path, exist_ok=True)
        for dyn_file in os.listdir(patient_dyn_path):
            img = sitk.ReadImage(os.path.join(patient_dyn_path, dyn_file))
            img_RAS = sitk.DICOMOrient(img, 'RAS')
            sitk.WriteImage(img_RAS, os.path.join(dyn_output_path, dyn_file))

        anat_output_path = os.path.join(patient_output_path, 'anat')
        os.makedirs(anat_output_path, exist_ok=True)
        for anat_file in os.listdir(patient_anat_path):
            img = sitk.ReadImage(os.path.join(patient_anat_path, anat_file))
            img_RAS = sitk.DICOMOrient(img, 'RAS')
            sitk.WriteImage(img_RAS, os.path.join(anat_output_path, anat_file))

        dwi_output_path = os.path.join(patient_output_path, 'dwi')
        os.makedirs(dwi_output_path, exist_ok=True)
        for dwi_file in os.listdir(patient_dwi_path):
            img = sitk.ReadImage(os.path.join(patient_dwi_path, dwi_file))
            img_RAS = sitk.DICOMOrient(img, 'RAS')
            sitk.WriteImage(img_RAS, os.path.join(dwi_output_path, dwi_file))


if __name__ == "__main__":
    main()
