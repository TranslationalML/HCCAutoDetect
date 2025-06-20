import os
import pandas as pd
import SimpleITK as sitk
import shutil

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants_with_session_depersonalised.tsv', sep="\t", index_col=0, dtype=str)

    dyn_selection_path = main_dir
    output_derivative_path = os.path.join(main_dir, 'derivatives/1_reoriented_images')
    output_derivative_path = output_derivative_path
    os.makedirs(output_derivative_path, exist_ok=True)

    for patient_ID in participants['Sub-ID'].values:
        
        print(patient_ID)
        patient_path = os.path.join(dyn_selection_path, patient_ID)
        try:
            files = os.listdir(patient_path)
        except:
            print('No session found for patient: ', patient_ID)
            continue

        patient_dyn_path = os.path.join(patient_path, 'dyn')
        patient_dwi_path = os.path.join(patient_path, 'dwi')
        patient_anat_path = os.path.join(patient_path, 'anat')


        if os.path.exists(patient_dyn_path):
            patient_output_path = os.path.join(output_derivative_path, patient_ID)
            os.makedirs(patient_output_path, exist_ok=True)
            dyn_output_path = os.path.join(patient_output_path, 'dyn')
            os.makedirs(dyn_output_path, exist_ok=True)
            for dyn_file in os.listdir(patient_dyn_path):
                if dyn_file.endswith('.nii.gz'):
                    img = sitk.ReadImage(os.path.join(patient_dyn_path, dyn_file))

                    img_RAS = sitk.DICOMOrient(img, 'RAS')
                    sitk.WriteImage(img_RAS, os.path.join(dyn_output_path, dyn_file))
                        
                else:
                    shutil.copy(os.path.join(patient_dyn_path, dyn_file), dyn_output_path)


if __name__ == "__main__":
    main()
