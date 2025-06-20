import json
import os
import pandas as pd
import shutil

def main(reoriented_patients_path, output_derivative_path):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t', index_col=0)

    caseid = 0
    name_correspondance = {}
    for patient_id, row in participants.iterrows():
        print(patient_id)
        patient_path = os.path.join(reoriented_patients_path, patient_id)
        for dir_, subdirs, files in os.walk(patient_path):
            for file in files:
                if not any([x in file for x in ['_in.nii.gz', '_out.nii.gz', 'T2.nii.gz', 'DWI.nii.gz']]):
                    name = f'case_{caseid}_0000.nii.gz'
                    shutil.copy(os.path.join(dir_, file), os.path.join(output_derivative_path, 'imagesTr', name))
                    caseid += 1
                    name_correspondance[name] = os.path.join(dir_, file)
    
    with open(os.path.join(output_derivative_path, 'name_correspondance.json'), 'w') as f:
        json.dump(name_correspondance, f)



if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    input_path = os.path.join(main_dir, 'derivatives/2_bias_field_correction')
    output_path = os.path.join(main_dir, 'derivatives/3_all_images_grouped_nnunet_struct')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + '/imagesTr', exist_ok=True)
    os.makedirs(output_path + '/labelsTr', exist_ok=True)
    main(input_path, output_path)
