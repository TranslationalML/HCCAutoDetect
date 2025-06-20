import json
import os
import pandas as pd
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

def main(reoriented_patients_path, output_derivative_path):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t', index_col=0)

    name_correspondance = {}
    for patient_id, row in participants.iterrows():
        print(patient_id)
        patient_path = os.path.join(reoriented_patients_path, patient_id)
        for dir_, subdirs, files in os.walk(patient_path):
            for file in files:
                if not any([x in file for x in ['_in.nii.gz', '_out.nii.gz', 'T2.nii.gz', 'DWI.nii.gz', 'transform', '.txt']]):
                    input_img_path = os.path.join(dir_, file)
                    if 'native' in input_img_path:
                        input_img = nib.load(input_img_path)
                        output_img = totalsegmentator(input_img, task='total_mr', fast=True)
                        output_path = input_img_path.replace('6_2_T1_registration_groupwise', '7_total_segmentator_predictions')
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        nib.save(output_img, output_path)
                        print()
    
    with open(os.path.join(output_derivative_path, '7_grouped_images_nnunet_preds_name_correspondance.json'), 'w') as f:
        json.dump(name_correspondance, f)

if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    input_path = os.path.join(main_dir, 'derivatives/6_2_T1_registration_groupwise')
    output_path = os.path.join(main_dir, 'derivatives/7_total_segmentator_predictions')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path + '/imagesTr', exist_ok=True)
    os.makedirs(output_path + '/labelsTr', exist_ok=True)
    main(input_path, output_path)
