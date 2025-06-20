import os
import pandas as pd
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

def main(reg_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')
    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:
        reg_path_patient = os.path.join(reg_path, patient_id, 'dyn')
        output_path_patient = os.path.join(output_path, patient_id, 'dyn')
        if os.path.exists(output_path_patient):
            continue
        os.makedirs(output_path_patient, exist_ok=True)

        img_files = [file for file in os.listdir(reg_path_patient) if 'nii.gz' in file and 'venous' in file]
        if len(img_files) == 0:
            print(f'No images for patient {patient_id}')
            continue
        for file in img_files:
            img = nib.load(os.path.join(reg_path_patient, file))
            tot_seg = totalsegmentator(img)
            nib.save(tot_seg, os.path.join(output_path_patient, file))

if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    registration_path = os.path.join(main_dir, 'derivatives/9_2_T1_registration_groupwise')
    output_path = os.path.join(main_dir, 'derivatives/9_total_segmentator')
    os.makedirs(output_path, exist_ok=True)
    main(registration_path, output_path)
