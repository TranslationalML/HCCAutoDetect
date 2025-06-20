import os
import pandas as pd
import shutil
import re 

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t", index_col=0)
    data_path = os.path.join(main_dir, 'sourcedata/LLD-MMRI-TumorSeg-MedSAM')
    data_path = data_path
    files = os.listdir(data_path)

    main_dir = main_dir
    output_path = os.path.join(main_dir, 'derivatives', 'segmentations')
    os.makedirs(output_path, exist_ok=True)
    for patient_ID, row in participants.iterrows():
        patient_output = os.path.join(output_path, patient_ID)
        os.makedirs(patient_output, exist_ok=True)
        patient_files = [os.path.join(data_path, file) for file in files if re.search(patient_ID, file)]

        output_T1 = os.path.join(patient_output, 'dyn')
        os.makedirs(output_T1, exist_ok=True)
        output_T2 = os.path.join(patient_output, 'anat')
        os.makedirs(output_T2, exist_ok=True)
        output_DWI = os.path.join(patient_output, 'dwi')
        os.makedirs(output_DWI, exist_ok=True)

        T2 = [file for file in patient_files if 'T2WI' in file][0]
        T1_nat = [file for file in patient_files if 'C-pre' in file][0]
        T1_art = [file for file in patient_files if 'C+A' in file][0]
        T1_ven = [file for file in patient_files if 'C+V' in file][0]
        T1_del = [file for file in patient_files if 'C+Delay' in file][0]
        T1_in = [file for file in patient_files if 'InPhase' in file][0]
        T1_out = [file for file in patient_files if 'OutPhase' in file][0]
        DWI = [file for file in patient_files if 'DWI' in file][0]


        shutil.copy(T2, os.path.join(output_T2, patient_ID + '-T2.nii.gz'))
        shutil.copy(T1_nat, os.path.join(output_T1, patient_ID + '-T1_phase_1_native.nii.gz'))
        shutil.copy(T1_art, os.path.join(output_T1, patient_ID + '-T1_phase_2_arterial.nii.gz'))
        shutil.copy(T1_ven, os.path.join(output_T1, patient_ID + '-T1_phase_3_venous.nii.gz'))
        shutil.copy(T1_del, os.path.join(output_T1, patient_ID + '-T1_phase_4_delayed.nii.gz'))
        shutil.copy(T1_in, os.path.join(output_T1, patient_ID + '-T1_in.nii.gz'))
        shutil.copy(T1_out, os.path.join(output_T1, patient_ID + '-T1_out.nii.gz'))
        shutil.copy(DWI, os.path.join(output_DWI, patient_ID + '-DWI.nii.gz'))

         
if __name__ == "__main__":
    main()
