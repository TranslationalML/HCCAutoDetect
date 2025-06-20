import os
import pandas as pd
import shutil

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t", index_col=0)
    data_path = os.path.join(main_dir, 'sourcedata/images')
    data_path = data_path

    output_path = main_dir
    for patient_ID, row in participants.iterrows():
        patient_output = os.path.join(output_path, patient_ID)
        os.makedirs(patient_output, exist_ok=True)
        dicom_path = os.path.join(data_path, patient_ID, row['studyUID'])
        T2 = os.path.join(dicom_path, row['T2WI'] + '.nii.gz')
        T1_nat = os.path.join(dicom_path, row['C-pre'] + '.nii.gz')
        T1_art = os.path.join(dicom_path, row['C+A'] + '.nii.gz')
        T1_ven = os.path.join(dicom_path, row['C+V'] + '.nii.gz')
        T1_del = os.path.join(dicom_path, row['C+Delay'] + '.nii.gz')
        T1_in = os.path.join(dicom_path, row['In Phase'] + '.nii.gz')
        T1_out = os.path.join(dicom_path, row['Out Phase'] + '.nii.gz')
        DWI = os.path.join(dicom_path, row['DWI'] + '.nii.gz')


        output_T1 = os.path.join(patient_output, 'dyn')
        os.makedirs(output_T1, exist_ok=True)
        output_T2 = os.path.join(patient_output, 'anat')
        os.makedirs(output_T2, exist_ok=True)
        output_DWI = os.path.join(patient_output, 'dwi')
        os.makedirs(output_DWI, exist_ok=True)

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