import os
import dcm2bids
import pandas as pd
import json


def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t")
    data_path = os.path.join(main_dir, 'sourcedata/all_patients_depersonalised')
    data_path = data_path
    config_map_path = os.path.join(main_dir, 'code/pipeline/dcm2bids_config.json')
    main_dir = main_dir
    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        dicom_path = os.path.join(data_path, patient)
        dicom_sub_dirs = os.listdir(dicom_path)
        if len(dicom_sub_dirs) == 1:
            dirs = [dir_ for dir_, subdirs, files in os.walk(os.path.join(dicom_path, dicom_sub_dirs[0]))
                    if os.path.basename(dir_) != dicom_sub_dirs[0]]

            convert_patient = dcm2bids.Dcm2bids(dirs, patient_ID, config_map_path, output_dir=main_dir)
            convert_patient.run()

        else:
            dirs_to_keep = []
            for sub_dir in dicom_sub_dirs:
                sub_sub_dirs = os.listdir(os.path.join(dicom_path, sub_dir))
                for path in sub_sub_dirs:
                    if any([x in path for x in ['T1', 'T2', 't1', 't2', 'diff']]):
                        dirs_to_keep.append(sub_dir)
                        break
            for dicom_sub_dir in dirs_to_keep:
                dirs = [dir_ for dir_, subdirs, files in os.walk(os.path.join(dicom_path, dicom_sub_dir))
                        if os.path.basename(dir_) != dicom_sub_dir]

                convert_patient = dcm2bids.Dcm2bids(dirs, patient_ID, config_map_path, output_dir=main_dir)
                convert_patient.run()


if __name__ == "__main__":
    main()
