import os
import dcm2bids
import pandas as pd
import json


def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    config_map_path = os.path.join(main_dir, 'code/pipeline/dcm2bids_config.json')
    
    participants = pd.read_csv(main_dir + '/participants_with_session_depersonalised.tsv', sep="\t", index_col=0, dtype=str)
    main_dir = main_dir

    data_path = os.path.join(main_dir, 'sourcedata/all_patients_all_dates_depersonalised')


    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        session = participants['session'][participants['Sub-ID'] == patient_ID].values[0]
        if '.' in session:
            session = session.split('.')[0]
        dicom_path = os.path.join(data_path, patient, session)
    
        dirs = [dir_ for dir_, subdirs, files in os.walk(os.path.join(dicom_path))
                if os.path.basename(dir_) != dicom_path]

        convert_patient = dcm2bids.Dcm2bids(dirs, participant=patient_ID, config=config_map_path, output_dir=main_dir)
        convert_patient.run()



if __name__ == "__main__":
    main()
