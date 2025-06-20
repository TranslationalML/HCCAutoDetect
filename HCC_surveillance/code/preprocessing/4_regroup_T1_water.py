import os
import pandas as pd
import json


def main(output_derivative_path, data_dir):
    participants = pd.read_csv((main_dir + '/participants.tsv'), sep='\t')
    for patient, sub_id in participants[['Patient', 'Sub-ID']].values:

        patient_output_path = os.path.join(output_derivative_path, sub_id)
        os.makedirs(patient_output_path, exist_ok=True)
        patient_output_dyn_path = os.path.join(patient_output_path, 'dyn')
        os.makedirs(patient_output_dyn_path, exist_ok=True)

        patient_dyn_path = os.path.join(data_dir, sub_id + '/dyn')
        dyn_files = patient_dyn_path + '/phases_to_keep.json'
        if os.path.exists(dyn_files):
            with open(dyn_files, 'r') as file:
                files_to_keep = json.load(file)
            water_files = [file for file in files_to_keep if any([x in file for x in ["dixon_w_", "caipi"]])]
        else:
            continue

        with open(os.path.join(patient_output_dyn_path) + '/water_phases.json', 'w') as file:
            file.write(json.dumps(water_files, indent=4))


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    data_dir = os.path.join(main_dir, "derivatives/3_select_dyn_phases")
    output_derivative_path = os.path.join(main_dir, 'derivatives/4_T1_water_images')
    os.makedirs(output_derivative_path, exist_ok=True)
    main(output_derivative_path, data_dir)
