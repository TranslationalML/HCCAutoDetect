import pandas as pd
import json
import os
import numpy as np
from datetime import timedelta, datetime

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t")

    input_derivative = os.path.join(main_dir, 'derivatives/2_bias_field_correction')
    output_derivative_path = os.path.join(main_dir, 'derivatives/3_select_dyn_phases')
    os.makedirs(output_derivative_path, exist_ok=True)

    main_dir = main_dir
    time_per_phase = {}
    overtime_patients = {}
    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        print(patient_ID)
        patient_dyn_path = os.path.join(input_derivative, patient_ID, 'dyn')

        patient_output_path = os.path.join(output_derivative_path, patient_ID)
        os.makedirs(patient_output_path, exist_ok=True)
        patient_output_dyn_path = os.path.join(patient_output_path, 'dyn')
        os.makedirs(patient_output_dyn_path, exist_ok=True)
        dyn_files = os.listdir(patient_dyn_path)
        metadata_files = [txt for txt in dyn_files if txt[-4:] == 'json']

        is_dixon = [file for file in metadata_files if 'dixon' in file]
        if is_dixon:
            native = [file for file in metadata_files if 'native_dixon_w' in file and not "SUB" in file]
            venous = [file for file in metadata_files if 'venous_dixon_w' in file and not "SUB" in file]
            delayed = [file for file in metadata_files if 'delayed_dixon_w' in file and not "SUB" in file]
            arterial_TTC_1 = [file for file in metadata_files if '_dixon_w_TTC_1' in file and not "SUB" in file]
            arterial_TTC_2 = [file for file in metadata_files if '_dixon_w_TTC_2' in file and not "SUB" in file]
            arterial_TTC_3 = [file for file in metadata_files if '_dixon_w_TTC_3' in file and not "SUB" in file]
            sorted_phases = [native, arterial_TTC_1, arterial_TTC_2, arterial_TTC_3, venous, delayed]
            sorted_phases = [sublist for sublist in sorted_phases if sublist]

        else:
            native = [file for file in metadata_files if 'native' in file and not "SUB" in file]
            venous = [file for file in metadata_files if 'venous' in file and not "SUB" in file]
            delayed = [file for file in metadata_files if 'delayed' in file and not "SUB" in file]
            arterial = [file for file in metadata_files if 'arterial' in file and not "SUB" in file]
            sorted_phases = [native, arterial, venous, delayed]

        sorted_phases = [[elem for elem in sublist if 'REG' in elem]
                         if len([file for file in sublist if 'REG' in file]) != 0 else sublist
                         for sublist in sorted_phases]

        all_phases = {}
        phases_selection = {}
        for phases in sorted_phases:
            if len(phases) > 1:
                time = {}
                for file in phases:
                    with open(os.path.join(patient_dyn_path, file)) as f:
                        metadata = json.load(f)
                    time[file] = metadata['AcquisitionTime']
                    all_phases[file] = metadata['AcquisitionTime']
                max_time = max(time, key=time.get)

                format_string = "%H:%M:%S.%f"
                datetime_object = datetime.strptime(time[max_time], format_string)

                if 'native' in max_time:
                    phases_selection[max_time] = time[max_time]
                    previous_time = datetime_object
                elif datetime_object - previous_time < timedelta(minutes=10):
                    phases_selection[max_time] = time[max_time]
                    previous_time = datetime_object
                else:
                    min_time = min(time, key=time.get)
                    phases_selection[min_time] = time[min_time]
                    previous_time = datetime_object

            elif len(phases) == 1:
                with open(os.path.join(patient_dyn_path, phases[0])) as f:
                    metadata = json.load(f)
                phases_selection[phases[0]] = metadata['AcquisitionTime']
                format_string = "%H:%M:%S.%f"
                previous_time = datetime.strptime(metadata['AcquisitionTime'], format_string)

            else:
                print('Missing phase for patient ', patient_ID)
                continue

        df = pd.DataFrame(phases_selection, index=['time']).T
        df['time'] = sorted(pd.to_datetime(df['time']))
        delta_time = (df['time'].values[-1] - df['time'].values[0])/np.timedelta64(1, 'm')
        df['delta_time'] = (df['time'] - df['time'][0]).apply(lambda x: '{:02d}:{:02d}'.format(int(x.total_seconds() // 60), int(x.total_seconds() % 60)))
        del df['time']
        time_per_phase[patient_ID] = df
        if delta_time > 10:
            print('Problem in time of acquisition of the phases for patient ', patient_ID)
            overtime_patients[patient_ID] = delta_time

        img_to_keep = []
        for json_file in phases_selection.keys():
            if is_dixon:
                for dixon_type in ['_w_', '_in_', '_opp_']:
                    json_file_dixon = json_file.replace('_w_', dixon_type)
                    img_to_keep.append(os.path.join(patient_dyn_path, json_file_dixon))
            else:
                img_to_keep.append(os.path.join(patient_dyn_path, json_file))

        img_to_keep = [file.replace('.json', '.nii.gz') for file in img_to_keep if any(dyn_file in file for dyn_file in dyn_files)]
        print(patient_ID, ' ', len(dyn_files)-len(phases_selection),
              ' - images dropped (before/after: ', len(dyn_files), '/', len(phases_selection), ')')

        with open(patient_output_dyn_path + '/phases_to_keep.json', 'w') as file:
            file.write(json.dumps(img_to_keep, indent=4))
        
    
    with open(output_derivative_path + '/time_per_phase.txt', 'w') as file:
        for key, value in time_per_phase.items():
            file.write(f"{key}: {value}\n")

    with open(output_derivative_path + '/over_time_patients.txt', 'w') as file:
        for key, value in overtime_patients.items():
            file.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
