import pandas as pd
import json
import os
import shutil
import numpy as np
from datetime import timedelta, datetime

def main():
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv(main_dir + '/participants.tsv', sep="\t")

    input_derivative = os.path.join(main_dir, 'derivatives/2_bias_field_correction')
    output_derivative_path = os.path.join(main_dir, 'derivatives/3_select_dyn_phases')
    os.makedirs(output_derivative_path, exist_ok=True)
    segmentation_path = os.path.join(main_dir, 'derivatives/tumor_segmentations')

    time_per_phase = {}
    overtime_patients = {}
    for patient_ID in participants['Sub-ID'].values:
        print(patient_ID)
        patient_path = os.path.join(input_derivative, patient_ID)
        patient_dyn_path = os.path.join(patient_path, 'dyn')
        patient_seg_path = os.path.join(segmentation_path, patient_ID, 'dyn')

        patient_output_path = os.path.join(output_derivative_path, patient_ID)
        os.makedirs(patient_output_path, exist_ok=True)
        patient_output_dyn_path = os.path.join(patient_output_path, 'dyn')
        os.makedirs(patient_output_dyn_path, exist_ok=True)

        try:
            dyn_files = os.listdir(patient_dyn_path)
        except:
            print('No dyn files for patient ', patient_ID)
            continue

        segmentation_path = os.path.join(main_dir, 'derivatives', 'tumor_segmentations', patient_ID, 'dyn')
        if os.path.exists(segmentation_path):
            segmentations = os.listdir(segmentation_path)

            lesion_files = [file for file in segmentations if not ('liver' in file or 'Unkown' in file)]
            lesions = list(set(['L' + file.split('_L')[1][0] for file in lesion_files]))
            lesion_nbr = len(lesions)
            if lesion_nbr == 0:
                print('No lesion for patient ', patient_ID)
                continue
        else:
            print('No segmentation for patient ', patient_ID)
            continue
            
        art_files = [file for file in dyn_files if 'arterial' in file and '.json' in file]
        arterial_annotations = [file for file in lesion_files if 'arterial' in file]
        if len(arterial_annotations) == 0:
            annot_phases = [file.split('_phase_')[1][0] for file in lesion_files]
            import collections
            from collections import Counter
            best_phase = Counter(annot_phases).most_common(1)[0][0]
            art_files = [file for file in dyn_files if 'phase_' + best_phase in file and '.json' in file]
        art_with_most_annot = {file: len(set(['L' + seg.split('_L')[1][0] 
                                               for seg in lesion_files if seg.startswith(file.split('mod')[0])])) 
                                               for file in art_files}
        art_best_annot = max(art_with_most_annot, key=art_with_most_annot.get)
        art_with_less_annot = [file for file in art_files if file != art_best_annot]
        dyn_files = [file for file in dyn_files if not any([art_file[:-4] in file for art_file in art_with_less_annot])]


        metadata_files = [txt for txt in dyn_files if txt[-4:] == 'json']

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

        if len(all_phases) == 0:
            print('No phases for patient ', patient_ID)
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
            img_to_keep.append(os.path.join(patient_dyn_path, json_file))

        img_to_keep = [file.replace('.json', '.nii.gz') for file in img_to_keep if any(dyn_file in file for dyn_file in dyn_files)]
        print(patient_ID, ' ', len(dyn_files)-len(phases_selection),
              ' - images dropped (before/after: ', len([file for file in os.listdir(patient_dyn_path) if '.json' in file]), '/', len(phases_selection), ')')
        if len(img_to_keep) < 4:
            print('Missing phase for patient ', patient_ID)
            continue
        img_name_to_keep = [os.path.basename(file) for file in img_to_keep]
        all_seg_files_included = [seg_file for seg_file in lesion_files if any([file.split('mod')[0] in seg_file for file in img_name_to_keep])]
        lesion_remaining = len(set(['L' + seg.split('_L')[1][0] for seg in all_seg_files_included]))
        if lesion_remaining < lesion_nbr:
            print('Missing segmentation for patient ', patient_ID)
            break
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
