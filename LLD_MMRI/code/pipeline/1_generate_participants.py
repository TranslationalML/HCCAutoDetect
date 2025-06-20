import os
import pandas as pd
import json

def generate_participants(main_dir):
    data_path = os.path.join(main_dir, 'sourcedata')
    data_path = data_path
    with open(os.path.join(data_path, "labels/Annotation.json"), "r") as f:
       all_annotations = json.load(f)
    patients_id = all_annotations['Annotation_info'].keys()
    def find_key_by_value(d, value):
        for key, val in d.items():
            if val == value:
                return key
        return None
    lesion_categories = all_annotations['Category_info']

    all_patients_info_dict = {}
    for patient in patients_id:
        patient_info_dict = {}
        patient_info = all_annotations['Annotation_info'][patient]
        
        patient_info_dict['studyUID'] = patient_info[0]['studyUID']
        for image_dict in patient_info:
            patient_info_dict[image_dict['phase']] = image_dict['seriesUID']
            lesion_type_codes = [image_dict['annotation']['lesion'][x]['category'] for x in image_dict['annotation']['lesion'].keys()]
            
            lesion_types = [find_key_by_value(lesion_categories, lesion_type) for lesion_type in lesion_type_codes]
            patient_info_dict[image_dict['phase'] + '_lesions'] = lesion_types
        all_patients_info_dict[patient] = patient_info_dict
    df_participants = pd.DataFrame.from_dict(all_patients_info_dict, orient='index')
    df_participants.to_csv(main_dir + '/participants_more_infos.tsv', sep="\t")


if __name__ == '__main__':
    cwd = os.getcwd()
    data_path = os.path.dirname(os.path.dirname(cwd))
    generate_participants(data_path)
