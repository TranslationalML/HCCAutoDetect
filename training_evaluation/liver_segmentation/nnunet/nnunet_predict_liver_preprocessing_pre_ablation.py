import logging
import os
import sys
import json
import shutil
import yaml
import pandas as pd

import monai
from monai.apps.nnunet import nnUNetV2Runner


def main(data_to_predict, output_dir, data_set_ID):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    participants = pd.read_csv(main_dir + '/HCC_pre_ablation/participants.tsv', sep="\t")
    all_files = []
    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        print(patient_ID)
        patient_path = os.path.join(data_to_predict, patient_ID)
        patient_dyn_path = os.path.join(patient_path, 'dyn')
        select_files = patient_dyn_path + '/phases_to_keep.json'
        if os.path.exists(select_files):
            with open(select_files, 'r') as file:
                patient_water_files = json.load(file)
            all_files.extend(patient_water_files)

    all_files_selection = [file for file in all_files]
    train_data_dicts = [{"image": image_name, "label": image_name}
                       for image_name in all_files_selection]

    data_dict = {
        "name": "HCC Surveillance Liver",
        "description": "Liver segmentation",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI"},
        "labels": {"background": 0,
                   "liver": 1,
                   },
        "test": train_data_dicts[:1],
        "training": train_data_dicts}
    dataset_path = os.getcwd() + '/dataset.json'
    with open(dataset_path, 'w') as f:
        json.dump(data_dict, f, sort_keys=False)

    yaml_dict = {"modality": "MRI",
                 "datalist": dataset_path,
                 "dataroot": data_to_predict,
                 "dataset_name_or_id": data_set_ID,
                 }
    yaml_path = os.getcwd() + '/input.yaml'
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(yaml_dict, file)

    runner = nnUNetV2Runner(input_config=yaml_dict)
    runner.convert_dataset()
    model_path = main_dir + '/training/liver_segmentation/nnunet/work_dir/nnUNet_trained_models/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/'
    source = os.getcwd() + '/work_dir/nnUNet_raw_data_base/Dataset006_3_select_dyn_phases/imagesTr/'

    runner.predict(list_of_lists_or_source_folder=source,
                   output_folder=output_dir,
                   model_training_output_dir=model_path,
                   use_folds=(0, 1, 2, 3, 4),
                   use_gaussian=True,
                   use_mirroring=True,
                   save_probabilities=True
                   ,
                   checkpoint_name="checkpoint_final.pth"
                   )

    json_convert_names_dict = os.path.join(main_dir, 'training_evaluation/liver_segmentation/nnunet/work_dir/nnUNet_raw_data_base/Dataset006_3_select_dyn_phases/datalist.json')
    with open(json_convert_names_dict) as f:
        convert_names_dict = json.load(f)

    for patient_img in convert_names_dict['training']:
        sub_id = os.path.basename(os.path.dirname(os.path.dirname(patient_img['image'])))
        output_dir_patient = os.path.join(os.path.dirname(output_dir), sub_id)
        output_dir_dyn = os.path.join(output_dir_patient, 'dyn')
        os.makedirs(output_dir_dyn, exist_ok=True)
        pred_file = os.path.join(output_dir, patient_img['new_name'] + '.nii.gz')
        pred_output_file = os.path.join(output_dir_dyn, os.path.basename(patient_img['image']))
        if os.path.exists(pred_file):
            shutil.copyfile(pred_file, pred_output_file)


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    data_set_ID = 6
    data_to_predict = main_dir + '/HCC_pre_ablation/derivatives/3_select_dyn_phases/'
    output_dir = main_dir + '/HCC_pre_ablation/derivatives/3_select_dyn_phases/predictions'
    os.makedirs(output_dir, exist_ok=True)
    main(data_to_predict, output_dir, data_set_ID)
