import logging
import os
import sys
import json
import yaml
import monai
from monai.apps.nnunet import nnUNetV2Runner
import pandas as pd

def main(dataroot, data_to_predict, main_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    participants = pd.read_csv((main_dir + '/HCC_Surveillance/participants.tsv'), sep='\t')

    all_files = []
    for patient, patient_ID in participants[['Patient', 'Sub-ID']].values:
        print(patient_ID)
        patient_path = os.path.join(data_to_predict, patient_ID)
        patient_dyn_path = os.path.join(patient_path, 'dyn')
        select_files = patient_dyn_path + '/water_phases.json'
        if os.path.exists(select_files):
            with open(select_files, 'r') as file:
                patient_water_files = json.load(file)
            all_files.extend(patient_water_files)

    train_data_dicts = [{"image": image_name, "label": image_name} for image_name in all_files]

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
                 "dataroot": dataroot,
                 "dataset_name_or_id": 3
                 }
    yaml_path = os.getcwd() + '/input.yaml'
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(yaml_dict, file)

    runner = nnUNetV2Runner(input_config=yaml_dict)
    runner.convert_dataset()
    model_path = os.getcwd() + '/work_dir/nnUNet_trained_models/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/'
    source = os.getcwd() + '/work_dir/nnUNet_raw_data_base/Dataset003_HCC Surveillance_water/imagesTr/'
    output_dir = os.getcwd() + '/4_T1_water_images'
    os.makedirs(output_dir, exist_ok=True)

    runner.predict(list_of_lists_or_source_folder=source,
                   output_folder=output_dir,
                   model_training_output_dir=model_path,
                   use_folds=(0, 1, 2, 3, 4),
                   use_gaussian=True,
                   use_mirroring=True,
                   save_probabilities=True,
                   checkpoint_name="checkpoint_final.pth"
                   )


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))

    dataroot = os.path.join(cwd, 'data/HCC Surveillance_water')
    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(dataroot + '/imagesTs', exist_ok=True)
    os.makedirs(dataroot + '/imagesTr', exist_ok=True)

    data_to_predict = main_dir + '/HCC_Surveillance/derivatives/3_select_dyn_phases/'
    main(dataroot, data_to_predict, main_dir)
