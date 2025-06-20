import logging
import os
import sys
import json
import yaml

import monai
from monai.apps.nnunet import nnUNetV2Runner


def main(data_to_predict, output_dir, data_set_ID):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))

    all_files_selection = [os.path.join(data_to_predict, file) for file in os.listdir(data_to_predict)]
    train_data_dicts = [{"image": image_name, "label": image_name}
                       for image_name in all_files_selection]
    data_dict = {
        "name": "LLD_MMRI",
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
    model_path = main_dir + '/training/liver_segmentation/nnunet/work_dir/nnUNet_trained_models/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres/'

    source = data_to_predict
    runner.predict(list_of_lists_or_source_folder=source,
                   output_folder=output_dir,
                   model_training_output_dir=model_path,
                   use_folds=(0, 1, 2, 3, 4),
                   use_gaussian=True,
                   use_mirroring=True,
                   save_probabilities=True,
                   checkpoint_name="checkpoint_final.pth",
                   )


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    
    data_set_ID = 5
    data_to_predict = main_dir + '/training/liver_segmentation/nnunet/data/Dataset005_7_grouped_images_for_nnunet_predictions/imagesTr/'
    output_dir = main_dir + '/training/liver_segmentation/nnunet/data/Dataset005_7_grouped_images_for_nnunet_predictions/predictions/'
    os.makedirs(output_dir, exist_ok=True)

    main(data_to_predict, output_dir, data_set_ID)
