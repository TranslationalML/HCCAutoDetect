
import logging
import os
import sys
import glob
import json
import shutil
import yaml
import torch
import monai
from monai.apps.nnunet import nnUNetV2Runner
from utils import create_liver_dataset, get_train_test_dict


def main(CHUV_data_dir, dataroot, dataset_name_or_id, trainer_class_name):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    create_liver_dataset(dataroot, CHUV_data_dir)
    train_data_dicts, test_data_dicts = get_train_test_dict(dataroot)
    data_dict = {
                 "name": "Liver",
                 "description": "Liver segmentation",
                 "tensorImageSize": "3D",
                 "modality": {"0": "MRI"},
                 "labels:": {"background": 0,
                             "liver": 1},
                 "numTraining": len(train_data_dicts),
                 "numTest": len(test_data_dicts),
                 "training": train_data_dicts,
                 "test": test_data_dicts}
    dataset_path = os.getcwd() + '/dataset.json'
    with open(dataset_path, 'w') as f:
        json.dump(data_dict, f, sort_keys=False)

    yaml_dict = {"modality": "MRI",
                 "datalist": dataset_path,
                 "dataroot": dataroot,
                 "dataset_name_or_id": dataset_name_or_id
                 }
    yaml_path = os.getcwd() + '/input.yaml'
    with open(yaml_path, 'w') as file:
        documents = yaml.dump(yaml_dict, file)

    runner = nnUNetV2Runner(input_config=yaml_dict, trainer_class_name=trainer_class_name)
    runner.convert_dataset()
    runner.plan_and_process()
    runner.train_single_model(config='3d_fullres', fold=0)
    runner.train_single_model(config='3d_fullres', fold=1)
    runner.train_single_model(config='3d_fullres', fold=2)
    runner.train_single_model(config='3d_fullres', fold=3)
    runner.train_single_model(config='3d_fullres', fold=4)


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))

    config_path = os.path.join(main_dir, 'training_experiments/liver_segmentation/nnunet_liver_segmentation.yaml')
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset_name_or_id = config['dataset']['name_or_id']
    trainer_class_name = config['trainer']['class_name']
    dataroot = os.path.join(cwd, config['dataset']['dataroot'])
    data_dir = os.path.join(main_dir, config['dataset']['datadir'])

    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(dataroot + '/imagesTr', exist_ok=True)
    os.makedirs(dataroot + '/labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/imagesTs', exist_ok=True)
    os.makedirs(dataroot + '/labelsTs', exist_ok=True)

    main(data_dir, dataroot, dataset_name_or_id, trainer_class_name, finetuning=config['trainer'])
