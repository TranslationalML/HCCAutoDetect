
import logging
import os
import sys
import glob
import json
import shutil
import yaml
import torch
import monai
from ruamel.yaml import YAML
import subprocess
cwd = os.getcwd()
os.environ["nnUNet_raw"] = cwd + '/work_dir/nnUNet_raw_data_base'
os.environ["nnUNet_preprocessed"] = cwd + "/work_dir/nnUNet_preprocessed"
os.environ["nnUNet_results"] = cwd + "/work_dir/nnUNet_results"


from utils import create_multimodality_tumor_dataset, get_train_test_dict
sys.path.append(os.path.dirname(os.getcwd()))
from nnunet_residual_encoder.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join


def main(data_dir, dataroot, dataset_name_or_id):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_images = sorted(glob.glob(os.path.join(data_dir, "images_4D", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    train_multilabels = sorted(glob.glob(os.path.join(data_dir, "multi_labels", "*.nii.gz")))
    
    # test dataset
    test_data_path = os.path.dirname(data_dir) + '/10_test_T1_dataset'
    test_labels = sorted(glob.glob(os.path.join(test_data_path, "labels", "*.nii.gz")))
    create_multimodality_tumor_dataset(dataroot, train_labels, test_labels)
    train_data_dicts, test_data_dicts = get_train_test_dict(dataroot)

    #Retrieve 3D images
    train_3D_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    for idx in range(len(train_data_dicts)):
        item = train_data_dicts[idx]
        images_3D = [file for file in train_3D_images if os.path.basename(item['image'])[:-7] in file]
        train_data_dicts[idx]['images_3D'] = images_3D
    test_3D_images = sorted(glob.glob(os.path.join(test_data_path, "images", "*.nii.gz")))
    for idx in range(len(test_data_dicts)):
        item = test_data_dicts[idx]
        images_3D = [file for file in test_3D_images if os.path.basename(item['image'])[:-7] in file]
        test_data_dicts[idx]['images_3D'] = images_3D

    fold_0_val = train_images[int(0.8 * len(train_images)):]
    fold_0_train = [file for file in train_images if file not in fold_0_val]
    fold_1_val = train_images[int(0.6 * len(train_images)):int(0.8 * len(train_images))]
    fold_1_train = [file for file in train_images if file not in fold_1_val]
    fold_2_val = train_images[int(0.4 * len(train_images)):int(0.6 * len(train_images))]
    fold_2_train = [file for file in train_images if file not in fold_2_val]
    fold_3_val = train_images[int(0.2 * len(train_images)):int(0.4 * len(train_images))]
    fold_3_train = [file for file in train_images if file not in fold_3_val]
    fold_4_val = train_images[:int(0.2 * len(train_images))]
    fold_4_train = [file for file in train_images if file not in fold_4_val]

    data_dict = {
                 "name": "Multimodality tumor segmentation 4D",
                 "description": "Segmentation with all phases of T1 MRI in a 4D volume",
                 "tensorImageSize": "3D",
                 "modality": {'0': 'native', '1': 'arterial', '2': 'venous', '3': 'delayed'},
                 "labels:": {"background": 0,
                             "tumor": 1},
                 "numTraining": len(train_data_dicts),
                 "numTest": len(test_data_dicts),
                 "training": train_data_dicts,
                 "test": test_data_dicts}

    nnunet_rawdata_path = os.path.join(os.getcwd(), 'work_dir', 'nnUNet_raw_data_base', 'Dataset' + str('{:03}'.format(dataset_name_or_id)) + '_' + os.path.basename(dataroot))
    os.makedirs(nnunet_rawdata_path, exist_ok=True)

    # rename patients
    all_patients_dict = train_data_dicts + test_data_dicts
    patient_new_names = {os.path.basename(patient['image'])[:-7]: f'case_{idx}' for idx, patient in enumerate(all_patients_dict)}

    for patient in train_data_dicts:
        patient['new_name'] = patient_new_names[os.path.basename(patient['image'])[:-7]]
    for patient in test_data_dicts:
        patient['new_name'] = patient_new_names[os.path.basename(patient['image'])[:-7]]
    
    datalist = {'training': train_data_dicts, 'test': test_data_dicts}
    dataset = {
                'channel_names': {'0': 'native', '1': 'arterial', '2': 'venous', '3': 'delayed'},
                'file_ending': '.nii.gz',
                'labels': {'background': 0, 'class1': 1},
                'numTraining': len(train_data_dicts),
                'numTest': len(test_data_dicts),
                }
    with open(nnunet_rawdata_path + '/datalist.json', 'w') as f:
        json.dump(datalist, f, sort_keys=False)
    with open(nnunet_rawdata_path + '/dataset.json', 'w') as f:
        json.dump(dataset, f, sort_keys=False)
    
    labelsTs_path = os.path.join(nnunet_rawdata_path, 'labelsTs')
    os.makedirs(labelsTs_path, exist_ok=True)
    for patient in test_data_dicts:
        shutil.copy(patient['label'], os.path.join(labelsTs_path, os.path.basename(patient['new_name']) + '.nii.gz'))
    
    labelsTr_path = os.path.join(nnunet_rawdata_path, 'labelsTr')
    os.makedirs(labelsTr_path, exist_ok=True)
    for patient in train_data_dicts:
        shutil.copy(patient['label'], os.path.join(labelsTr_path, os.path.basename(patient['new_name']) + '.nii.gz'))
    
    imagesTs_path = os.path.join(nnunet_rawdata_path, 'imagesTs')
    os.makedirs(imagesTs_path, exist_ok=True)
    for patient in test_data_dicts:
        for img in patient['images_3D']:
            if 'native' in img:
                new_name = os.path.basename(patient['new_name']) + '_0000.nii.gz'
            elif 'arterial' in img:
                new_name = os.path.basename(patient['new_name']) + '_0001.nii.gz'
            elif 'venous' in img:
                new_name = os.path.basename(patient['new_name']) + '_0002.nii.gz'
            elif 'delayed' in img:
                new_name = os.path.basename(patient['new_name']) + '_0003.nii.gz'
            shutil.copy(img, os.path.join(imagesTs_path, new_name))

    imagesTr_path = os.path.join(nnunet_rawdata_path, 'imagesTr')
    os.makedirs(imagesTr_path, exist_ok=True)
    for patient in train_data_dicts:
        for img in patient['images_3D']:
            if 'native' in img:
                new_name = os.path.basename(patient['new_name']) + '_0000.nii.gz'
            elif 'arterial' in img:
                new_name = os.path.basename(patient['new_name']) + '_0001.nii.gz'
            elif 'venous' in img:
                new_name = os.path.basename(patient['new_name']) + '_0002.nii.gz'
            elif 'delayed' in img:
                new_name = os.path.basename(patient['new_name']) + '_0003.nii.gz'
            shutil.copy(img, os.path.join(imagesTr_path, new_name))



    cmd_1 = "export nnUNet_raw=" + cwd + "/work_dir/nnUNet_raw_data_base"
    cmd_2 = "export nnUNet_preprocessed=" + cwd + "/work_dir/nnUNet_preprocessed"
    cmd_3 = "export nnUNet_results=" + cwd + "/work_dir/nnUNet_results"
    subprocess.run(cmd_1, shell=True, check=True, env=os.environ)
    subprocess.run(cmd_2, shell=True, check=True, env=os.environ)
    subprocess.run(cmd_3, shell=True, check=True, env=os.environ)


    if not os.path.exists(nnunet_rawdata_path.replace('nnUNet_raw_data_base', 'nnUNet_preprocessed')):
        command = f"nnUNetv2_plan_and_preprocess -d {dataset_name_or_id} --verify_dataset_integrity"
        subprocess.run(command, shell=True, check=True)
        print(f"Preprocessing for Task {dataset_name_or_id} completed successfully.")

    train_folds = [fold_0_train, fold_1_train, fold_2_train, fold_3_train, fold_4_train]
    val_folds = [fold_0_val, fold_1_val, fold_2_val, fold_3_val, fold_4_val]
    for fold_idx in range(5):
        train_folds[fold_idx] = [patient_new_names[os.path.basename(nnunet_file)[:-7]] for nnunet_file in train_folds[fold_idx]]
        val_folds[fold_idx] = [patient_new_names[os.path.basename(nnunet_file)[:-7]] for nnunet_file in val_folds[fold_idx]]

    cross_validation = [{'train': train_folds[0], 'val': val_folds[0]},
                        {'train': train_folds[1], 'val': val_folds[1]},
                        {'train': train_folds[2], 'val': val_folds[2]},
                        {'train': train_folds[3], 'val': val_folds[3]},
                        {'train': train_folds[4], 'val': val_folds[4]}]
    dataset_split_path = nnunet_rawdata_path.replace('nnUNet_raw_data_base', 'nnUNet_preprocessed') + '/splits_final.json'
    with open(dataset_split_path, 'w') as f:
        json.dump(cross_validation, f, sort_keys=False)
    
    predictor = nnUNetPredictor(
                                tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=True,
                                perform_everything_on_device=True,
                                device=torch.device('cuda', 0),
                                verbose=True,
                                verbose_preprocessing=True,
                                allow_tqdm=True
                                )


    predictor.initialize_from_trained_model_folder(
        join(os.environ["nnUNet_results"], 'Dataset' + str('{:03}'.format(dataset_name_or_id)) + '_' + 
             os.path.basename(dataroot)) + '/' + config['trainer']['class_name'] + '__nnUNetPlans__3d_fullres',
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor.predict_from_files(join(os.environ['nnUNet_raw'], nnunet_rawdata_path + '/imagesTs'),
                                join(os.environ['nnUNet_raw'], nnunet_rawdata_path + '/imagesTsPreds'),
                                save_probabilities=True, overwrite=True,
                                num_processes_preprocessing=2, 
                                num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, 
                                num_parts=1, 
                                part_id=0)


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    yaml = YAML()   
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_pretraining/1. Benchmark_nnunet.yaml')
    with open(config_path, 'r') as file:
        config = yaml.load(file)

    dataset_name_or_id = config['dataset']['name_or_id']
    trainer_class_name = config['trainer']['class_name']
    dataroot = os.path.join(cwd, config['dataset']['dataroot'])
    data_dir = os.path.join(main_dir, config['dataset']['datadir'])


    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(dataroot + '/imagesTr', exist_ok=True)
    os.makedirs(dataroot + '/labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/imagesTs', exist_ok=True)
    os.makedirs(dataroot + '/labelsTs', exist_ok=True)
    os.makedirs(dataroot + '/multi_labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/multi_labelsTs', exist_ok=True)
    os.makedirs(dataroot + '/liver_labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/liver_labelsTs', exist_ok=True)

    main(data_dir, dataroot, dataset_name_or_id)
