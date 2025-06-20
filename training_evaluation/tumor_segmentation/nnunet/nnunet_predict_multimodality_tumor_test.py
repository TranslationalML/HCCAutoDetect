import os
import json
import yaml
import sys
cwd = os.getcwd()
os.environ["nnUNet_raw"] = cwd + '/work_dir/nnUNet_raw_data_base'
os.environ["nnUNet_preprocessed"] = cwd + "/work_dir/nnUNet_preprocessed"
os.environ["nnUNet_results"] = cwd + "/work_dir/nnUNet_results"

training_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'training/tumor_segmentation'))
sys.path.append(training_dir)

from nnunet_residual_encoder.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join
from utils import get_train_test_dict, create_multimodality_tumor_dataset
import glob
import shutil
import torch
import yaml
from ruamel.yaml import YAML

def main(data_dir, output_dir, source, dataset_ID, model_path, config):
    os.makedirs(output_dir, exist_ok=True)
    dataroot = os.path.join(cwd, 'data/' + os.path.basename(dataset_dir))
    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(dataroot + '/imagesTr', exist_ok=True)
    os.makedirs(dataroot + '/labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/imagesTs', exist_ok=True)
    os.makedirs(dataroot + '/labelsTs', exist_ok=True)
    os.makedirs(dataroot + '/multi_labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/multi_labelsTs', exist_ok=True)
    os.makedirs(dataroot + '/liver_labelsTr', exist_ok=True)
    os.makedirs(dataroot + '/liver_labelsTs', exist_ok=True)

    images = sorted(glob.glob(os.path.join(data_dir, "images_4D", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    multilabels = sorted(glob.glob(os.path.join(data_dir, "multi_labels", "*.nii.gz")))

    create_multimodality_tumor_dataset(dataroot, labels, labels[:1])
    train_data_dicts, test_data_dicts = get_train_test_dict(dataroot)

    train_3D_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    for idx in range(len(train_data_dicts)):
        item = train_data_dicts[idx]
        images_3D = [file for file in train_3D_images if os.path.basename(item['image'])[:-7] in file]
        train_data_dicts[idx]['images_3D'] = images_3D
    test_3D_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    for idx in range(len(test_data_dicts)):
        item = test_data_dicts[idx]
        images_3D = [file for file in test_3D_images if os.path.basename(item['image'])[:-7] in file]
        test_data_dicts[idx]['images_3D'] = images_3D

    nnunet_rawdata_path = os.path.join(os.getcwd(), 'work_dir', 'nnUNet_raw_data_base', 'Dataset' + str('{:03}'.format(dataset_ID)) + '_' + os.path.basename(dataroot))
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
                                                    join(model_path),
                                                    use_folds=(0, 1, 2, 3, 4),
                                                    checkpoint_name='checkpoint_best.pth',
                                                    )
    predictor.predict_from_files(join(os.environ['nnUNet_raw'], nnunet_rawdata_path + '/imagesTr'),
                                join(os.environ['nnUNet_raw'], nnunet_rawdata_path + '/imagesTrPreds'),
                                save_probabilities=True, overwrite=True,
                                num_processes_preprocessing=2, 
                                num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, 
                                num_parts=1, 
                                part_id=0)

if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    training_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/training/tumor_segmentation/nnunet"

    if len(sys.argv) > 1:
        fold = int(sys.argv[1])
    else:
        fold = 0

    config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_pretraining/1. Benchmark_nnunet.yaml')
    yaml = YAML()   
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(config_path, 'r') as file:
        config = yaml.load(file)

    output_dir = os.getcwd() + '/test_set_prediction/10_test_T1_dataset' + str(fold)
    dataset_dir = os.path.join(main_dir, 'HCC_Surveillance/derivatives/10_test_T1_dataset')
    source = os.getcwd() + '/work_dir/nnUNet_raw_data_base/Dataset034_10_test_T1_dataset/imagesTr/'
    model_path = training_dir + '/work_dir/nnUNet_results/Dataset030_HCC_Surveillance/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/'
    dataset_ID = 34
    main(dataset_dir, output_dir, source, dataset_ID, model_path, config)

    # output_dir = os.getcwd() + '/test_set_prediction/6_T1_dataset' + str(fold)
    # dataset_dir = os.path.join(main_dir, 'CHUV_RFAvsMWA/derivatives/6_T1_dataset')
    # source = os.getcwd() + '/work_dir/nnUNet_raw_data_base/Dataset033_6_T1_dataset/imagesTr/'
    # model_path = training_dir + '/work_dir/nnUNet_results/Dataset030_HCC_Surveillance/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/'
    # dataset_ID = 33
    # main(dataset_dir, output_dir, source, dataset_ID, model_path, config)