import os
import numpy as np
import SimpleITK as sitk
import shutil
import glob
import json


def get_train_test_dict(dataroot):
    train_images = sorted(glob.glob(os.path.join(dataroot, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(dataroot, "labelsTr", "*.nii.gz")))
    train_data_dicts = [{"image": image_name, "label": label_name}
                       for image_name, label_name in zip(train_images, train_labels)]

    test_images = sorted(glob.glob(os.path.join(dataroot, "imagesTs", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(dataroot, "labelsTs", "*.nii.gz")))
    test_data_dicts = [{"image": image_name, "label": label_name}
                       for image_name, label_name in zip(test_images, test_labels)]
    return train_data_dicts, test_data_dicts



def create_multimodality_tumor_dataset(dataroot, train_labels, test_labels):
    for label in train_labels:
        label_saving_path = dataroot + '/labelsTr/' + os.path.basename(label)
        shutil.copyfile(label, label_saving_path)
        img = label.replace('/labels', '/images_4D')
        saving_path = dataroot + '/imagesTr/' + os.path.basename(img)
        shutil.copyfile(img, saving_path)
        multilabels = label.replace('/labels', '/multi_labels')
        saving_path = dataroot + '/multi_labelsTr/' + os.path.basename(img)
        shutil.copyfile(multilabels, saving_path)
        liver_labels = label.replace('/labels', '/liver_labels')
        saving_path = dataroot + '/liver_labelsTr/' + os.path.basename(img)
        shutil.copyfile(liver_labels, saving_path)

    for label in test_labels:
        saving_path = dataroot + '/labelsTs/' + os.path.basename(label)
        shutil.copyfile(label, saving_path)
        img = label.replace('/labels', '/images_4D')
        saving_path = dataroot + '/imagesTs/' + os.path.basename(img)
        shutil.copyfile(img, saving_path)
        multilabels = label.replace('/labels', '/multi_labels')
        saving_path = dataroot + '/multi_labelsTs/' + os.path.basename(img)
        shutil.copyfile(multilabels, saving_path)
        liver_labels = label.replace('/labels', '/liver_labels')
        saving_path = dataroot + '/liver_labelsTs/' + os.path.basename(img)
        shutil.copyfile(liver_labels, saving_path)
    return


def rename_patients(patient_predictions_dir, labels_dir, data_list_json):
    with open(data_list_json, 'r') as f:
        data_list = json.load(f)

    patient_match_names = {}
    for patient_dict in data_list['training']:
        nnunet_name = patient_dict['new_name']
        patient_match_names[nnunet_name] = os.path.basename(patient_dict['image'])[:-7]

    prob_map_paths = os.path.join(patient_predictions_dir, 'prob_maps')
    os.makedirs(prob_map_paths, exist_ok=True)
    preds_05_paths = os.path.join(patient_predictions_dir, 'preds_0.5')
    os.makedirs(preds_05_paths, exist_ok=True)
    for file in os.listdir(patient_predictions_dir):
        if 'prob_map' in file:
            continue
        if file.endswith('.npz'):
            nnunet_patient_name = file[:-4]
            new_patient_name = patient_match_names[nnunet_patient_name]

            prob_map_saving_name = os.path.join(prob_map_paths, patient_match_names[nnunet_patient_name] + '.nii.gz')
            if not os.path.exists(prob_map_saving_name):
                prob_map = np.load(os.path.join(patient_predictions_dir, file))
                prob_map_tumor = prob_map['probabilities'][1]
                prob_map_tumor_image = sitk.GetImageFromArray(prob_map_tumor)

                patient_pred_img = sitk.ReadImage(os.path.join(patient_predictions_dir, file).replace('.npz', '.nii.gz'))
                prob_map_tumor_image.SetSpacing(patient_pred_img.GetSpacing())
                prob_map_tumor_image.SetOrigin(patient_pred_img.GetOrigin())
                prob_map_tumor_image.SetDirection(patient_pred_img.GetDirection())

                sitk.WriteImage(prob_map_tumor_image, prob_map_saving_name)

        elif file.endswith('.nii.gz'):
            nnunet_patient_name = file[:-7]
            file_type = '.nii.gz'
            sub_dir = preds_05_paths
            new_patient_name = patient_match_names[nnunet_patient_name]
            shutil.copyfile(os.path.join(patient_predictions_dir, file), os.path.join(sub_dir, new_patient_name + file_type))
        else:
            continue

    gt_path = os.path.join(patient_predictions_dir, 'ground_truth')
    os.makedirs(gt_path, exist_ok=True)
    for gt in os.listdir(labels_dir):
        gt_nnunet_name = os.path.basename(gt)[:-7]
        new_name = patient_match_names[gt_nnunet_name]
        shutil.copyfile(os.path.join(labels_dir, gt), os.path.join(patient_predictions_dir, gt_path, new_name + '.nii.gz'))
    return
