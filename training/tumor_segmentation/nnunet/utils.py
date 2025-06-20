import glob
import os
import shutil

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