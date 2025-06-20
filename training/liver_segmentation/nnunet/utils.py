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

def create_liver_dataset(dataroot, data_dir):
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "liver_labels", "*.nii.gz")))

    for img in images[:32]:
        saving_path = dataroot + '/imagesTr/' + os.path.basename(img)
        shutil.copyfile(img, saving_path)
    for img in images[32:]:
        saving_path = dataroot + '/imagesTs/' + os.path.basename(img)
        shutil.copyfile(img, saving_path)

    for label in labels[:32]:
        saving_path = dataroot + '/labelsTr/' + os.path.basename(label)
        shutil.copyfile(label, saving_path)
    for label in labels[32:]:
        saving_path = dataroot + '/labelsTs/' + os.path.basename(label)
        shutil.copyfile(label, saving_path)
    return
