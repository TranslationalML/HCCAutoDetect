import logging
import os
import sys
import glob
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import monai
from monai.data import list_data_collate, pad_list_data_collate
from models import UNETTumorSegmentationTrainer
import yaml

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    RandFlipd,
    RandAdjustContrastd,
    ThresholdIntensityd,
)
import pandas as pd

def main(data_dir, config, CV_fold):
    torch.cuda.empty_cache()
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_images = sorted(glob.glob(os.path.join(data_dir, "images_4D", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    train_multilabels = sorted(glob.glob(os.path.join(data_dir, "multi_labels", "*.nii.gz")))
    train_liver_labels = sorted(glob.glob(os.path.join(data_dir, "liver_labels", "*.nii.gz")))
    df_all_patients = pd.read_csv(os.path.join(data_dir, 'tumors_characteristics.csv'), sep=';')
    if df_all_patients.shape[0] == 1:
        df_all_patients = pd.read_csv(os.path.join(data_dir, 'tumors_characteristics.csv'), sep=',')

    data_dicts = [{"img": image_name,
                    "seg": label_name,
                    "multi_labels": multi_labels,
                    "liver_label": liver_label_name,
                    "lesion_type": np.pad(df_all_patients.loc[df_all_patients['ID'] == os.path.basename(image_name)[:-7]]['HCC'].values, 
                                           (0, max(0, 5 - (df_all_patients.loc[df_all_patients['ID'] == os.path.basename(image_name)[:-7]]['HCC'].values).size)), 
                                           constant_values=-1), 
                    }
                    for image_name, label_name, multi_labels, liver_label_name
                    in zip(train_images, train_labels, train_multilabels, train_liver_labels)]
    print("Cross validation fold: ", CV_fold)
    if CV_fold == 0:
        val_files = data_dicts[int(0.8 * len(data_dicts)):]
        train_files = [file for file in data_dicts if file not in val_files]
    if CV_fold == 1:
        val_files = data_dicts[int(0.6 * len(data_dicts)):int(0.8 * len(data_dicts))]
        train_files = [file for file in data_dicts if file not in val_files]
    if CV_fold == 2:
        val_files = data_dicts[int(0.4 * len(data_dicts)):int(0.6 * len(data_dicts))]
        train_files = [file for file in data_dicts if file not in val_files]
    if CV_fold == 3:
        val_files = data_dicts[int(0.2 * len(data_dicts)):int(0.4 * len(data_dicts))]
        train_files = [file for file in data_dicts if file not in val_files]
    if CV_fold == 4:
        val_files = data_dicts[:int(0.2 * len(data_dicts))]
        train_files = [file for file in data_dicts if file not in val_files]

    train_transforms = Compose([LoadImaged(keys=["img", "seg", "liver_label", "multi_labels"]),
                                EnsureChannelFirstd(keys=["img", "seg", "liver_label", "multi_labels"]),
                                Orientationd(keys=["img", "seg", "liver_label", "multi_labels"], axcodes="RAS"),
                                Spacingd(keys=["img", "seg", "liver_label", "multi_labels"],
                                         pixdim=config['pixel_resampling_size'],
                                         mode=("nearest", "nearest", "nearest", "nearest"),
                                         padding_mode=('zeros', 'zeros', 'zeros', 'zeros')),
                                SpatialPadd(keys=["img", "seg", "liver_label", "multi_labels"],
                                            spatial_size=config['patch_size'],
                                            mode=('constant', 'constant', 'constant', 'constant')),
                                ScaleIntensityRangePercentilesd(keys=["img"], lower=5, upper=100, b_min=0, b_max=1, channel_wise=True),
                                RandCropByPosNegLabeld(
                                    keys=["img", "seg", "liver_label", "multi_labels"],
                                    label_key="seg",
                                    spatial_size=config['patch_size'],
                                    pos=config["foreground_patches"],
                                    neg=config["background_patches"],
                                    num_samples=config["patch_samples"],
                                    image_key="img",
                                ),
                                RandAdjustContrastd(
                                    keys=["img"],
                                    prob=0.15,
                                    gamma=config["contrast_gamma"],
                                ),
                                RandFlipd(
                                    keys=["img", "seg", "liver_label", "multi_labels"],
                                    spatial_axis=[0],
                                    prob=0.50,
                                ),
                                RandFlipd(
                                    keys=["img", "seg", "liver_label", "multi_labels"],
                                    spatial_axis=[1],
                                    prob=0.50,
                                ),

                                ThresholdIntensityd(keys=["img"], threshold=0, above=True, cval=0),
                                ThresholdIntensityd(keys=["img"], threshold=1, above=False, cval=1),
                            ]
                        )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg", "multi_labels", "liver_label"]),
            EnsureChannelFirstd(keys=["img", "seg", "multi_labels", "liver_label"]),
            Orientationd(keys=["img", "seg", "multi_labels", "liver_label"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(keys=["img"], lower=5, upper=100, b_min=0, b_max=1, channel_wise=True),
            Spacingd(keys=["img", "seg", "multi_labels", "liver_label"],
                     pixdim=config['pixel_resampling_size'],
                     mode=("nearest", "nearest", "nearest", "nearest"),
                     padding_mode=('zeros', 'zeros', 'zeros', 'zeros')),
            ThresholdIntensityd(keys=["img"], threshold=0, above=True, cval=0),
            ThresholdIntensityd(keys=["img"], threshold=1, above=False, cval=1),
        ]
    )
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(
        check_ds,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=pad_list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=config['val_batch_size'],
                            num_workers=config['num_workers'],
                            collate_fn=list_data_collate)

    lesion_detector = UNETTumorSegmentationTrainer(config, model_path=config["model_path_list"][CV_fold])
    lesion_detector.training(train_loader, val_loader, config)


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    # config_path = os.path.join(main_dir, 'training_experiments/lits_pretraining/2. Unet_Tversky.yaml')
    config_path = os.path.join(main_dir, 'training_experiments/tumor_segmentation/tumor_segmentation_finetuning/3. Unet_pretrained_Tversky_loss.yaml')

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = os.path.join(main_dir, config["dataset"])

    config['config_path'] = config_path
    config['fold'] = 0
    if config['model_path_list'][0] != None:
        config['model_path_list'] = [os.path.join(cwd, path) for path in config['model_path_list']]

    if len(sys.argv) > 1:
        config["fold"] = int(sys.argv[1])
        print(config["fold"])
        print(config["model_path_list"][config["fold"]])
        config['saving_name'] = config['saving_name'] + '_CV_fold_' + str(config['fold'])
        config["run_name"] = config['saving_name']
        main(data_dir, config, config['fold'])

    else:
        config["run_name"] = config['saving_name'] + "_CV_fold_{}".format(config['fold'])
        config['saving_name'] = config['saving_name'] + "_CV_fold_{}".format(config['fold'])
        main(data_dir, config, config['fold'])
