import logging
import os
import sys
import glob
import torch
from torch.utils.data import DataLoader
import yaml
import monai
from monai.data import list_data_collate
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
    RandFlipd,
    RandAdjustContrastd,
    ThresholdIntensityd)

def main(data_dir, config, CV_fold):
    torch.cuda.empty_cache()
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_images = sorted(glob.glob(os.path.join(data_dir, "images_4D", "*.nii.gz")))

    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    train_multilabels = sorted(glob.glob(os.path.join(data_dir, "multi_labels", "*.nii.gz")))
    train_liver_labels = sorted(glob.glob(os.path.join(data_dir, "liver_labels", "*.nii.gz")))
    data_dicts = [{"img": image_name,
                        "seg": label_name,
                        "multi_labels": multi_labels,
                        "liver_label": liver_label_name,
                        }
                       for image_name, label_name, multi_labels, liver_label_name
                       in zip(train_images, train_labels, train_multilabels, train_liver_labels)]

    data_dicts = [file for file in data_dicts if not any([x in file['img'] for x in config['test_files']])]
    val_files = data_dicts[int(0.8 * len(data_dicts)):]
    train_files = [file for file in data_dicts if file not in val_files]

    train_transforms = Compose([LoadImaged(keys=["img", "seg", "liver_label"]),
                                EnsureChannelFirstd(keys=["img", "seg", "liver_label"]),
                                Orientationd(keys=["img", "seg", "liver_label"], axcodes="RAS"),
                                Spacingd(keys=["img", "seg", "liver_label"],
                                         pixdim=config['pixel_resampling_size'],
                                         mode=("nearest", "nearest", "nearest"),
                                         padding_mode=('zeros', 'zeros', 'zeros')),
                                ScaleIntensityRangePercentilesd(keys=["img"], lower=5, upper=100, b_min=0, b_max=1, channel_wise=True),
                                RandCropByPosNegLabeld(
                                    keys=["img", "seg", "liver_label"],
                                    label_key="seg",
                                    spatial_size=config['patch_size'],
                                    pos=1,
                                    neg=10,
                                    num_samples=config["patch_samples"],
                                    image_key="img",
                                ),
                                RandAdjustContrastd(
                                    keys=["img"],
                                    prob=0.15,
                                    gamma=config["contrast_gamma"]
                                ),
                                RandFlipd(
                                    keys=["img", "seg", "liver_label"],
                                    spatial_axis=[0],
                                    prob=0.50,
                                ),
                                RandFlipd(
                                    keys=["img", "seg", "liver_label"],
                                    spatial_axis=[1],
                                    prob=0.50,
                                ),
                                RandFlipd(
                                    keys=["img", "seg", "liver_label"],
                                    spatial_axis=[2],
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
    config_path = os.path.join(main_dir, 'training_experiments/lits_pretraining/3.1 LiTS_model.yaml')

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = os.path.join(main_dir, config["dataset"])

    config['config_path'] = config_path
    config['fold'] = 0
    if config['model_path_list'][0] != None:
        config['model_path_list'] = [os.path.join(cwd, path) for path in config['model_path_list'] if path != None]


    if len(sys.argv) > 1:
        config["fold"] = int(sys.argv[1])
        print(config["fold"])
        print(config["model_path_list"][config["fold"]])
        config['saving_name'] = config['saving_name'] + '_CV_fold_' + sys.argv[1]
        config["run_name"] = config['saving_name']
        main(data_dir, config, int(sys.argv[1]))

    else:
        config["run_name"] = config['saving_name'] + "_CV_fold_{}".format(config['fold'])
        config['saving_name'] = config['saving_name'] + "_CV_fold_{}".format(config['fold'])
        main(data_dir, config, config['fold'])
