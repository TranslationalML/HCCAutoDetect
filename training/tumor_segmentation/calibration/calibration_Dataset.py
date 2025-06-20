from torch.utils.data import Dataset
import torch
import numpy as np
import os
import SimpleITK as sitk


class CALIBRATION(Dataset):
    def __init__(self, exp_name, logits_list):
        self.logits_list = logits_list
        self.exp_name = exp_name

    def __len__(self):
        return len(self.logits_list)

    def __getitem__(self, id):
        id_logits = sitk.ReadImage(self.logits_list[id])
        id_logits = sitk.GetArrayFromImage(id_logits)
        id_logits = torch.from_numpy(id_logits).float()

        sub_id = os.path.basename(self.logits_list[id])
        file_path = os.path.dirname(os.path.dirname(self.logits_list[id]))
        label_item_file = self.logits_list[id].replace('logit_tta', 'ground_truth')
        image_item_file = self.logits_list[id].replace('logit_tta', 'preprocessed_img')
        id_label = sitk.ReadImage(label_item_file)
        id_label_array = sitk.GetArrayFromImage(id_label)
        id_image_nat = sitk.ReadImage(image_item_file[:-10] + 'nat_img.nii.gz')
        id_image_nat_array = sitk.GetArrayFromImage(id_image_nat)
        id_image_art = sitk.ReadImage(image_item_file[:-10] + 'art_img.nii.gz')
        id_image_art_array = sitk.GetArrayFromImage(id_image_art)
        id_image_ven = sitk.ReadImage(image_item_file[:-10] + 'ven_img.nii.gz')
        id_image_ven_array = sitk.GetArrayFromImage(id_image_ven)
        id_image_del = sitk.ReadImage(image_item_file[:-10] + 'del_img.nii.gz')
        id_image_del_array = sitk.GetArrayFromImage(id_image_del)
        id_image_4D = np.stack((id_image_nat_array, id_image_art_array, id_image_ven_array, id_image_del_array), axis=0)

        sample = {
        'image': torch.from_numpy(id_image_4D), 
        'logits': id_logits.squeeze(), 
        'label': torch.from_numpy(id_label_array), 
        'preds': 'None',
        'boundary': 'None'
        }

        return sample['image'], sample['logits'], sample['label'], sample['preds'], sample['boundary'], sub_id, file_path