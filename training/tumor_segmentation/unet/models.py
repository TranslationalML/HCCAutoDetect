import wandb
import SimpleITK as sitk
import torch
import yaml
import time
import pandas as pd
from collections import Counter
import argparse
import numpy as np

import monai
from tqdm import tqdm
from torch.optim.lr_scheduler import PolynomialLR, CosineAnnealingLR
from test_time_augmentation import CustomTestTimeAugmentation
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    SaveImage
)
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import shutil
from sklearn.metrics import auc
from monai.inferers import sliding_window_inference

import sys
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

from calibration.calibration_Dataset import CALIBRATION
from torch.utils.data import DataLoader
from calibration.calibration_models import LTS_CamVid_With_Image
from torch import nn, optim


class UNETTumorSegmentationEval():
    def __init__(self, config, model_path):
        super().__init__()

        self._model = monai.networks.nets.AttentionUnet(**config['model_params'])
        self._max_epochs = config['max_epochs']
        self._check_point_rate = config['check_point_rate']
        if config.get('lesion_characteristics', None):
            self._lesion_characteristics = config['lesion_characteristics']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        self._threshold = config['post_trans_threshold']
        self._learning_rate = config['learning_rate']

        self._loss_function = monai.losses.TverskyLoss(sigmoid=True, include_background=False, alpha=config['alpha'],
                                                       beta=config['beta'])
        self._loss_type = "TverskyLoss"
        self._optimizer = torch.optim.Adam(self._model.parameters(), np.float64(config['learning_rate']))
        self._dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self._post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=config['post_trans_threshold'])])
        if config['lr_scheduler'] == 'cosine':
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=config['max_epochs']/10, eta_min=np.float64(config['learning_rate']))
        else:
            self._scheduler = PolynomialLR(self._optimizer, total_iters=config['max_epochs'], power=0.9)
        self._start_epoch = 0
        if model_path != None and model_path != 'None':
            checkpoint = torch.load(model_path)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            if config["load_previous_model_params"]:
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self._start_epoch = checkpoint['epoch'] + 1

    def eval(self, device, dataloader, config):
        self._model.eval()
        with (torch.no_grad()):
            val_images = None
            val_labels = None
            val_outputs = None

            metrics_list = []
            LIRADS_list = []
            wash_ven_list = []
            wash_del_list = []
            caps_ven_list = []
            caps_del_list = []
            hyper_art_list = []
            TP_list = []
            FP_list = []
            FN_list = []
            TN_list = []
            step = 0
            epoch_loss = 0
            dice_base_loss = 0
            size_loss = 0
            wb_logs = ()
            patch_list_selection = []
            if config.get('lesion_characteristics', None):
                lesions_characteristics = pd.read_csv(self._lesion_characteristics)
            plt_number = 0
            for val_data in dataloader:
                if config.get('lesion_characteristics', None):
                    sub_df = lesions_characteristics[lesions_characteristics['ID'] == os.path.basename(val_data['img_meta_dict']['filename_or_obj'][0])[:7]]
                step += 1
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = config['patch_size']
                sw_batch_size = 8

                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self._model, overlap=config['overlap'], 
                                                       mode=config['sliding_window_inf_mode'], 
                                                       sigma_scale=config['sliding_window_sigma_scale'])
                loss = self._loss_function(val_outputs, val_labels)
                dice_base_loss += loss.item()
                val_outputs_post = [self._post_trans(i) for i in decollate_batch(val_outputs)]
                self._dice_metric(y_pred=val_outputs_post, y=val_labels)

                if step == 1:
                    if val_images.shape[1] > 1:
                        wb_logs = self.log_slices(val_images[0, 1, :, :, :].detach().cpu(), val_labels[0][0].detach().cpu(),
                                         val_outputs_post[0][0])
                    else:
                        wb_logs = self.log_slices(val_images[0, 0, :, :, :].detach().cpu(),
                                                  val_labels[0][0].detach().cpu(),
                                                  val_outputs_post[0][0])

                val_multilab = val_data["multi_labels"][0][0].cpu().detach().numpy()
                gt_tumor_idx = Counter(val_multilab.ravel())
                del gt_tumor_idx[0]
                post_trans_test = Compose([Activations(sigmoid=True)])

                threshold_TP = []
                threshold_FP = []
                threshold_FN = []
                threshold_TN = []
                threshold_LIRADS = []
                threshold_wash_ven = []
                threshold_wash_del = []
                threshold_caps_ven = []
                threshold_caps_del = []
                threshold_hyper_art = []
                for threshold in np.arange(0.1, 1, 0.1):
                    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=threshold)])
                    val_outputs_post = [post_trans(i) for i in decollate_batch(val_outputs)]

                    multi_label_pred_map, tumor_pred_idx = ndimage.label(val_outputs_post[0][0].cpu().detach().numpy())
                    TP = []
                    FN = []
                    LIRADS = []
                    wash_ven = []
                    wash_del = []
                    caps_ven = []
                    caps_del = []
                    hyper_art = []
                    if len(gt_tumor_idx) != 0:
                        for gt_label in gt_tumor_idx.keys():
                            if config.get('lesion_characteristics', None):
                                tumor_characteristics = sub_df[sub_df['label'] == gt_label]
                                LIRADS.append(tumor_characteristics['LIRADS'].values[0])
                                wash_ven.append(tumor_characteristics['Venous washout'].values[0])
                                wash_del.append(tumor_characteristics['Delayed washout'].values[0])
                                caps_ven.append(tumor_characteristics['Venous capsule'].values[0])
                                caps_del.append(tumor_characteristics['Delayed capsule'].values[0])
                                hyper_art.append(tumor_characteristics['Arterial'].values[0])

                            gt_binary_map = (val_multilab == gt_label).astype(np.uint8)
                            touching_pred_label = Counter((gt_binary_map*multi_label_pred_map).ravel())
                            del touching_pred_label[0]

                            if len(touching_pred_label) == 0:
                                FN.append(1)
                                TP.append(0)
                            else:
                                TP.append(1)
                                FN.append(0)
                    FP = tumor_pred_idx-np.sum(TP) if tumor_pred_idx > np.sum(TP) else 0
                    TN = len(gt_tumor_idx) - FP if len(gt_tumor_idx) > FP else 0

                    threshold_FP.append(FP)
                    threshold_FN.append(FN)
                    threshold_TP.append(TP)
                    threshold_TN.append(TN)
                    threshold_LIRADS.append(LIRADS)
                    threshold_wash_ven.append(wash_ven)
                    threshold_wash_del.append(wash_del)
                    threshold_caps_ven.append(caps_ven)
                    threshold_caps_del.append(caps_del)
                    threshold_hyper_art.append(hyper_art)
                TP_list.append(threshold_TP)
                FP_list.append(threshold_FP)
                FN_list.append(threshold_FN)
                TN_list.append(threshold_TN)
                LIRADS_list.append(threshold_LIRADS)
                wash_ven_list.append(threshold_wash_ven)
                wash_del_list.append(threshold_wash_del)
                caps_ven_list.append(threshold_caps_ven)
                caps_del_list.append(threshold_caps_del)
                hyper_art_list.append(threshold_hyper_art)

            tpr_list = []
            fpr_list = []
            precision_list = []
            f1_score_list = []
            f2_score_list = []

            LR_5_list, LR_4_list, LR_3_list, LR_2_list, LR_M_list, LR_TIV_list = [], [], [], [], [], []
            ven_wash_perf_list, del_wash_perf_list, ven_caps_perf_list, del_caps_perf_list, hyper_art_perf_list = [], [], [], [], []
            for thresh in range(9):
                TP = np.sum([np.sum(TP_[thresh]) for TP_ in TP_list])
                FP = np.sum([np.sum(FP_[thresh]) for FP_ in FP_list])
                FN = np.sum([np.sum(FN_[thresh]) for FN_ in FN_list])
                TN = np.sum([np.sum(TN_[thresh]) for TN_ in TN_list])
                if config.get('lesion_characteristics', None):
                    LIRADS = [
                        [[d for d, c in zip(data_list[thresh], cond_list[thresh]) if c]]
                        for data_list, cond_list in zip(LIRADS_list, TP_list)
                    ]
                    LR_5 = np.sum([1 if d == '5' else 0 for data_list in LIRADS for sub_data in data_list for d in sub_data])
                    LR_4 = np.sum([1 if d == '4' else 0 for data_list in LIRADS for sub_data in data_list for d in sub_data])
                    LR_3 = np.sum([1 if d == '3' else 0 for data_list in LIRADS for sub_data in data_list for d in sub_data])
                    LR_2 = np.sum([1 if d == '2' else 0 for data_list in LIRADS for sub_data in data_list for d in sub_data])
                    LR_M = np.sum([1 if d == 'M' else 0 for data_list in LIRADS for sub_data in data_list for d in sub_data])
                    LR_TIV = np.sum([1 if d == 'TIV' else 0 for data_list in LIRADS for sub_data in data_list for d in sub_data])


                    ven_wash = np.sum([d for data_list, cond_list in zip(wash_ven_list, TP_list)
                            for d, c in zip(data_list[thresh], cond_list[thresh]) if c])
                    ven_caps = np.sum([d for data_list, cond_list in zip(caps_ven_list, TP_list)
                                    for d, c in zip(data_list[thresh], cond_list[thresh]) if c])
                    del_wash = np.sum([d for data_list, cond_list in zip(wash_del_list, TP_list)
                                    for d, c in zip(data_list[thresh], cond_list[thresh]) if c])
                    del_caps = np.sum([d for data_list, cond_list in zip(caps_del_list, TP_list)
                                    for d, c in zip(data_list[thresh], cond_list[thresh]) if c])
                    hyper_art = np.sum([d for data_list, cond_list in zip(hyper_art_list, TP_list)
                                    for d, c in zip(data_list[thresh], cond_list[thresh]) if c])

                    LR_5_total = np.sum([1 if d == '5' else 0 for data_list in LIRADS_list for sub_data in data_list[0] for d in sub_data])
                    LR_4_total = np.sum([1 if d == '4' else 0 for data_list in LIRADS_list for sub_data in data_list[0] for d in sub_data])
                    LR_3_total = np.sum([1 if d == '3' else 0 for data_list in LIRADS_list for sub_data in data_list[0] for d in sub_data])
                    LR_2_total = np.sum([1 if d == '2' else 0 for data_list in LIRADS_list for sub_data in data_list[0] for d in sub_data])
                    LR_M_total = np.sum([1 if d == 'M' else 0 for data_list in LIRADS_list for sub_data in data_list[0] for d in sub_data])
                    LR_TIV_total = np.sum([1 if d == 'TIV' else 0 for data_list in LIRADS_list for sub_data in data_list[0] for d in sub_data])

                    ven_wash_total = np.sum([np.sum(data_list[thresh]) for data_list in wash_ven_list])
                    del_wash_total = np.sum([np.sum(data_list[thresh]) for data_list in wash_del_list])
                    ven_caps_total = np.sum([np.sum(data_list[thresh]) for data_list in caps_ven_list])
                    del_caps_total = np.sum([np.sum(data_list[thresh]) for data_list in caps_del_list])
                    hyper_art_total = np.sum([np.sum(data_list[thresh]) for data_list in hyper_art_list])


                if TP != 0:
                    recall = TP / (TP + FN)
                    precision = TP / (TP + FP)
                else:
                    recall = 0
                    precision = 0
                if FP != 0:
                    fpr = FP / (FP + TN)
                else:
                    fpr = 0
                fpr_list.append(fpr)
                tpr_list.append(recall)
                precision_list.append(precision)
                if precision != 0 and recall != 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    f2 = 5 * (precision * recall) / (4 * precision + recall)
                else:
                    f1 = 0
                    f2 = 0
                f1_score_list.append(f1)
                f2_score_list.append(f2)

                if config.get('lesion_characteristics', None):
                    LR_5_list.append(LR_5/LR_5_total) if LR_5_total != 0 else LR_5_list.append(0)
                    LR_4_list.append(LR_4/LR_4_total) if LR_4_total != 0 else LR_4_list.append(0)
                    LR_3_list.append(LR_3/LR_3_total) if LR_3_total != 0 else LR_3_list.append(0)
                    LR_2_list.append(LR_2/LR_2_total) if LR_2_total != 0 else LR_2_list.append(0)
                    LR_M_list.append(LR_M/LR_M_total) if LR_M_total != 0 else LR_M_list.append(0)
                    LR_TIV_list.append(LR_TIV/LR_TIV_total) if LR_TIV_total != 0 else LR_TIV_list.append(0)
                    ven_wash_perf_list.append(ven_wash/ven_wash_total) if ven_wash_total != 0 else ven_wash_perf_list.append(0)
                    del_wash_perf_list.append(del_wash/del_wash_total) if del_wash_total != 0 else del_wash_perf_list.append(0)
                    ven_caps_perf_list.append(ven_caps/ven_caps_total) if ven_caps_total != 0 else ven_caps_perf_list.append(0)
                    del_caps_perf_list.append(del_caps/del_caps_total) if del_caps_total != 0 else del_caps_perf_list.append(0)
                    hyper_art_perf_list.append(hyper_art/hyper_art_total) if hyper_art_total != 0 else hyper_art_list.append(0)
            fpr_list.append(0)
            tpr_list.append(0)
            fpr_inv = fpr_list[::-1]
            tpr_inv = tpr_list[::-1]
            fpr_inv.append(1)
            tpr_inv.append(1)

            try:
                auc_score = auc(fpr_inv[:-1], tpr_inv[:-1])
            except:
                auc_score = 0
            dice_metric_agg = self._dice_metric.aggregate().item()
            self._dice_metric.reset()

            epoch_loss /= step
            dice_base_loss /= step
            size_loss /= step

            LR_5_list = [0] + LR_5_list + [1]
            LR_4_list = [0] + LR_4_list + [1]
            LR_3_list = [0] + LR_3_list + [1]
            LR_2_list = [0] + LR_2_list + [1]
            LR_M_list = [0] + LR_M_list + [1]
            LR_TIV_list = [0] + LR_TIV_list + [1]
            ven_wash_perf_list = [0] + ven_wash_perf_list + [1]
            del_wash_perf_list = [0] + del_wash_perf_list + [1]
            ven_caps_perf_list = [0] + ven_caps_perf_list + [1]
            del_caps_perf_list = [0] + del_caps_perf_list + [1]
            hyper_art_perf_list = [0] + hyper_art_perf_list + [1]

            return (epoch_loss, dice_base_loss, dice_metric_agg, tpr_list[:-1], precision_list, f1_score_list, f2_score_list,
                    auc_score, fpr_inv, LR_5_list, LR_4_list, LR_3_list, LR_2_list, LR_M_list, LR_TIV_list, ven_wash_perf_list,
                    del_wash_perf_list, ven_caps_perf_list, del_caps_perf_list, hyper_art_perf_list,
                    size_loss, patch_list_selection, wb_logs)

    def plot_patches(self, patch_list_selection):
        weird_patch_logs = []
        for patch in range(len(patch_list_selection)):
            best_plane = np.argmax(
                [np.sum(patch_list_selection[patch][1][:, :, i]) for i in range(patch_list_selection[0][1].shape[2])])
            plt.imshow(patch_list_selection[patch][1][:, :, best_plane], cmap='gray', alpha=0.5)
            plt.imshow(patch_list_selection[patch][4][:, :, best_plane], cmap='gray', alpha=0.5)
            plt.imshow(patch_list_selection[patch][0][0, :, :, best_plane], cmap='gray', alpha=0.25)
            weird_patch_logs.append(wandb.Image(plt, caption=f"Patch: {patch}"))
            plt.close()
        return weird_patch_logs

    def log_slices(self, image, label, prediction):
        wandb_mask_logs = []
        wandb_img_logs = []
        wandb_prediction_logs = []

        for img_slice_no in range(image.shape[2]):
            img = image[:, :, img_slice_no]
            lbl = label[:, :, img_slice_no]
            pred = prediction[:, :, img_slice_no]
            wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {img_slice_no}"))
            wandb_mask_logs.append(wandb.Image(img, masks={"ground truth" :
                                                                  {"mask_data" : lbl,
                                                                   "class_labels" : {0: "background", 1: "mask"} }}))
            wandb_prediction_logs.append(wandb.Image(img, masks={"prediction":
                                                               {"mask_data": pred,
                                                                "class_labels": {0: "background", 1: "mask"}}}))

        return wandb_img_logs, wandb_mask_logs, wandb_prediction_logs


    def predict(self, device, dataloader, output_dir, config, sub_dir, multi_mod, fold):
        self._model.eval()
        with torch.no_grad():
            images = None
            labels = None
            outputs = None

            saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preds_{config["post_trans_threshold"]}',
                              output_ext=".nii.gz",
                              output_postfix="seg",
                              separate_folder=False)
            logit_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/logit',
                                       output_ext=".nii.gz",
                                       output_postfix="seg",
                                       separate_folder=False)
            prob_map_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/prob_maps',
                                       output_ext=".nii.gz",
                                       output_postfix="seg",
                                       separate_folder=False)
            gt_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/ground_truth',
                                 output_ext=".nii.gz",
                                 output_postfix="seg",
                                 separate_folder=False)

            if multi_mod > 1:
                prepro_nat_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="nat_img",
                                             separate_folder=False)
                prepro_ven_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="ven_img",
                                             separate_folder=False)
                prepro_art_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="art_img",
                                             separate_folder=False)
                prepro_del_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="del_img",
                                             separate_folder=False)
                prepro_natart_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                                output_ext=".nii.gz",
                                                output_postfix="natart_img",
                                                separate_folder=False)
                prepro_artven_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                                output_ext=".nii.gz",
                                                output_postfix="artven_img",
                                                separate_folder=False)

            else:
                prepro_img_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="img",
                                             separate_folder=False)

            prob_post_trans = Compose([Activations(sigmoid=True)])
            for data in tqdm(dataloader):
                patient_id = os.path.basename(data['img_meta_dict']['filename_or_obj'][0])[:-7]

                if os.path.exists(os.path.join(output_dir, sub_dir, 'logit', patient_id + '_seg.nii.gz')):
                    continue
                images, labels, liver_label = data["img"].to(device), data["seg"].to(device), data["liver_label"].to(device)

                roi_size = config['patch_size']

                sw_batch_size = 1

                outputs = sliding_window_inference(images, roi_size, sw_batch_size, self._model, overlap=config['overlap'], 
                                                   mode=config['sliding_window_inf_mode'], sigma_scale=config['sliding_window_sigma_scale'],
                                                    padding_mode=config['padding_mode'])

                map_outputs = [prob_post_trans(i) for i in decollate_batch(outputs)]
                outputs_threshold = [self._post_trans(i) for i in decollate_batch(outputs)]
                
                for idx in range(len(outputs)):
                    outputs[idx] = outputs[idx]*liver_label[idx]
                    outputs_threshold[idx] = outputs_threshold[idx]*liver_label[idx]
                    map_outputs[idx] = map_outputs[idx]*liver_label[idx]

                    saver(outputs_threshold[idx])
                    logit_saver(outputs[idx])
                    prob_map_saver(map_outputs[idx])
                    gt_saver(labels[idx])
                    if multi_mod == 2:
                        prepro_art_saver(images[idx][0])
                        prepro_ven_saver(images[idx][1])
                    elif multi_mod == 3:
                        prepro_art_saver(images[idx][0])
                        prepro_ven_saver(images[idx][1])
                        prepro_del_saver(images[idx][2])
                    elif multi_mod == 4:
                        prepro_nat_saver(images[idx][0])
                        prepro_art_saver(images[idx][1])
                        prepro_ven_saver(images[idx][2])
                        prepro_del_saver(images[idx][3])
                    elif multi_mod == 5:
                        prepro_nat_saver(images[idx][0])
                        prepro_art_saver(images[idx][1])
                        prepro_ven_saver(images[idx][2])
                        prepro_del_saver(images[idx][3])
                        prepro_natart_saver(images[idx][4])
                    elif multi_mod == 6:
                        prepro_nat_saver(images[idx][0])
                        prepro_art_saver(images[idx][1])
                        prepro_ven_saver(images[idx][2])
                        prepro_del_saver(images[idx][3])
                        prepro_natart_saver(images[idx][4])
                        prepro_artven_saver(images[idx][5])
                    else:
                        prepro_img_saver(images[idx])

            return


    def test_time_predict(self, device, data_dict, output_dir, config, transform, sub_dir, num_examples, fold):
        self._model.eval()
        logit_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/logit_tta',
                            output_ext=".nii.gz",
                            output_postfix="seg",
                            separate_folder=False,
                            resample=True)
        saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preds_tta_' + str(num_examples) + '_' + str(config['tta_threshold']),
                          output_ext=".nii.gz",
                          output_postfix="seg",
                          separate_folder=False,
                          resample=True)
        prob_map_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/prob_maps_tta',
                          output_ext=".nii.gz",
                          output_postfix="seg",
                          separate_folder=False,
                          resample=True)

        tt_aug = CustomTestTimeAugmentation(transform,
                                            batch_size=1,
                                            patch_size=config['patch_size'],
                                            num_workers=0,
                                            inferrer_fn=self._model,
                                            device=device,
                                            sliding_window_overlap=config['overlap'],
                                            sliding_window_mode=config['sliding_window_inf_mode'],
                                            sliding_window_sigma_scale=config['sliding_window_sigma_scale'],

                                      )

        for patient_dict in data_dict:
            logit_saving_path = os.path.join(output_dir, sub_dir, 'logit_tta', os.path.basename(patient_dict['img'])[:-7] + '_seg.nii.gz')
            prob_map_saving_path = os.path.join(output_dir, sub_dir, 'prob_maps_tta', os.path.basename(patient_dict['img'])[:-7] + '_seg.nii.gz')
            threshold_map_saving_path = os.path.join(output_dir, sub_dir, 'preds_tta_' + str(num_examples) + '_' + str(config['tta_threshold']), 
                                                     os.path.basename(patient_dict['img'])[:-7] + '_seg.nii.gz')

            if os.path.exists(logit_saving_path): 
                if not os.path.exists(threshold_map_saving_path):
                    os.maskedirs(os.path.dirname(threshold_map_saving_path), exist_ok=True)
                    
                    post_trans = Compose([AsDiscrete(threshold=config['tta_threshold'])])
                    sitk_image = sitk.ReadImage(prob_map_saving_path)
                    image_ar = sitk.GetArrayFromImage(sitk_image)
                    mean = torch.tensor(image_ar).unsqueeze(0)
                    liver = sitk.ReadImage(patient_dict['liver'])
                    liver_ar = sitk.GetArrayFromImage(liver)
                    
                    output = post_trans(mean)
                    output_image = sitk.GetImageFromArray(output[0].cpu().numpy())
                    output_image = output_image * liver_ar
                    output_image.CopyInformation(sitk_image)
                    sitk.WriteImage(output_image, output_dir + f'/{sub_dir}/preds_tta_' + str(num_examples) + '_' + str(config['tta_threshold']) 
                                    + '/' + os.path.basename(patient_dict['img'][:-7] + '_seg.nii.gz'))

            else:
                post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=config['tta_threshold'])])
                post_prob_map = Compose([Activations(sigmoid=True)])
                mode, mean, std, vvc, output_embeddings_mean = tt_aug(patient_dict, num_examples=num_examples)
                prob_map_saver(post_prob_map(mean))
                logit_saver(mean)

                output = post_trans(mean)
                saver(output, patient_dict['img'])
         
    
    def calibration(self, logits, output_dir, experiment_config):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', default=0, type=int, help='index of used GPU')
        parser.add_argument('--model-name', default='LTS', type=str, help='model name: IBTS, LTS, TS')
        parser.add_argument('--epochs', default=200, type=int, help='max epochs')
        parser.add_argument('--batch-size', default=1, type=int, help='batch size')
        parser.add_argument('--lr', default=1e-5, type=float, help='inital learning rate')
        parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
        parser.add_argument('--save-per-epoch', default=1, type=int, help='number of epochs to save model.')
        args = parser.parse_args()
    
        dataset_name = os.path.basename(experiment_config['dataset'])
        calibration_mode_path = output_dir + f'/calibration_models'
        os.makedirs(calibration_mode_path, exist_ok=True)

        train_logits_list = logits[:int(0.9*len(logits))]
        val_logits_list   = logits[int(0.9*len(logits)):]

        nll_criterion = nn.CrossEntropyLoss()
        max_epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr

        tumor_experiment_name = 'LTS_CamVid'
        train_dataset = CALIBRATION(tumor_experiment_name, train_logits_list)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_dataset = CALIBRATION(tumor_experiment_name, val_logits_list)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        experiment_name = 'LTS_CamVid_max_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = LTS_CamVid_With_Image(args)

        calibration_model.weights_init()
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            calibration_model.cuda(args.gpu)
        else:
            calibration_model.cuda()

        optimizer = optim.Adam(calibration_model.parameters(), lr=lr)
        
        print("Computing Loss")
        val_loss = 0
        for val_image, val_logits, val_labels, val_preds, val_boundary, sub_id, file_path in val_dataloader:
            val_labels = val_labels.float().cuda(args.gpu)
            val_logits = val_logits.float().cuda(args.gpu)
            val_loss += nll_criterion(val_logits, val_labels).item()
        mean_val_loss = val_loss/len(val_dataloader)

        print('Before calibration - NLL: %.5f' % (mean_val_loss))
        calibration_model.train()
        patience = 0
        if not os.path.isfile(calibration_mode_path + experiment_name + '_checkpoint.pth.tar'):
            for epoch in range(max_epochs):
                for i, (train_image, train_logits, train_labels, train_preds, train_boundary, sub_id, file_path) in enumerate(train_dataloader):
                    global_step = epoch * len(train_dataloader) + (i + 1) * batch_size
                    train_image, train_logits, train_labels = train_image.cuda(args.gpu), train_logits.cuda(args.gpu), train_labels.float().cuda(args.gpu)
                    optimizer.zero_grad()
                    logits_calibrate = calibration_model(train_logits, train_image)
                    loss = nll_criterion(logits_calibrate.squeeze(0), train_labels)
                    loss.backward()
                    optimizer.step()
                    print("{} epoch, {} iter, training loss: {:.5f}".format(epoch, i + 1, loss.item()))

                with torch.set_grad_enabled(False):
                    tmp_loss = 0
                    for val_image, val_logits, val_labels, val_preds, val_boundary, sub_id, file_path in val_dataloader:
                        val_image, val_logits, val_labels = val_image.cuda(args.gpu), val_logits.cuda(args.gpu), val_labels.float().cuda(args.gpu)
                        logits_cali = calibration_model(val_logits, val_image)
                        tmp_loss += nll_criterion(logits_cali.squeeze(0), val_labels).item()
                    mean_tmp_loss = tmp_loss/len(val_dataloader)
                    print("{} epoch, {} iter, training loss: {:.5f}, val loss: {:.5f}".format(epoch, i+1, loss.item(), mean_tmp_loss))

                    if mean_tmp_loss < mean_val_loss:
                        patience = 0
                        if mean_val_loss - mean_tmp_loss < 0.03:
                            patience = 6
                            print('Early stopping!')
                            break
                        mean_val_loss = mean_tmp_loss
                        print('%d epoch, current lowest - NLL: %.5f' % (epoch, mean_val_loss))
                        torch.save(calibration_model.state_dict(), calibration_mode_path + '/' + experiment_name + '_params.pth.tar')
                        best_state = {'epoch': epoch,
                                    'state_dict': calibration_model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'best_score': mean_val_loss,
                                    'global_step': global_step
                                    }
                        os.makedirs(calibration_mode_path, exist_ok=True)
                        torch.save(best_state, calibration_mode_path + '/' + experiment_name + '_model_best.pth.tar')
                    else:
                        patience += 1
                    current_state = {'epoch': epoch,
                                    'state_dict': calibration_model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'best_score': mean_tmp_loss,
                                    'global_step': global_step
                                    }
                    torch.save(current_state, calibration_mode_path + '/' + experiment_name + '_checkpoint.pth.tar')
                if patience > 5:
                    print('Early stopping!')
                    break
        else:
            calibration_model.load_state_dict(torch.load(calibration_mode_path + '/' + experiment_name + '_model_best.pth.tar'))

        pred_dataset = CALIBRATION(tumor_experiment_name, val_logits_list + train_logits_list)
        pred_dataloader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        logit_type = os.path.basename(os.path.dirname(logits[0]))
        prob_post_trans = Compose([Activations(sigmoid=True)])
        calibration_model.eval()
        with torch.set_grad_enabled(False):
            for val_image, val_logits, val_labels, val_preds, val_boundary, sub_id, file_path in pred_dataloader:
                print(file_path[0])

                val_image, val_logits, val_labels = val_image.cuda(args.gpu), val_logits.cuda(args.gpu), val_labels.float().cuda(args.gpu)
                logits_cali = calibration_model(val_logits, val_image)
                
                prob_map = prob_post_trans(logits_cali)
                prob_map = np.squeeze(prob_map)
                output_prediction = np.squeeze(logits_cali)


                val_label = sitk.ReadImage(os.path.join(file_path[0], 'ground_truth', sub_id[0]))
                output_image = sitk.GetImageFromArray(output_prediction.cpu().numpy())
                output_image.CopyInformation(val_label)
                output_dir = os.path.join(file_path[0], 'calibrated_' + logit_type)
                os.makedirs(output_dir, exist_ok=True)
                sitk.WriteImage(output_image, os.path.join(output_dir, sub_id[0]))

                prob_map_image = sitk.GetImageFromArray(prob_map)
                prob_map_image.CopyInformation(val_label)
                output_dir = os.path.join(file_path[0], 'calibrated_prob_map_' + logit_type)
                os.makedirs(output_dir, exist_ok=True)
                sitk.WriteImage(prob_map_image, os.path.join(output_dir, sub_id[0]))
        return

    
class UNETTumorSegmentationTrainer():
    def __init__(self, config, model_path):
        super().__init__()
        self._model = monai.networks.nets.AttentionUnet(**config['model_params'])
        self._max_epochs = config['max_epochs']
        self._check_point_rate = config['check_point_rate']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        self._threshold = config['post_trans_threshold']
        self._learning_rate = float(config['learning_rate'])

        self._loss_function = monai.losses.TverskyLoss(sigmoid=True, include_background=True, alpha=config['alpha'], beta=config['beta'], reduction=config['loss_reduction'])
        self._loss_type = "TverskyLoss"
        self._optimizer = torch.optim.Adam(self._model.parameters(), np.float64(config['learning_rate']))
        self._dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self._post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=config['post_trans_threshold'])])
        if config['lr_scheduler'] == 'cosine':
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=config['max_epochs']/10, eta_min=np.float64(config['learning_rate']))
        else:
            self._scheduler = PolynomialLR(self._optimizer, total_iters=config['max_epochs'], power=0.9)
        self._start_epoch = 0
        if model_path != None and model_path != 'None':
            checkpoint = torch.load(model_path)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            if config["load_previous_model_params"]:
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self._start_epoch = checkpoint['epoch'] + 1

    def train_epochs(self, device, dataloader, config):
        self._model.train()
        epoch_loss = 0
        step = 0
        for batch_data in dataloader:
            step += 1

            inputs, labels, liver_label = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data["liver_label"].to(device)
            self._optimizer.zero_grad()
            outputs = self._model(inputs)

            outputs = outputs * liver_label

            loss = self._loss_function(outputs, labels)
            if config['loss_reduction'] == 'none':
                if 'LLD_MMRI' in config['dataset']:
                    sample_weight = torch.tensor([10 if lesion_type == 'Hepatocellular_carcinoma' else 1 for lesion_type in batch_data['lesion_type']]).to(device)
                elif '10_T1_dataset' in config['dataset']:
                    lesion_labels = [torch.max(batch_data['multi_labels'][sample, :, :, :, :]).astype(int) for sample in range(batch_data['multi_labels'].shape[0])]
                    HCC = [batch_data['lesion_type'][sample][lesion_labels[sample]-1] if lesion_labels[sample] != 0 else 0 for sample in range(batch_data['multi_labels'].shape[0])]
                    sample_weight = torch.tensor([10 if lesion == 1 else 1 for lesion in HCC]).to(device)
                else:
                    print('No sample weight as dataset was not retrieved')
                    break
                loss = loss * sample_weight
                loss = loss.mean()
            liver_label = batch_data["liver_label"].to(device)
            if np.sum(liver_label.ravel()) == 0:
                print()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        return epoch_loss

    def eval(self, device, dataloader, config, epoch):
        self._model.eval()
        with (torch.no_grad()):
            val_images = None
            val_labels = None
            val_outputs = None

            step = 0
            epoch_loss = 0
            wb_logs = ()
            for val_data in dataloader:
                step += 1
                val_images, val_labels, val_liver = val_data["img"].to(device), val_data["seg"].to(device), val_data["liver_label"].to(device)
                roi_size = config['patch_size']
                sw_batch_size = 8
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self._model, overlap=config['overlap'], 
                                                       mode=config['sliding_window_inf_mode'], sigma_scale=config['sliding_window_sigma_scale'])
                loss = self._loss_function(val_outputs, val_labels)
                epoch_loss += loss.item()

                val_outputs_post = [self._post_trans(i) for i in decollate_batch(val_outputs)]

                val_outputs_post = [i * val_liver for i in val_outputs_post]

                self._dice_metric(y_pred=val_outputs_post, y=val_labels)

                if step == 1:
                    if val_images.shape[1] > 1:
                        wb_logs = self.log_slices(val_images[0, 1, :, :, :].detach().cpu(), val_labels[0][0].detach().cpu(),
                                         val_outputs_post[0][0])
                    else:
                        wb_logs = self.log_slices(val_images[0, 0, :, :, :].detach().cpu(),
                                                  val_labels[0][0].detach().cpu(),
                                                  val_outputs_post[0][0])
                    os.makedirs(wandb.run.dir + '/val_preds', exist_ok=True)
                    img_saver = SaveImage(output_dir=wandb.run.dir + '/val_preds',
                                        output_ext=".nii.gz",
                                        output_postfix="img",
                                        separate_folder=False)
                    img_saver(val_images[0][0])
                    label_saver = SaveImage(output_dir=wandb.run.dir + '/val_preds',
                                        output_ext=".nii.gz",
                                        output_postfix="label",
                                        separate_folder=False)
                    label_saver(val_labels[0][0])
                    pred_saver = SaveImage(output_dir=wandb.run.dir + '/val_preds',
                                        output_ext=".nii.gz",
                                        output_postfix="pred_epoch_" + str(epoch),
                                        separate_folder=False)
                    pred_saver(val_outputs_post[0][0])
                    
            dice_metric_agg = self._dice_metric.aggregate().item()
            self._dice_metric.reset()

            epoch_loss /= step

            return (epoch_loss, dice_metric_agg, wb_logs)


    def plot_patches(self, patch_list_selection):
        weird_patch_logs = []
        for patch in range(len(patch_list_selection)):
            best_plane = np.argmax(
                [np.sum(patch_list_selection[patch][1][:, :, i]) for i in range(patch_list_selection[0][1].shape[2])])
            plt.imshow(patch_list_selection[patch][1][:, :, best_plane], cmap='gray', alpha=0.5)
            plt.imshow(patch_list_selection[patch][4][:, :, best_plane], cmap='gray', alpha=0.5)
            plt.imshow(patch_list_selection[patch][0][0, :, :, best_plane], cmap='gray', alpha=0.25)
            weird_patch_logs.append(wandb.Image(plt, caption=f"Patch: {patch}"))
            plt.close()
        return weird_patch_logs

    def log_slices(self, image, label, prediction):
        wandb_mask_logs = []
        wandb_img_logs = []
        wandb_prediction_logs = []

        for img_slice_no in range(image.shape[2]):
            img = image[:, :, img_slice_no]
            lbl = label[:, :, img_slice_no]
            pred = prediction[0, :, :, img_slice_no]

            wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {img_slice_no}"))
            wandb_mask_logs.append(wandb.Image(img, masks={"ground truth" :
                                                                  {"mask_data" : lbl,
                                                                   "class_labels" : {0: "background", 1: "mask"} }}))
            wandb_prediction_logs.append(wandb.Image(img, masks={"prediction":
                                                               {"mask_data": pred,
                                                                "class_labels": {0: "background", 1: "mask"}}}))
        return wandb_img_logs, wandb_mask_logs, wandb_prediction_logs

    def training(self, train_loader, val_loader, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.init(
            project=config["wandb_project_name"],
            name=config['run_name'],
            config=config,
            save_code=False,
            id=config['run_name'],
            mode="offline",
        )
        os.makedirs(os.path.join(wandb.run.dir, "code/training/unet"), exist_ok=True)
        config['saving_path_wandb'] = wandb.run.dir
        config['saving_name_wandb'] = os.path.basename(os.path.dirname(wandb.run.dir))
        with open(config['config_path'], 'r') as file:
            original_config = yaml.load(file, Loader=yaml.FullLoader)
        fold_name = "saving_name_wandb_fold_{}".format(config['fold'])
        
        original_config.update({fold_name: os.path.basename(os.path.dirname(wandb.run.dir))})
        with open(config['config_path'], 'w') as file:
            yaml.dump(original_config, file, default_flow_style=False, sort_keys=False)

        shutil.copyfile(os.getcwd() + '/models.py', os.path.join(wandb.run.dir, "code/training/unet/models.py"))
        shutil.copyfile(os.getcwd() + '/train_unet_tumor_finetuning.py', os.path.join(wandb.run.dir, "code/training/unet/train_unet_tumor_finetuning.py"))
        shutil.copyfile(os.getcwd() + '/train_unet_tumor.py', os.path.join(wandb.run.dir, "code/training/unet/train_unet_tumor.py"))
        shutil.copyfile(config['config_path'], os.path.join(wandb.run.dir, "code/training/unet/" + os.path.basename(config['config_path'])))

        wandb.config.update(config)

        val_interval = config['val_interval']
        max_epochs = config['max_epochs']
        best_metric = np.inf
        best_metric_epoch = 0
        epoch_loss_values = list()
        metric_values = list()
        patience = 0
        for epoch in range(self._start_epoch, max_epochs):
            start_time = time.time()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            epoch_loss = self.train_epochs(device, train_loader, config)
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                (val_epoch_loss, dice_metric_agg, val_wb_logs) = self.eval(device, val_loader, config, epoch)

                if (epoch + 1) % self._check_point_rate == 0:
                    torch.save({'model_state_dict': self._model.state_dict(),
                                     'optimizer_state_dict': self._optimizer.state_dict(),
                                     'scheduler_state_dict': self._scheduler.state_dict(),
                                     'epoch': epoch
                                    },
                               (os.path.join(wandb.run.dir, "last_checkpoint.pth.gz")))

                metric_values.append(epoch_loss)
                if best_metric > dice_metric_agg > 0:
                    best_metric = dice_metric_agg
                    best_metric_epoch = epoch + 1
                    torch.save({'model_state_dict': self._model.state_dict(),
                                    'optimizer_state_dict': self._optimizer.state_dict(),
                                    'scheduler_state_dict': self._scheduler.state_dict(),
                                    'epoch': epoch
                                    },
                               (os.path.join(wandb.run.dir, "best_metric_model.pth.gz")))
                    patience = 0

                    best_model_log_message = f"saved new best metric model at the {epoch +1}th epoch"
                    print(best_model_log_message)
                else:
                    patience += 1
                message1 = f"current epoch: {epoch + 1} current mean loss: {val_epoch_loss:.4f}"
                message2 = f"\nbest mean dice: {best_metric:.4f} "
                message3 = f"at epoch: {best_metric_epoch}"

                print(message1, message2, message3)

            end_time = time.time()
            epoch_time = end_time - start_time

            if (epoch + 1) % val_interval == 0:
                wandb.log({"Epoch time": epoch_time,
                           "train_loss": epoch_loss,
                           "val_dice_based": dice_metric_agg,
                           "val_loss": val_epoch_loss,
                           "best_dice_metric": best_metric,
                           "best_metric_epoch": best_metric_epoch,
                           "step": epoch,
                           "Segmentation mask": val_wb_logs[1],
                           "Prediction": val_wb_logs[2],
                           })

            else:
                wandb.log({"Epoch time": epoch_time,
                           "train_loss": epoch_loss,
                           "step": epoch,
                           })


            if patience == config['early_stopping']:
                break

            self._scheduler.step()
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    def predict(self, device, dataloader, output_dir, config, sub_dir, multi_mod, fold):
        self._model.eval()
        with torch.no_grad():
            images = None
            labels = None
            outputs = None

            saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preds_{config["post_trans_threshold"]}',
                              output_ext=".nii.gz",
                              output_postfix="seg",
                              separate_folder=False)

            prob_map_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/prob_maps',
                                       output_ext=".nii.gz",
                                       output_postfix="seg",
                                       separate_folder=False)
            gt_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/ground_truth',
                                 output_ext=".nii.gz",
                                 output_postfix="seg",
                                 separate_folder=False)

            if multi_mod > 1:
                prepro_nat_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="nat_img",
                                             separate_folder=False)
                prepro_ven_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="ven_img",
                                             separate_folder=False)
                prepro_art_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="art_img",
                                             separate_folder=False)
                prepro_del_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="del_img",
                                             separate_folder=False)
                prepro_natart_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                                output_ext=".nii.gz",
                                                output_postfix="natart_img",
                                                separate_folder=False)
                prepro_artven_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                                output_ext=".nii.gz",
                                                output_postfix="artven_img",
                                                separate_folder=False)

            else:
                prepro_img_saver = SaveImage(output_dir=output_dir + f'/{sub_dir}/preprocessed_img',
                                             output_ext=".nii.gz",
                                             output_postfix="img",
                                             separate_folder=False)

            prob_post_trans = Compose([Activations(sigmoid=True)])
            for data in dataloader:
                images, labels, liver_label = data["img"].to(device), data["seg"].to(device), data["liver_label"].to(device)
                roi_size = config['patch_size']

                sw_batch_size = 4

                outputs = sliding_window_inference(images, roi_size, sw_batch_size, self._model, overlap=config['overlap'], mode=config['sliding_window_inf_mode'], sigma_scale=config['sliding_window_sigma_scale'],
                                                   padding_mode='replicate')
                map_outputs = [prob_post_trans(i) for i in decollate_batch(outputs)]
                outputs = [self._post_trans(i) for i in decollate_batch(outputs)]
                outputs = [i * liver_label for i in outputs]

                for idx in range(len(outputs)):
                    saver(outputs[idx])
                    prob_map_saver(map_outputs[idx])
                    gt_saver(labels[idx])
                    if multi_mod == 2:
                        prepro_art_saver(images[idx][0])
                        prepro_ven_saver(images[idx][1])
                    elif multi_mod == 3:
                        prepro_art_saver(images[idx][0])
                        prepro_ven_saver(images[idx][1])
                        prepro_del_saver(images[idx][2])
                    elif multi_mod == 4:
                        prepro_nat_saver(images[idx][0])
                        prepro_art_saver(images[idx][1])
                        prepro_ven_saver(images[idx][2])
                        prepro_del_saver(images[idx][3])
                    elif multi_mod == 5:
                        prepro_nat_saver(images[idx][0])
                        prepro_art_saver(images[idx][1])
                        prepro_ven_saver(images[idx][2])
                        prepro_del_saver(images[idx][3])
                        prepro_natart_saver(images[idx][4])
                    elif multi_mod == 6:
                        prepro_nat_saver(images[idx][0])
                        prepro_art_saver(images[idx][1])
                        prepro_ven_saver(images[idx][2])
                        prepro_del_saver(images[idx][3])
                        prepro_natart_saver(images[idx][4])
                        prepro_artven_saver(images[idx][5])
                    else:
                        prepro_img_saver(images[idx])

            return