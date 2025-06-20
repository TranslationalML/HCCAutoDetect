import numpy as np
import os
import json
import SimpleITK as sitk
import cc3d
from collections import Counter

def main(predictions_dir, output_seg_derivative_path, margin=None, use_prob_map=False, foregrnd_threshold=0.5, backgrnd_threshold=0.5):
    os.makedirs(output_seg_derivative_path, exist_ok=True)
    patient_match_name = json.load(open(os.path.join(os.path.dirname(predictions_dir), '7_grouped_images_nnunet_preds_name_correspondance.json')))
    failed = []
    cases = [case for case in os.listdir(predictions_dir) if '.nii.gz' in case]
    for case in cases:
        print(case)
        patient_ID = os.path.basename(os.path.dirname(os.path.dirname((patient_match_name[case[:-7] + '_0000.nii.gz']))))
        file_name = os.path.basename(patient_match_name[case[:-7] + '_0000.nii.gz'])

        seg_cat_output = os.path.join(output_seg_derivative_path, patient_ID, 'dyn')
        os.makedirs(seg_cat_output, exist_ok=True)

        if len(os.listdir(seg_cat_output)) == 4:
            continue

        pred = os.path.join(predictions_dir, case)
        mask = sitk.ReadImage(pred)
        if use_prob_map:
            try:
                prob_map = np.load(pred.replace('.nii.gz', '.npz'))
            except:
                print('Fail', patient_ID)
                failed.append((patient_ID, pred))
                continue

            foreground_prob_map = prob_map['probabilities'][1]
            background_prob_map = prob_map['probabilities'][0]
            foreground_threshold = foregrnd_threshold
            background_threshold = backgrnd_threshold
            mask_ar = np.where(foreground_prob_map >= foreground_threshold, 1,
                                    np.where(background_prob_map > background_threshold, 0, 0))
        else:
            mask_ar = sitk.GetArrayFromImage(mask)

        cc_mask = cc3d.connected_components(mask_ar, delta=0.1)
        counter = Counter(cc_mask.ravel())
        del counter[0]
        biggest = counter.most_common(1)
        itensity_to_keep = biggest[0][0]
        cc_mask_liver = np.where(cc_mask == itensity_to_keep, 1, 0)

        liver_mask = sitk.GetImageFromArray(cc_mask_liver)
        liver_mask.SetSpacing(mask.GetSpacing())
        liver_mask.SetDirection(mask.GetDirection())
        liver_mask.SetOrigin(mask.GetOrigin())

        mask_saving_path = os.path.join(seg_cat_output, file_name)

        if margin:
            radius = margin
            bd_filter = sitk.BinaryDilateImageFilter()
            bd_filter.SetForegroundValue(1)
            bd_filter.SetKernelRadius(radius)
            dilated_mask = bd_filter.Execute(liver_mask)
            sitk.WriteImage(dilated_mask, mask_saving_path)
        sitk.WriteImage(liver_mask, mask_saving_path)


if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    predictions_dir = os.path.join(main_dir, "derivatives/7_grouped_images_for_nnunet_predictions/predictions")
    output_seg_derivative_path = os.path.join(main_dir, 'derivatives/8_liver_masks_corrected/liver_masks_no_margin_original_threshold')
    main(predictions_dir, output_seg_derivative_path, margin=0, use_prob_map=False, foregrnd_threshold=0.15, backgrnd_threshold=0.85)
