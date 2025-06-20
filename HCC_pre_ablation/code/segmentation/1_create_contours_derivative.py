import os
import pandas as pd
import pydicom
from dcmrtstruct2nii import dcmrtstruct2nii
import SimpleITK as sitk
import numpy as np
import re


def get_file_name(nifti_RAS, reoriented_dcm_path):
    for dir_, subdirs, files in os.walk(reoriented_dcm_path):
        for file in files:
            if file[-6:] == 'nii.gz':
                img = sitk.ReadImage(os.path.join(dir_, file))
                img_ar = sitk.GetArrayFromImage(img)
                min_val = np.min(img_ar)
                max_val = np.max(img_ar)
                img_ar = (img_ar - min_val) / (max_val - min_val)
                try:
                    difference = np.sum(np.abs(nifti_RAS - img_ar))
                    if difference < 1:
                        img_category = os.path.basename(dir_)
                        return file, img_category
                except:
                    pass

    return None, None

def search_data(patient_path, modality):
    dict_ = {}
    for dir, subdirs, files in os.walk(patient_path):
        for file in files:
            try:
                dcm_path = os.path.join(dir, file)
                dcm_file = pydicom.read_file(dcm_path)
                mod = dcm_file.Modality
                if mod == 'RTSTRUCT' and mod == modality:
                    for ref_frames in dcm_file.ReferencedFrameOfReferenceSequence:
                        for ref_study in ref_frames.RTReferencedStudySequence:
                            for ref_series in ref_study.RTReferencedSeriesSequence:
                                if ref_series.SeriesInstanceUID not in dict_:
                                    dict_[ref_series.SeriesInstanceUID] = dcm_path

                    break
                if mod == 'MR' and modality == mod:
                    if dcm_file.SeriesInstanceUID not in dict_:
                        if len(os.listdir(dir)) > 1:
                            dict_[dcm_file.SeriesInstanceUID] = dir
                        else:
                            print('Too few files (len: ', len(os.listdir(dir)), ')')
                    else:
                        print('Multiple MR found for the same series')
                    break
                else:
                    continue
            except:
                print('Error in reading file: ', dcm_path)
                break
    return dict_

def create_dirs():
    cwd = os.getcwd()
    main_path = os.path.dirname(os.path.dirname(cwd))
    participants = pd.read_csv((main_path + '/participants_with_session_depersonalised.tsv'), sep='\t', dtype=str)
    segmentation_path = os.path.join(main_path, 'sourcedata/segmentation depersonalised/')
    dcm_path = os.path.join(main_path, 'sourcedata/all_patients_all_dates_depersonalised/')

    miss_segmentation = []
    miss_RTSS = []
    output_derivative_path = os.path.join(main_path, 'derivatives/tumor_segmentations')
    reoriented_dcm_path = os.path.join(main_path, 'derivatives/1_reoriented_images')
    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:
        print(patient_id)

        session_selected = participants['session'][participants['Sub-ID'] == patient_id].values[0]
        patient_dcm_path = os.path.join(dcm_path, patient)
        patient_rtss_path = os.path.join(segmentation_path, patient)

        sessions = os.listdir(patient_rtss_path)
        session_full_name = [session for session in sessions if session.startswith(session_selected)][0]
        RTStruct_dict = search_data(os.path.join(patient_rtss_path, session_full_name), 'RTSTRUCT')

        DCM_dict = search_data(os.path.join(patient_dcm_path), 'MR')

        for idx, (key_UID, dir_value) in enumerate(RTStruct_dict.items()):
            rtstruct_path = dir_value
            try:
                source_dcm_path = DCM_dict[key_UID]
            except:
                print('No DICOM of reference found for RTSS: ', rtstruct_path)
                miss_RTSS.append({patient_id: ("DICOM ref not found", rtstruct_path)})
                continue
            patient_derivative_path = os.path.join(output_derivative_path, patient_id)
            os.makedirs(patient_derivative_path, exist_ok=True)
            output_path = patient_derivative_path

            try:
                dcmrtstruct2nii(rtstruct_path, source_dcm_path, output_path)
            except:
                print('Error in conversion of RTSTRUCT to NIFTI for patient: ', patient_id, ', path: ', rtstruct_path)
                continue

            image_path = os.path.join(output_path, 'image.nii.gz')
            nifti_img = sitk.ReadImage(image_path, sitk.sitkFloat32)
            nifti_RAS = sitk.DICOMOrient(nifti_img, 'RAS')
            nifti_RAS_ar = sitk.GetArrayFromImage(nifti_RAS)
            min_val = np.min(nifti_RAS_ar)
            max_val = np.max(nifti_RAS_ar)
            nifti_RAS_ar = (nifti_RAS_ar - min_val) / (max_val - min_val)
            file_name, img_category = get_file_name(nifti_RAS_ar, os.path.join(reoriented_dcm_path, patient_id))

            if file_name is None:
                print('DICOM series of reference not found for patient: ', patient_id, ', path: ', source_dcm_path)
                os.remove(image_path)
                miss_segmentation.append({patient_id: ("Volume of ref not found", os.path.basename(source_dcm_path))})

                files = os.listdir(output_path)
                unchecked_files = [file_name for file_name in files if file_name[:4] == 'mask']
                [os.remove(os.path.join(output_path, file_name)) for file_name in unchecked_files if os.path.isfile(os.path.join(output_path, file_name))]
                continue
            os.remove(image_path)

            files = os.listdir(output_path)
            unchecked_files = [file_name for file_name in files if file_name[:4] == 'mask']
            for idx, seg in enumerate(unchecked_files):
                file_path = os.path.join(output_path, seg)
                segmentation = sitk.ReadImage(file_path)
                segmentation_RAS = sitk.DICOMOrient(segmentation, 'RAS')
                segmentation_RAS = sitk.Clamp(segmentation_RAS, upperBound=1)

                mint_category = seg[5:8]


                if any([x in seg for x in ['Liver', 'liver']]):
                    mint_category = mint_category + '_liver'
                    saving_name = file_name[:-7] + '_' + mint_category + '.nii.gz'
                else:
                    pattern = r'L[1-5]'
                    match = re.search(pattern, seg)
                    if match is not None:
                        lesion = match.group()
                    else:
                        lesion = 'Unkown'
                    mint_category = mint_category
                    saving_name = (file_name[:-7] + '_' + mint_category + '_' + lesion + '.nii.gz')
                try:
                    output_cat_path = os.path.join(output_path, img_category)
                    os.makedirs(output_cat_path, exist_ok=True)
                    sitk.WriteImage(segmentation_RAS, os.path.join(output_cat_path, saving_name))
                except:
                    print('Image not converted')
                    miss_segmentation.append({patient_id: ("Image not converted", os.path.basename(seg))})
                if os.path.isfile(file_path):
                    os.remove(file_path)

    # convert list of dict to dict
    full_dict = {}
    for elem in miss_segmentation:
        key = list(elem.keys())[0]
        if key not in full_dict:
            full_dict[key] = []
        full_dict[key].append(elem[key])

    #load prvious missed segmentations
    import json
    if os.path.isfile(output_derivative_path + '/missed_segmentations.json'):
        with open(output_derivative_path + '/missed_segmentations.json', 'r') as f:
            previous_missed_segmentations = json.load(f)
        for key, value in previous_missed_segmentations.items():
            if key not in full_dict:
                full_dict[key] = value

    # sort dict by key
    full_dict = dict(sorted(full_dict.items()))
    import json
    with open(output_derivative_path + '/missed_segmentations_3.json', 'w') as f:
        json.dump(full_dict, f, indent=4)

if __name__ == '__main__':
    create_dirs()

