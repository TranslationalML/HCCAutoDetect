import os
import pandas as pd
import pydicom
from dcmrtstruct2nii import dcmrtstruct2nii
import SimpleITK as sitk
import numpy as np


def get_file_name(nifti_RAS, reoriented_dcm_path):
    for dir_, subdirs, files in os.walk(reoriented_dcm_path):
        for file in files:
            if file[-6:] == 'nii.gz':
                img = sitk.ReadImage(os.path.join(dir_, file))
                try:
                    img = sitk.DICOMOrient(img, 'RAS')
                except:
                    print('Can not read file', file)
                    pass

                try:
                    img_ar = sitk.GetArrayFromImage(img)
                    difference = np.sum(np.abs(nifti_RAS - img_ar))
                    if difference == 0:
                        return file
                except:
                    pass


def create_dirs(main_path, seg_data_path_annot_1, seg_data_path_annot_2, source_data_path, folder_output_path,
                convert_all_segmentations):
    participants = pd.read_csv((main_path + '/participants.tsv'), sep='\t')

    reoriented_dcm_path = os.path.join(main_path, 'derivatives/2_reoriented_images')
    for patient, patient_id in participants[['Patient', 'Sub-ID']].values:
        print(patient_id)
        source_patient_path = os.path.join(source_data_path, patient)
        seg_patient_path_annot_1 = os.path.join(seg_data_path_annot_1, patient)
        seg_patient_path_annot_2 = os.path.join(seg_data_path_annot_2, patient)
        DCM_dict = {}

        RTStruct_dict_annot_1 = {}
        RTStruct_dict_annot_2 = {}
        for dir, subdirs, files in os.walk(seg_patient_path_annot_1):
            for file in files:
                dcm_path = os.path.join(dir, file)
                dcm_file = pydicom.read_file(dcm_path)
                mod = dcm_file.Modality
                if mod == 'RTSTRUCT':
                    for ref_frames in dcm_file.ReferencedFrameOfReferenceSequence:
                        for ref_study in ref_frames.RTReferencedStudySequence:
                            for ref_series in ref_study.RTReferencedSeriesSequence:
                                RTStruct_dict_annot_1[ref_series.SeriesInstanceUID] = dcm_path

        for dir, subdirs, files in os.walk(seg_patient_path_annot_2):
            for file in files:
                dcm_path = os.path.join(dir, file)
                dcm_file = pydicom.read_file(dcm_path)
                mod = dcm_file.Modality
                if mod == 'RTSTRUCT':
                    for ref_frames in dcm_file.ReferencedFrameOfReferenceSequence:
                        for ref_study in ref_frames.RTReferencedStudySequence:
                            for ref_series in ref_study.RTReferencedSeriesSequence:
                                RTStruct_dict_annot_2[ref_series.SeriesInstanceUID] = dcm_path
        if not convert_all_segmentations and any([len(RTStruct_dict_annot_2.keys()) == 0, len(RTStruct_dict_annot_1.keys()) == 0]):
            continue
        else:
            patient_output_path = os.path.join(folder_output_path, patient_id)
            os.makedirs(patient_output_path, exist_ok=True)

        for dir, subdirs, files in os.walk(source_patient_path):
            for file in files:
                try:
                    dcm_path = os.path.join(dir, file)
                    dcm_file = pydicom.read_file(dcm_path)
                    mod = dcm_file.Modality
                    if mod == 'MR':
                        DCM_dict[dcm_file.SeriesInstanceUID] = dir
                        break
                    else:
                        continue
                except:
                    pass

        if not convert_all_segmentations:
            matched_RTStruct = [key_annot_1 for (key_annot_1, value_annot_1) in RTStruct_dict_annot_1.items() if
                                key_annot_1 in RTStruct_dict_annot_2.keys()]

            for RTStruct_UID in matched_RTStruct:
                rtstruct_path_annot_1 = RTStruct_dict_annot_1[RTStruct_UID]
                rtstruct_path_annot_2 = RTStruct_dict_annot_2[RTStruct_UID]
                dcm_ref_path = DCM_dict[RTStruct_UID]
                output_path = patient_output_path

                for annotator, RTSS in [('annotator_1', rtstruct_path_annot_1), ('annotator_2', rtstruct_path_annot_2)]:
                    dcmrtstruct2nii(RTSS, dcm_ref_path, output_path)

                    nifti_image_path = os.path.join(output_path, 'image.nii.gz')
                    nifti_img = sitk.ReadImage(nifti_image_path)
                    nifti_RAS = sitk.DICOMOrient(nifti_img, 'RAS')
                    nifti_RAS_ar = sitk.GetArrayFromImage(nifti_RAS)
                    file_name = get_file_name(nifti_RAS_ar, os.path.join(main_path, patient_id))
                    if 'T2' in file_name:
                        saving_path = output_path + '/anat'
                    elif 'T1' in file_name:
                        saving_path = output_path + '/dyn'
                    else:
                        print('No DICOM of reference found')
                        break
                    os.remove(nifti_image_path)

                    os.makedirs(saving_path, exist_ok=True)
                    files = os.listdir(output_path)
                    mask_files = [file_name for file_name in files if file_name[:4] == 'mask']
                    for seg_idx, seg in enumerate(mask_files):
                        file_path = os.path.join(output_path, seg)
                        segmentation = sitk.ReadImage(file_path)
                        segmentation_RAS = sitk.DICOMOrient(segmentation, 'RAS')

                        if 'WL' in seg:
                            seg_type = 'liver_seg'
                        else:
                            seg_type = 'lesion_seg'

                        seg_saving_name = file_name[:-7] + '_' + seg_type + '_' + str(seg_idx) + '_' + annotator + '.nii.gz'
                        sitk.WriteImage(segmentation_RAS, os.path.join(saving_path, seg_saving_name))

                        if os.path.isfile(file_path):
                            os.remove(file_path)

        if convert_all_segmentations:
            for annotator, RT_dict in [('annotator_1', RTStruct_dict_annot_1), ('annotator_2', RTStruct_dict_annot_2)]:
                for patient_seg_idx, (key_UID, dir_value) in enumerate(RT_dict.items()):
                    rtstruct_path = dir_value
                    try:
                        dcm_ref_path = DCM_dict[key_UID]
                    except:
                        print('DCM of reference not found')
                        continue
                    output_path = patient_output_path

                    dcmrtstruct2nii(rtstruct_path, dcm_ref_path, output_path)

                    nifti_image_path = os.path.join(output_path, 'image.nii.gz')
                    nifti_img = sitk.ReadImage(nifti_image_path)
                    nifti_RAS = sitk.DICOMOrient(nifti_img, 'RAS')
                    nifti_RAS_ar = sitk.GetArrayFromImage(nifti_RAS)
                    file_name = get_file_name(nifti_RAS_ar, os.path.join(reoriented_dcm_path, patient_id))
                    os.remove(nifti_image_path)

                    if not file_name:
                        if any([x in dcm_ref_path for x in ['TRACE', 'ADC', 'ep2d', 'diff']]):
                            dwi_nifti = os.path.join(main_dir, patient_id + '/dwi')
                            dwi_nifti_list = os.listdir(dwi_nifti)
                            trace_files = [file for file in dwi_nifti_list if 'TRACE' in file and '.nii.gz' in file]
                            if len(trace_files) == 1:
                                file_name = trace_files[0]
                            for trace in trace_files:
                                img = sitk.ReadImage(os.path.join(dwi_nifti, trace))
                                img_ar = sitk.GetArrayFromImage(img)
                                img_conc = np.concatenate(img_ar, axis=0)
                                if img_conc.shape == nifti_RAS_ar.shape:
                                    file_name = trace
                    if not file_name:
                        print('No matching nifti image found for reference dicom: ', dcm_ref_path)
                        files = os.listdir(output_path)
                        mask_files = [file_name for file_name in files if file_name[:4] == 'mask']
                        for seg_idx, seg in enumerate(mask_files):
                            file_path = os.path.join(output_path, seg)
                            os.remove(file_path)
                        continue

                    if 'T2' in file_name:
                        saving_path = output_path + '/anat'
                    elif 'T1' in file_name:
                        saving_path = output_path + '/dyn'
                    elif any([x in file_name for x in ['TRACE', 'ADC']]):
                        saving_path = output_path + '/dwi'
                    else:
                        print('Error: image category not found for reference dicom: ', dcm_ref_path)
                        files = os.listdir(output_path)
                        mask_files = [file_name for file_name in files if file_name[:4] == 'mask']
                        for seg_idx, seg in enumerate(mask_files):
                            file_path = os.path.join(output_path, seg)
                            os.remove(file_path)
                        continue
                    os.makedirs(saving_path, exist_ok=True)

                    files = os.listdir(output_path)
                    mask_files = [file_name for file_name in files if file_name[:4] == 'mask']
                    for seg_idx, seg in enumerate(mask_files):
                        file_path = os.path.join(output_path, seg)
                        segmentation = sitk.ReadImage(file_path)
                        segmentation_RAS = sitk.DICOMOrient(segmentation, 'RAS')
                        segmentation_RAS = sitk.Clamp(segmentation_RAS, upperBound=1)

                        if 'WL' in seg:
                            seg_type = 'liver_seg'
                        else:
                            seg_type = 'lesion_seg'

                        seg_saving_name = file_name[:-7] + seg[4:]
                        sitk.WriteImage(segmentation_RAS, os.path.join(saving_path, seg_saving_name))

                        if os.path.isfile(file_path):
                            os.remove(file_path)


if __name__ == '__main__':
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))

    folder_output_path = os.path.join(main_dir, 'derivatives/inter_agreement_all_segmentations_full')
    os.makedirs(folder_output_path, exist_ok=True)
    seg_data_path_1 = os.path.join(main_dir, 'derivatives/annotations_1')
    seg_data_path_2 = os.path.join(main_dir, 'derivatives/annotations_2')
    source_data_path = os.path.join(main_dir, 'sourcedata/all_patients_depersonalised/')
    create_dirs(main_dir, seg_data_path_1, seg_data_path_2, source_data_path,  folder_output_path,
                convert_all_segmentations=True)