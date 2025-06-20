import os
import SimpleITK as sitk
import numpy as np
import pandas as pd


def list_nii_gz_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii'):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    return file_paths


def extract_liver():
    cwd = os.getcwd()
    data_dir = os.path.dirname(os.path.dirname(cwd)) + "/sourcedata"
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(cwd)))
    data_dir = data_dir
    dataset_1 = data_dir + "/dataset_1"
    dataset_2 = data_dir + "/dataset_2"
    segmentations = data_dir + "/segmentations"
    dataset_1_files = list_nii_gz_files(dataset_1)
    dataset_2_files = list_nii_gz_files(dataset_2)
    all_files = sorted(dataset_1_files + dataset_2_files)
    segmentation_files = sorted(list_nii_gz_files(segmentations))

    output_folder = main_dir + "/LiTS/derivatives/Dataset_liver_extracted"
    os.makedirs(output_folder, exist_ok=True)
    output_image_folder = output_folder + "/images"
    os.makedirs(output_image_folder, exist_ok=True)
    output_labels_folder = output_folder + "/labels"
    os.makedirs(output_labels_folder, exist_ok=True)
    output_images_folder = output_folder + "/images"
    os.makedirs(output_images_folder, exist_ok=True)
    output_images_4D_folder = output_folder + "/images_4D"
    os.makedirs(output_images_4D_folder, exist_ok=True)
    output_liver_label_folder = output_folder + "/liver_labels"
    os.makedirs(output_liver_label_folder, exist_ok=True)
    output_region_based_label_folder = output_folder + "/region_based"
    os.makedirs(output_region_based_label_folder, exist_ok=True)
    output_multi_label_folder = output_folder + "/multi_labels"
    os.makedirs(output_multi_label_folder, exist_ok=True)

    patient_lesion_characteristics_list = []
    for file in all_files:
        print('Patient:', os.path.basename(file))

        seg_mask = os.path.dirname(os.path.dirname(os.path.dirname(file))) + "/segmentations/" + os.path.basename(file).replace("volume", "segmentation")
        sitk_image = sitk.ReadImage(file)
        sitk_seg = sitk.ReadImage(seg_mask)

        sitk_seg.SetDirection(sitk_image.GetDirection())
        sitk_seg.SetOrigin(sitk_image.GetOrigin())
        if sitk_image.GetSpacing() != sitk_seg.GetSpacing():
            sitk_seg.SetSpacing(sitk_image.GetSpacing())
        # liver label
        liver_ar = sitk.GetArrayFromImage(sitk_seg).astype(np.uint8)
        liver_ar[liver_ar >= 1] = 1
        liver_ar[liver_ar < 1] = 0

        liver_ar = liver_ar.astype(np.uint8)
        liver_image = sitk.GetImageFromArray(liver_ar)
        liver_image.CopyInformation(sitk_seg)

        connected_components = sitk.ConnectedComponent(liver_image)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(connected_components)

        largest_label = None
        largest_size = 0

        for label in stats.GetLabels():
            size = stats.GetNumberOfPixels(label)
            if size > largest_size:
                largest_size = size
                largest_label = label

        bounding_box = stats.GetBoundingBox(largest_label)
        liver_image_crop = sitk.RegionOfInterest(liver_image, bounding_box[3:], bounding_box[:3])
        sitk.WriteImage(liver_image_crop, output_liver_label_folder + "/" + os.path.basename(file) + ".gz")

        # Crop image and extract liver
        liver_crop_ar = sitk.GetArrayFromImage(liver_image_crop)
        image_crop = sitk.RegionOfInterest(sitk_image, bounding_box[3:], bounding_box[:3])
        image_crop_ar = sitk.GetArrayFromImage(image_crop)
        image_liver_ext_ar = image_crop_ar * liver_crop_ar

        liver_pixels = image_crop_ar[liver_crop_ar == 1]

        min_liver = np.min(liver_pixels)
        max_liver = np.max(liver_pixels)

        # image_liver_ext_ar
        normalized_image = image_liver_ext_ar.copy().astype(np.float32)
        normalized_image[liver_crop_ar == 1] = (normalized_image[liver_crop_ar == 1] - min_liver) / (max_liver - min_liver)
        normalized_image[liver_crop_ar == 0] = 0
        image_liver_ext = sitk.GetImageFromArray(normalized_image)
        image_liver_ext.CopyInformation(image_crop)

        output_file = output_image_folder + "/" + os.path.basename(file) + ".gz"
        sitk.WriteImage(image_liver_ext, output_file)


        # Image 4D
        vectorOfImages = sitk.VectorOfImage()
        for idx in range(4):
            vectorOfImages.push_back(image_liver_ext)

        image_4D = sitk.JoinSeries(vectorOfImages)
        output_file = output_images_4D_folder + "/" + os.path.basename(file) + ".gz"
        sitk.WriteImage(image_4D, output_file)

        # # Tumor mask
        tumor_mask = sitk.GetArrayFromImage(sitk_seg).astype(np.uint8)
        tumor_mask[tumor_mask == 1] = 0
        tumor_mask[tumor_mask == 2] = 1
        tumor_mask = sitk.GetImageFromArray(tumor_mask)
        tumor_mask.CopyInformation(sitk_seg)
        tumor_mask_crop = sitk.RegionOfInterest(tumor_mask, bounding_box[3:], bounding_box[:3])
        output_file = output_labels_folder + "/" + os.path.basename(file) + ".gz"
        sitk.WriteImage(tumor_mask_crop, output_file)

        # Region-based mask
        label_mask_crop = sitk.RegionOfInterest(sitk_seg, bounding_box[3:], bounding_box[:3])
        output_file = output_multi_label_folder + "/" + os.path.basename(file) + ".gz"
        sitk.WriteImage(label_mask_crop, output_file)

        # Multi-label mask
        connected_components = sitk.ConnectedComponent(tumor_mask_crop)
        sitk.WriteImage(connected_components, output_multi_label_folder + "/" + os.path.basename(file) + ".gz")

        # register lesion characteristics
        lesion_characteristics = {}
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(connected_components)
        num_labels = stats.GetNumberOfLabels()
        for label in range(1, num_labels + 1):
            if label not in lesion_characteristics:
                lesion_characteristics[label] = {}
            lesion_characteristics[label] = stats.GetPhysicalSize(label)

        df = pd.DataFrame(lesion_characteristics.items(), columns=['label', 'Connected components size'])
        df['ID'] = os.path.basename(file[:-4])
        df['Arterial'] = None
        df['Venous washout'] = None
        df['Venous capsule'] = None
        df['Delayed capsule'] = None
        df['Delayed washout'] = None
        df['LIRADS'] = None
        df['Lesion diameter'] = None
        df['Location'] = None

        patient_lesion_characteristics_list.append(df)
    all_tumor_df = pd.concat(patient_lesion_characteristics_list)
    all_tumor_df.to_csv(os.path.join(output_folder, 'tumors_characteristics.csv'), index=False)
    return


if __name__ == "__main__":
    extract_liver()
