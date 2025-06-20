import os
import SimpleITK as sitk

def main(segmentation_path, output_derivative_path):
    
    for patient_id in os.listdir(segmentation_path):
        patient_dir = os.path.join(segmentation_path, patient_id, 'dyn')
        output_patient_dir = os.path.join(output_derivative_path, patient_id, 'dyn')
        os.makedirs(output_patient_dir, exist_ok=True)
        for file in os.listdir(patient_dir):
            img = sitk.ReadImage(os.path.join(patient_dir, file))
            img_RAS = sitk.DICOMOrient(img, 'RAS')
            sitk.WriteImage(img_RAS, os.path.join(output_patient_dir, file))



if __name__ == "__main__":
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    segmentation_path = os.path.join(main_dir, 'derivatives/segmentations')
    output_derivative_path = os.path.join(main_dir, 'derivatives/reoriented_segmentations')
    os.makedirs(output_derivative_path, exist_ok=True)
    main(segmentation_path, output_derivative_path)