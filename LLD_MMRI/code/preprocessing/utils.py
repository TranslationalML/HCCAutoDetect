import itertools
from functools import wraps
import os
import SimpleITK as sitk
import shutil


def resampleTTC(img_list, fixed_img, output_path, saving_name):
    """
    Resample arterial TTC images to match the fixed image's spacing, origin, and direction.
    Parameters:
        img_list: List of paths to arterial TTC images.
        fixed_img: Path to the fixed image for resampling.
        output_path: Directory where the resampled images will be saved.
        saving_name: Suffix for the saved resampled images.
    Returns:
        List of paths to the resampled arterial TTC images.
    """    
    arterial_TTC_resampled_path = []
    for art_path in img_list:
        art_img = sitk.ReadImage(art_path)
        resampled_img = sitk.Resample(art_img, fixed_img, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, 0, art_img.GetPixelID())
        os.makedirs(output_path + '/temp', exist_ok=True)
        saving_path = os.path.join(output_path + '/temp', os.path.basename(art_path)[:-7] + saving_name)
        sitk.WriteImage(resampled_img, saving_path)
        arterial_TTC_resampled_path.append(saving_path)
    return arterial_TTC_resampled_path


def getDynList(patient_path):
    """
    Get a list of dynamic MRI files from the specified patient directory.
    Parameters:
        patient_path: Path to the patient's directory containing dynamic MRI files.
    Returns:
        A list of paths to dynamic MRI files.
    """    
    dyn_list = []
    for dir_, subdirs, files in os.walk(patient_path):
        for file in files:
            if file[-6:] == 'nii.gz':
                img_category = os.path.basename(dir_)
                nii_path = os.path.join(dir_, file)
                if img_category == 'dyn':
                    dyn_list.append(nii_path)

    return dyn_list


def updatePopulation(existing_files, T1_TTC_path):
    """
    Update the list of existing files with the T1-TTC files if they are present.
    Parameters:
        existing_files: List of existing file paths.
        T1_TTC_path: List of T1-TTC file paths.
    Returns:
        A list of updated file paths, including T1-TTC files if they match the existing files.
    """
    update_pop_to_transform = []
    for file in existing_files:
        if 'TTC' in file:
            for TTC_path in T1_TTC_path:
                if os.path.basename(TTC_path) == os.path.basename(file):
                    update_pop_to_transform.append(TTC_path)
        else:
            update_pop_to_transform.append(file)
    return update_pop_to_transform

def slice_by_slice_decorator(func):
    """
    A function decorator which executes func on each 3D sub-volume and *in-place* pastes the results into the
    input image. The input image type and the output image type are required to be the same type.
    Parameters:
        func: A function which take a SimpleITK Image as it's first argument and returns an Image as results.
    Returns: 
        A decorated function.
    """
    iter_dim = 2

    @wraps(func)
    def slice_by_slice(image, shrink_factor):
        dim = image.GetDimension()
        extract_size = list(image.GetSize())
        extract_size[iter_dim:] = itertools.repeat(0,  dim-iter_dim)

        extract_index = [0] * dim
        paste_idx = [slice(None, None)] * dim

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(extract_size)

        for high_idx in itertools.product(*[range(s) for s in image.GetSize()[iter_dim:]]):
            extract_index[iter_dim:] = high_idx
            extractor.SetIndex(extract_index)

            paste_idx[iter_dim:] = high_idx
            shrinked_image = sitk.Shrink(image[paste_idx], [shrink_factor] * image[paste_idx].GetDimension())
            corrected_image = func.Execute(shrinked_image)
            log_bias_field = sitk.Cast(func.GetLogBiasFieldAsImage(image[paste_idx]), sitk.sitkFloat64)
            image[paste_idx] = image[paste_idx] / sitk.Exp(log_bias_field)

        return image

    return slice_by_slice


def rigid_transform(rigid_config, bspline_config, fix_img, fix_mask, mov_img, mov_mask, registration_output_path, save_trans_name):
    """
    Perform a rigid and B-spline registration using SimpleITK ElastixImageFilter.
    Parameters:
        rigid_config: Path to the parameter file for the rigid registration.
        bspline_config: Path to the parameter file for the B-spline registration.
        fix_img: Path to the fixed image.
        fix_mask: Path to the fixed mask (optional).
        mov_img: Path to the moving image.
        mov_mask: Path to the moving mask (optional).
        registration_output_path: Path to the output directory where the results will be saved.
    """
    parameter_map_1 = sitk.ReadParameterFile(rigid_config)
    parameter_map_2 = sitk.ReadParameterFile(bspline_config)
    fixed_mask = sitk.Cast(sitk.ReadImage(fix_mask), sitk.sitkUInt8)
    fixed_img = sitk.ReadImage(fix_img)
    moving_img = sitk.ReadImage(mov_img)
    moving_mask = sitk.Cast(sitk.ReadImage(mov_mask), sitk.sitkUInt8)
    moving_mask.SetSpacing(moving_img.GetSpacing())
    moving_mask.SetDirection(moving_img.GetDirection())
    moving_mask.SetOrigin(moving_img.GetOrigin())

    fixed_mask.SetSpacing(fixed_img.GetSpacing())
    fixed_mask.SetDirection(fixed_img.GetDirection())
    fixed_mask.SetOrigin(fixed_img.GetOrigin())
    moving_mask.SetSpacing(moving_img.GetSpacing())
    moving_mask.SetDirection(moving_img.GetDirection())
    moving_mask.SetOrigin(moving_img.GetOrigin())

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_img)
    elastixImageFilter.SetFixedMask(fixed_mask)
    elastixImageFilter.SetMovingImage(moving_img)
    elastixImageFilter.SetMovingMask(moving_mask)
    elastixImageFilter.SetParameterMap(parameter_map_1)
    elastixImageFilter.AddParameterMap(parameter_map_2)
    elastixImageFilter.Execute()
    for txt_file in ['IterationInfo.0.R0.txt', 'IterationInfo.0.R1.txt', 'IterationInfo.1.R0.txt',
                     'TransformParameters.0.txt', 'TransformParameters.1.txt']:
        shutil.copyfile(os.path.join(os.getcwd(), txt_file),
                        os.path.join(registration_output_path,
                                     'transforms/' + txt_file[:-4]) + save_trans_name + '.txt')
        os.remove(os.path.join(os.getcwd(), txt_file))
