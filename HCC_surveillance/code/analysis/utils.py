import SimpleITK as sitk
import numpy as np

def contrast_check(image, label):
    """
    Calculate the contrast between the inside and around the tumor for nat, art, ven, and del.
    Parameters:
        image: 4D image with shape (C, D, H, W) where C is the number of channels
        label: 3D label image with shape (D, H, W) where each pixel is labeled as nat, art, ven, or del
    Returns:
        contrast_nat: contrast between nat inside and around the tumor
        contrast_art: contrast between art inside and around the tumor
        contrast_ven: contrast between ven inside and around the tumor
        contrast_del: contrast between del inside and around the tumor
    """
    inside_tumor = sitk.GetArrayFromImage(image) * sitk.GetArrayFromImage(label)
    dilated_tumor_large = sitk.BinaryDilate(label, (10, 10, 10))
    dilated_tumor_small = sitk.BinaryDilate(label, (3, 3, 3))
    around_tumor_large = sitk.GetArrayFromImage(image) * sitk.GetArrayFromImage(dilated_tumor_large)
    around_tumor_small = sitk.GetArrayFromImage(image) * sitk.GetArrayFromImage(dilated_tumor_small)
    around_tumor = around_tumor_large - around_tumor_small

    if sum(inside_tumor[1, :, :, :].ravel()) == 0:
        return 0, 0, 0, 0
    
    nat_perc_25_75_inside = np.percentile(inside_tumor[0, :, :, :][inside_tumor[0, :, :, :] != 0].ravel(), (25, 75))
    nat_perc_25_75_outside = np.percentile(around_tumor[0, :, :, :][around_tumor[0, :, :, :] != 0].ravel(), (25, 75))
    art_perc_25_75_inside = np.percentile(inside_tumor[1, :, :, :][inside_tumor[1, :, :, :] != 0].ravel(), (25, 75))
    art_perc_25_75_outside = np.percentile(around_tumor[1, :, :, :][around_tumor[1, :, :, :] != 0].ravel(), (25, 75))
    ven_perc_25_75_inside = np.percentile(inside_tumor[2, :, :, :][inside_tumor[2, :, :, :] != 0].ravel(), (25, 75))
    ven_perc_25_75_outside = np.percentile(around_tumor[2, :, :, :][around_tumor[2, :, :, :] != 0].ravel(), (25, 75))
    del_perc_25_75_inside = np.percentile(inside_tumor[2, :, :, :][inside_tumor[2, :, :, :] != 0].ravel(), (25, 75))
    del_perc_25_75_outside = np.percentile(around_tumor[2, :, :, :][around_tumor[2, :, :, :] != 0].ravel(), (25, 75))


    nat_inside_mean = np.mean(inside_tumor[0, :, :, :][(inside_tumor[0, :, :, :] > nat_perc_25_75_inside[0]) & (inside_tumor[0, :, :, :] < nat_perc_25_75_inside[1])])
    nat_outside_mean = np.mean(around_tumor[0, :, :, :][(around_tumor[0, :, :, :] > nat_perc_25_75_outside[0]) & (around_tumor[0, :, :, :] < nat_perc_25_75_outside[1])])
    art_inside_mean = np.mean(inside_tumor[1, :, :, :][(inside_tumor[1, :, :, :] > art_perc_25_75_inside[0]) & (inside_tumor[1, :, :, :] < art_perc_25_75_inside[1])])
    art_outside_mean = np.mean(around_tumor[1, :, :, :][(around_tumor[1, :, :, :] > art_perc_25_75_outside[0]) & (around_tumor[1, :, :, :] < art_perc_25_75_outside[1])])
    ven_inside_mean = np.mean(inside_tumor[2, :, :, :][(inside_tumor[2, :, :, :] > ven_perc_25_75_inside[0]) & (inside_tumor[2, :, :, :] < ven_perc_25_75_inside[1])])
    ven_outside_mean = np.mean(around_tumor[2, :, :, :][(around_tumor[2, :, :, :] > ven_perc_25_75_outside[0]) & (around_tumor[2, :, :, :] < ven_perc_25_75_outside[1])])
    del_inside_mean = np.mean(inside_tumor[2, :, :, :][(inside_tumor[2, :, :, :] > del_perc_25_75_inside[0]) & (inside_tumor[2, :, :, :] < del_perc_25_75_inside[1])])
    del_outside_mean = np.mean(around_tumor[2, :, :, :][(around_tumor[2, :, :, :] > del_perc_25_75_outside[0]) & (around_tumor[2, :, :, :] < del_perc_25_75_outside[1])])


    contrast_nat = nat_inside_mean/ nat_outside_mean
    contrast_art = art_inside_mean/ art_outside_mean
    contrast_ven = ven_inside_mean/ ven_outside_mean
    contrast_del = del_inside_mean/ del_outside_mean

    if any(np.isnan([contrast_nat, contrast_art, contrast_ven, contrast_del])):
        print('Nan values in contrast')
        print(contrast_nat, contrast_art, contrast_ven, contrast_del)
        return 0, 0, 0, 0


    return contrast_nat, contrast_art, contrast_ven, contrast_del

