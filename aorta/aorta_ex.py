import os
import nibabel as nib
import numpy as np
from skimage.draw import disk
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from multiprocessing import Pool
from functools import partial


def compute_seg_boundary_inds(img):
    """
    Calculate the bbox of a seg.
    :param img:
    :return: xmin, xmax, ymin, ymax, zmin, zmax indexes
    """
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def AortaSegmentation(name, path):
    """
    This function gets the case name and path, find the location of the L1 in the ct, and search for circles in the CT
    near this position.
    :param name:
    :param path:
    :return: The VOD and DICE results of the comparison between gt to my results.
    """
    path_to_L1 = f'{path}/{name}_L1.nii.gz'
    L1_nifti = nib.load(path_to_L1)
    L1_data = L1_nifti.get_fdata()

    path_to_ct = f'{path}/{name}_CT.nii.gz'
    ct_nifti = nib.load(path_to_ct)
    ct_data = ct_nifti.get_fdata()

    path_to_gt = f'{path}/{name}_Aorta.nii.gz'
    gt_nifti = nib.load(path_to_gt)
    gt_data = gt_nifti.get_fdata()

    # # find the z range of the L1.
    # z = np.any(L1_data, axis=(0, 1))
    img = np.zeros(L1_data.shape, dtype=np.uint8)
    # zmin, zmax = np.where(z)[0][[0, -1]]

    x_min, x_max, y_min, y_max, zmin, zmax = compute_seg_boundary_inds(L1_data)
    # crop the ct
    y_min -= 50
    y_max -= 100
    x_max -= 50
    new_nifti_file = ct_data[x_min:x_max, y_min:y_max, zmin:zmax]
    # new_nifti = nib.Nifti1Image(new_nifti_file.astype(np.float), self.nifti_file.affine)
    # nib.save(new_nifti, f'{self.name}_Aorta.nii.gz')

    # run on every slice.
    for i in range(1, new_nifti_file.shape[2]):
        curr_slice = new_nifti_file[..., i]

        # find the edges
        edges = canny(curr_slice, sigma=3, low_threshold=0.2, high_threshold=0.8)

        # find circles in the slice
        hough_radii = np.arange(14, 19, 2)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = curr_slice
        for center_y, center_x, radius in zip(cy, cx, radii):
            # circy, circx = circle_perimeter(center_y, center_x, radius,
            #                                 shape=image.shape)
            # plot the circles if needed.
            circ = Circle((center_y, center_x), radius)
            ax.add_patch(circ)

            # making a disk on the segmentation.
            rr, cc = disk((center_y, center_x), radius)
            # todo fix the plots
            img[x_min + rr, y_min + cc, i + zmin] = 1

        ax.imshow(image, cmap=plt.cm.gray)
        plt.show()
    # save the aorta
    new_nifti = nib.Nifti1Image(img.astype(np.float), ct_nifti.affine)
    nib.save(new_nifti, f'{name}_Aorta.nii.gz')
    # compare to GT
    vod, dice = evaluateSegmentation(img, zmin, zmax, gt_data, name)
    return name, vod, dice


def evaluateSegmentation(est_seg, zmin, zmax, gt_data, name):
    """
    compute the dice and vod.
    :param est_seg: seg to compute
    :param zmin: lower z value of the L1.
    :param zmax: upper z value of the L1
    :param gt_data: Ground Truth seg file
    :param name: case name
    :return: tuple of VOD and DICE
    """

    seg1_data = gt_data[:, :, zmin:zmax]
    seg2_data = est_seg[:, :, zmin:zmax]
    intersection = np.logical_and(seg1_data, seg2_data)
    union = np.logical_or(seg1_data, seg2_data)

    dice_result = (2 * intersection.sum()) / (seg1_data.sum() + seg2_data.sum())
    vod_result = 1 - (intersection.sum() / union.sum())

    print(f'Case {name} VOD, DICE: {np.round(vod_result, 3)} , {np.round(dice_result, 3)}')
    return vod_result, dice_result


def main():
    path = '/cs/casmip/public/for_aviv/MedicalImageProcessing/Targil1_data'
    cases = []
    for file in os.listdir(path):
        if 'CT' in file:
            if file.split('_')[0] in ['Case5', 'HardCase1', 'HardCase2', 'HardCase3', 'HardCase4']:
                continue
            cases.append(file.split('_')[0])

    # Run all the cases with Groundtruth with multiprocessing
    with Pool() as pool:
        vod_dice = list(pool.map(partial(AortaSegmentation, path=path), cases))

    dice_dict = dict((name, dice) for name, vod, dice in vod_dice)
    vod_dict = dict((name, vod) for name, vod, dice in vod_dice)

    # plot the dice and vod
    lists = sorted(dice_dict.items())
    x, y = zip(*lists)
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o', label='DICE coefficient')
    lists = sorted(vod_dict.items())
    x, y = zip(*lists)
    ax.plot(x, y, '-o', label='VOD coefficient')

    plt.xlabel('Case name')
    plt.ylabel('Value')
    plt.title('Aorta segmentation vs. GroundTruth')

    ax.legend()
    plt.savefig(f'Aorta_coefficients_plot.png')
    plt.show()


if __name__ == '__main__':
    main()
