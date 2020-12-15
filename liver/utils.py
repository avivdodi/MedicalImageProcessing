import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


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


def check_orientation(ct_image, ct_arr):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'R':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
    if y != 'P':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
    if z != 'S':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
    return ct_arr


def evaluateSegmentation(est_seg, gt_data, name):  # zmin, zmax,
    """
    compute the dice and vod.
    :param est_seg: seg to compute
    :param zmin: lower z value of the L1.
    :param zmax: upper z value of the L1
    :param gt_data: Ground Truth seg file
    :param name: case name
    :return: tuple of VOD and DICE
    """

    seg1_data = gt_data  # [:, :, zmin:zmax]
    seg2_data = est_seg  # [:, :, zmin:zmax]
    intersection = np.logical_and(seg1_data, seg2_data)
    union = np.logical_or(seg1_data, seg2_data)

    dice_result = (2 * intersection.sum()) / (seg1_data.sum() + seg2_data.sum())
    vod_result = 1 - (intersection.sum() / union.sum())

    print(f'Case {name} VOD, DICE: {np.round(vod_result, 3)} , {np.round(dice_result, 3)}')
    return name, vod_result, dice_result


def compute_widest_slice(img):
    """
    Compute the z index of the widest sagittal view (y axis).
    :param img: The nii array.
    :return: Z key slice of the y widest axis (sagittal)
    """
    y_sum = np.sum(img, axis=1)
    z_index = np.argmax(y_sum, axis=0)
    return np.argmax(z_index)


def plot_results(vod_dice, title):
    """
    Plot the results based on the input lis of tuples
    :param vod_dice: list of tuple
    :param title: plot title
    :return: plot
    """
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
    plt.title(title)

    ax.legend()
    plt.savefig(f'{title}.png')
    plt.show()


if __name__ == '__main__':
    nii = nib.load('/cs/casmip/public/for_aviv/MedicalImageProcessing/liver/test.nii.gz')
    compute_widest_slice(nii.get_fdata())
