import os
import nibabel as nib
import numpy as np
from skimage.draw import disk
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def compute_seg_boundary_inds(img):
    """
    Calculate the bbox of a seg.
    :param img:
    :return:
    """
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


class Segmentation:
    def __init__(self, path_to_nifti, path_to_L1, path_to_gt, name):
        self.nifti_file = nib.load(path_to_nifti)
        self.L1_nifti = nib.load(path_to_L1)
        self.seg_gt = nib.load(path_to_gt)
        self.img_data = None
        self.name = name

    def AortaSegmentation(self, nifty_file, L1_seg_nifti_file):
        # find the z range of the L1.
        z = np.any(self.L1_nifti.get_fdata(), axis=(0, 1))
        img = np.zeros(self.L1_nifti.get_fdata().shape, dtype=np.uint8)
        zmin, zmax = np.where(z)[0][[0, -1]]

        x_min, x_max, y_min, y_max, zmin, zmax = compute_seg_boundary_inds(self.L1_nifti.get_fdata())
        # crop the ct
        y_min -= 50
        y_max -= 100
        x_max -= 50
        nifty_file = self.nifti_file.get_fdata()[x_min:x_max, y_min:y_max, zmin:zmax]
        # new_nifti = nib.Nifti1Image(nifty_file.astype(np.float), self.nifti_file.affine)
        # nib.save(new_nifti, f'{self.name}_Aorta.nii.gz')
        for i in range(1, nifty_file.shape[2]):
            curr_slice = nifty_file[..., i]

            edges = canny(curr_slice, sigma=3, low_threshold=0.2, high_threshold=0.8)
            hough_radii = np.arange(14, 19, 2)
            hough_res = hough_circle(edges, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=1)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            image = curr_slice
            for center_y, center_x, radius in zip(cy, cx, radii):
                # circy, circx = circle_perimeter(center_y, center_x, radius,
                #                                 shape=image.shape)

                circ = Circle((center_y, center_x), radius)
                ax.add_patch(circ)

                rr, cc = disk((center_y, center_x), radius)
                # todo fix the plots
                img[x_min + rr, y_min + cc, i + zmin] = 1

            ax.imshow(image, cmap=plt.cm.gray)
            plt.show()
        new_nifti = nib.Nifti1Image(img.astype(np.float), self.nifti_file.affine)
        nib.save(new_nifti, f'{self.name}_Aorta.nii.gz')
        # compare to GT
        vod, dice = self.evaluateSegmentation(img, zmin, zmax)
        return vod, dice

    def evaluateSegmentation(self, est_seg, zmin, zmax):
        # compute the dice
        seg1_data = self.seg_gt.get_fdata()[:, :, zmin:zmax]
        seg2_data = est_seg[:, :, zmin:zmax]
        intersection = np.logical_and(seg1_data, seg2_data)
        union = union = np.logical_or(seg1_data, seg2_data)

        DICE_result = (2 * intersection.sum()) / (seg1_data.sum() + seg2_data.sum())
        VOD_result = 1 - (intersection.sum() / union.sum())
        # compute the vod
        print(f'Case {self.name} VOD, DICE: {np.round(VOD_result, 3)} , {np.round(DICE_result, 3)}')
        return VOD_result, DICE_result


def main():
    path = '/cs/casmip/public/for_aviv/MedicalImageProcessing/Targil1_data'
    # file = 'Case1_CT.nii.gz'
    dice_dict = {}
    vod_dict = {}
    for file in os.listdir(path):
        if 'CT' in file:
            if file.split('_')[0] in ['Case5', 'HardCase1', 'HardCase2', 'HardCase3', 'HardCase4']:
                continue
            name = file.split('_')[0]
            seg = Segmentation(f'{path}/{file}', f'{path}/{name}_L1.nii.gz', f'{path}/{name}_Aorta.nii.gz', name)
            vod, dice = seg.AortaSegmentation(f'{path}/{file}', f'{path}/{name}_L1.nii.gz')
            dice_dict[name] = dice
            vod_dict[name] = vod

    # plot the dice and vod
    lists = sorted(dice_dict.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    fig, ax = plt.subplots()
    ax.plot(x, y, '-o', label='DICE coefficient')
    lists = sorted(vod_dict.items())
    x, y = zip(*lists)
    ax.plot(x, y, '-o', label='VOD coefficient')


    # plt.plot(x, y, 'o-')
    plt.xlabel('Case name')
    plt.ylabel('Value')
    # plt.title(f'{name}')

    ax.legend()
    plt.savefig(f'Aorta_coefficients_plot.png')
    plt.show()

    # plot the graph


if __name__ == '__main__':
    main()
