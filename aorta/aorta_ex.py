import nibabel as nib
import operator
from skimage.measure import label
import os
from multiprocessing import Pool
from skimage.morphology import binary_closing, remove_small_objects
from skimage.morphology import diamond, octahedron, ball
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from matplotlib.patches import Circle
from skimage.draw import disk



def compute_seg_boundary_inds(img):
    """
    returns the segmentation bounding box indices.
    Assumes that seg is *not* all zeros
    :param seg:
    :param return_mid:
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
    def __init__(self, path_to_nifti, path2L1, name):
        self.nifti_file = nib.load(path_to_nifti)
        self.L1_nifti = nib.load(path2L1)
        self.img_data = None
        self.name = name

    def AortaSegmentation(self, nifty_file, L1_seg_nifti_file):
        # find the z range of the L1.
        z = np.any(self.L1_nifti.get_fdata(), axis=(0, 1))
        img = np.zeros(self.L1_nifti.get_fdata().shape, dtype=np.uint8)
        zmin, zmax = np.where(z)[0][[0, -1]]

        x_min, x_max, y_min, y_max, zmin, zmax = compute_seg_boundary_inds(self.L1_nifti.get_fdata())
        print(x_min)
        print(x_max)
        print(self.L1_nifti.get_fdata().shape)
        # crop the ct
        y_min -= 50
        y_max -=50
        x_max -=50
        nifty_file = self.nifti_file.get_fdata()[x_min:x_max, y_min:y_max, zmin:zmax]
        # # todo serch for connected circle ?
        # new_nifti = nib.Nifti1Image(nifty_file.astype(np.float), self.nifti_file.affine)
        # nib.save(new_nifti, f'{self.name}_Aorta.nii.gz')
        for i in range(1, nifty_file.shape[2]):
            curr_slice = nifty_file[..., i]

            edges = canny(curr_slice, sigma=3, low_threshold=0.2, high_threshold=0.8)
            hough_radii = np.arange(15, 18, 2)
            hough_res = hough_circle(edges, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                       total_num_peaks=3)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
            image = curr_slice
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=image.shape)

                circ = Circle((center_y, center_x), radius)
                ax.add_patch(circ)

                # center_x =
                # center_y =
                rr, cc = disk((center_y, center_x), radius)
                # # todo fix the places
                x = x_max - x_min
                y = y_max - y_min
                print(x)
                print(rr.sum())
                img[rr, cc, i+zmin] = 1

            ax.imshow(image, cmap=plt.cm.gray)
            plt.show()
        new_nifti = nib.Nifti1Image(img.astype(np.float), self.nifti_file.affine)
        nib.save(new_nifti, f'test.nii.gz')


    def evaluateSegmentation(self, GT_seg, est_seg):
        # compute the dice
        seg1_data = GT_seg.get_fdata()
        seg2_data = est_seg.get_fdata()
        seg1_2_and = np.logical_and(seg1_data, seg2_data)
        DICE_result = (2 * seg1_2_and.sum()) / (seg1_data.sum() + seg2_data.sum())

        # compute the vod

        return (VOD_result, DICE_result)


def main():
    path = '/cs/casmip/public/for_aviv/ex2/Targil1_data'
    file = 'Case1_CT.nii.gz'
    # for file in os.listdir(path):
    #     if 'CT' in file:
    name = file.split('.')[0]
    seg = Segmentation(f'{path}/{file}', f'{path}/Case1_L1.nii.gz', file)
    seg.AortaSegmentation(f'{path}/{file}', f'{path}/Case1_L1.nii.gz')

    # print(f'The min TH for case {file} is {i_min}')

    # plot the graph


if __name__ == '__main__':
    main()
