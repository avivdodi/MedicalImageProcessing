import nibabel as nib
import operator
from skimage.measure import label
import os
from multiprocessing import Pool
from skimage.morphology import binary_closing, remove_small_objects
from skimage.morphology import diamond, octahedron, ball
import numpy as np
import matplotlib.pylab as plt
from multiprocessing import Pool


class Segmentation:
    def __init__(self, path_to_nifti, i_max, name):
        self.path_to_nifti = path_to_nifti
        self.Imax = i_max
        self.Imin = None
        self.img_data = None
        self.name = name

    def SegmentationByTH(self, Imin, skeleton=False):
        """
        This func save save the segmentation based on threshold.
        :param Imin: the i-min th.
        :param skeleton: If true, the seg is saved with remove small objects and binary closing.
        :return: Min TH and number_of_components
        """
        nifti_file = nib.load(self.path_to_nifti)
        img_data = nifti_file.get_fdata()
        self.Imin = Imin
        img_data[img_data < Imin] = 0
        img_data[img_data >= Imin] = 1
        img_data[img_data > self.Imax] = 0
        self.img_data = img_data
        if not skeleton:
            try:
                new_nifti = nib.Nifti1Image(img_data, nifti_file.affine)
                nib.save(new_nifti, f'{self.name}_seg_<{Imin}>_<{self.Imax}>.nii.gz')
                number_of_components = label(img_data).max()
                return Imin, number_of_components
            except:
                print(f'Something went worng with {self.name}')
                return
        else:
            img_data = label(img_data)
            img_data = remove_small_objects(img_data, min_size=64, connectivity=24)
            img_data = binary_closing(img_data, selem=octahedron(2))
            new_nifti = nib.Nifti1Image(img_data.astype(np.float), nifti_file.affine)
            nib.save(new_nifti, f'{self.name}_SkeletonSegmentation.nii.gz')
            return

    def SkeletonTHFinder(self):
        """
        This function find the lower threshold based on the minimum number of labels.
        :param
        :return: int of the lower TH
        """''
        with Pool() as pool:
            number_of_components = list(pool.map(self.SegmentationByTH, range(150, 500, 14)))

        # save the number of labels for each TH
        th_components = dict((th, labels) for th, labels in number_of_components)

        # finds the lower number of labels val, and return the min key.
        i_min = min(th_components.items(), key=operator.itemgetter(1))[0]

        # save the final SkeletonSegmentation
        self.SegmentationByTH(i_min, skeleton=True)

        # plot the TH and number of labels
        lists = sorted(th_components.items())  # sorted by key, return a list of tuples
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y, 'o-')
        plt.xlabel('Min TH')
        plt.ylabel('Number of labels')
        plt.title(f'{self.name}')
        plt.savefig(f'{self.name}_plot.png')
        plt.show()
        return i_min


def main():
    path = '/cs/casmip/public/for_aviv/ex2/Targil1_data'
    for file in os.listdir(path):
        if 'CT' in file:
            name = file.split('.')[0]
            seg = Segmentation(f'{path}/{file}', i_max=1300, name=name)
            i_min = seg.SkeletonTHFinder()
            print(f'The min TH for case {file} is {i_min}')


if __name__ == '__main__':
    main()
