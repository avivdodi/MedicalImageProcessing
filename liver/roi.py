import os
import nibabel as nib
import numpy as np
from skimage.morphology import label, remove_small_objects, convex_hull_image, convex_hull_object, ball, dilation
import utils
from skimage.transform import pyramid_gaussian


class Segmentation:
    def __init__(self, path_to_scan, name):
        self.nii_scan = nib.load(path_to_scan)
        self.name = name
        self.ct_arr = utils.check_orientation(self.nii_scan, self.nii_scan.get_fdata())
        new_nifti = nib.Nifti1Image(self.ct_arr.astype(np.float), self.nii_scan.affine)
        nib.save(new_nifti, f'{name}-new_scan.nii.gz')

    def IsolateBody(self):
        """
        Body segmentation of all the body, without air.
        :param ct_scan:
        :return:Body segmentation
        """
        ct_arr = self.ct_arr
        body_seg = np.zeros(ct_arr.shape, dtype=np.uint8)

        # filter th between -500 to 2000
        body_seg[(ct_arr > -500) & (ct_arr < 2000)] = 1
        body_seg[(ct_arr <= -500) | (ct_arr >= 2000)] = 0

        # clean noise
        body_seg = label(body_seg)
        body_seg = remove_small_objects(body_seg, min_size=200, connectivity=24)

        # compute the largest connected component
        body_seg = label(body_seg)
        assert (body_seg.max() != 0)  # assume at least 1 CC
        body_seg = body_seg == np.argmax(np.bincount(body_seg.flat)[1:]) + 1

        new_nifti = nib.Nifti1Image(body_seg.astype(np.float), self.nii_scan.affine)
        nib.save(new_nifti, f'{self.name}-body.nii.gz')

        return body_seg

    def IsolateBS(self, body_seg):
        """
        Isolate the lungs from the body
        :param body_seg:
        :return: lungs segmentation, inferior slice, widest slice of the lungs (sagittal view)
        """

        # find the seg of air
        lungs_seg = 1 - body_seg

        # remove the air outside the body (upper)
        lungs_seg = label(lungs_seg)
        air = lungs_seg == np.argmax(np.bincount(lungs_seg.flat)[1:]) + 1
        lungs_seg -= air

        # remove the air outside the body (lower)
        lungs_seg = label(lungs_seg)
        air = lungs_seg == np.argmax(np.bincount(lungs_seg.flat)[1:]) + 1
        lungs_seg -= air

        # take only the lungs
        lungs_seg = label(lungs_seg)
        lungs_seg = lungs_seg == np.argmax(np.bincount(lungs_seg.flat)[1:]) + 1

        new_nifti = nib.Nifti1Image(lungs_seg.astype(np.float), self.nii_scan.affine)
        nib.save(new_nifti, f'{self.name}-lungs.nii.gz')

        xmin, xmax, ymin, ymax, zmin, zmax = utils.compute_seg_boundary_inds(lungs_seg)

        # inferior slice of lungs.
        bb = zmin
        # widest slice of lungs among y axis (sagittal view)
        cc = utils.compute_widest_slice(lungs_seg)

        return lungs_seg, cc, bb

    def ThreeDBand(self, body_seg, lungs_seg, bb, cc):
        # todo fix the function.
        """

        :param body_seg:
        :param lungs_seg:
        :param bb:
        :param cc:
        :return:
        """
        band = body_seg[:, :, bb:cc]
        band = np.zeros(body_seg.shape, dtype=np.uint8)
        for i in range(1, body_seg.shape[2]):
            if bb <= i <= cc:
                band[:, :, i] = body_seg[:, :, i]

        # aff = nib.load('/cs/casmip/public/for_aviv/MedicalImageProcessing/liver/test.nii.gz').affine
        # new_nifti = nib.Nifti1Image(band.astype(np.float), aff)
        # nib.save(new_nifti, f'band.nii.gz')
        return band

    def MergedROI(self, ct_scan, aorta_arr):
        # todo fix the function
        ct_arr = ct_scan.get_fdata()
        img = np.zeros(ct_arr.shape, dtype=np.uint8)

        for i in range(1, ct_arr.shape[2]):
            curr_slice = ct_arr[..., i]

        # aorta gt
        # spine roi

        filename = ''
        new_nifti = nib.Nifti1Image(band.astype(np.float), ct_scan.affine)
        nib.save(new_nifti, f'{filename}_ROI.nii.gz')

        # return roi

    def liver_roi(self, body_seg, aorta_nii):
        """
        Making an initial roi by finding the middle of the aorta, and draw a cube on the right side of the aorta.
        :param body_seg: Body segmentation
        :param aorta_nii: Aorta segmentation
        :return: ROI image
        """
        ct_arr = self.ct_arr
        aorta_arr = aorta_nii.get_fdata()
        aorta_arr = utils.check_orientation(aorta_nii, aorta_arr)

        img = np.zeros(ct_arr.shape, dtype=np.uint8)

        xmin, xmax, ymin, ymax, zmin, zmax = utils.compute_seg_boundary_inds(aorta_arr)

        # middle of the aorta
        z = zmax - zmin
        x = xmax - xmin + 200
        y = ymax - ymin

        # make a cube in the middle of the liver (approx)
        img[x + 100:x + 150, y + 50:y + 100, z - 15:z + 5] = 1

        new_nifti = nib.Nifti1Image(img.astype(np.float), aorta_nii.affine)
        nib.save(new_nifti, f'{self.name}-roi.nii.gz')

        return img

    def findSeeds(self, roi, seeds_num=200):
        """
        Find sample seeds from roi on ct_scan, after randomized choosing and neighbor averaging.
        :param ct_scan: The scan
        :param roi: The segmentation to search for seeds.
        :param seeds_num: number of seeds.
        :return: seeds list from the roi on the scan.
        """
        seeds_list = []
        ct_arr = self.ct_arr

        # Take the second array after the downscale.
        ct_arr = tuple(pyramid_gaussian(ct_arr, downscale=2, multichannel=True))[1]
        roi = tuple(pyramid_gaussian(roi.astype(float), downscale=2, multichannel=True))[1]

        # find random points
        points = np.argwhere(roi)
        np.random.shuffle(points)

        # take the HU value in the random points.
        for ind in points[0:seeds_num]:
            # seeds_list.append(ct_arr[ind[0], ind[1], ind[2]])
            seeds_list.append(ind)
        return seeds_list

    def multipleSeedsRG(self, roi):
        """
        Seeded Region Growing implementation by dilation each seed by a ball shape and check the new HU values of the added pixels.
        :param roi:
        :return: liver segmentation
        """

        def homogeneity(new_pix_hu, lower_threshold=-100, upper_threshold=200):
            if lower_threshold <= new_pix_hu <= upper_threshold:
                return True
            return False

        # get the points
        seeds_list = self.findSeeds(roi)
        ct_arr = self.ct_arr
        image = np.zeros(ct_arr.shape, dtype=np.uint8)
        # Perform Seeded Region Growing with N initial points

        for seed in seeds_list:
            image[seed[0], seed[1], seed[2]] = 1
            sphere = ball(1)
            # check the ball on the image
            dilated = dilation(image, selem=sphere)
            new_area = dilated - image
            new_pix = np.sum(new_area)

            if new_pix > 10:
                points = np.argwhere(new_area)
                for pixel in points:
                    if homogeneity(ct_arr[pixel[0], pixel[1], pixel[2]]):
                        image[pixel[0], pixel[1], pixel[2]] = 1

        liver_seg = image
        return liver_seg

    def segmentLiver(self, aorta_nii):
        """
        Create a liver segmentation from aorta ground-truth, CT scan.
        :param aorta_nii:
        :return: Liver segmentation array.
        """

        body_seg = self.IsolateBody()

        roi = self.liver_roi(body_seg, aorta_nii)
        liver_seg = self.multipleSeedsRG(roi)

        new_nifti = nib.Nifti1Image(liver_seg.astype(np.float), self.nii_scan.affine)
        nib.save(new_nifti, f'{self.name}_LiverSeg.nii.gz')

        return liver_seg


if __name__ == '__main__':
    data_path = '/cs/casmip/public/for_aviv/MedicalImageProcessing/Targil1_data'
    results = []
    for file in os.listdir(data_path):
        if 'CT' in file and 'Case1' in file:
            name = file.split('.')[0]
            seg = Segmentation(f'{data_path}/{file}', name)
            aorta_file = file.replace('CT', 'Aorta')
            aorta_nii = nib.load(f'{data_path}/{aorta_file}')

            liver_seg = seg.segmentLiver(aorta_nii)
            gt_file = file.replace('CT', 'liver_segmentation')
            gt = nib.load(f'{data_path}/{gt_file}')
            gt_data = gt.get_fdata()

            gt_data = utils.check_orientation(gt, gt_data)

            name = file
            name, vod_result, dice_result = utils.evaluateSegmentation(liver_seg, gt_data, name)
            results.append((name, vod_result, dice_result))

    utils.plot_results(results, 'Liver segmentation DICE and VOD comparison')
