import nibabel as nib
import numpy as np
from skimage.morphology import label, remove_small_objects, convex_hull_image, convex_hull_object
import utils


def IsolateBody(ct_scan):
    """
    Body segmentation of all the body, without air.
    :param ct_scan:
    :return:Body segmentation
    """
    ct_arr = ct_scan.get_fdata()

    # filter th between -500 to 2000
    ct_arr[(ct_arr > -500) & (ct_arr < 2000)] = 1
    ct_arr[(ct_arr <= -500) | (ct_arr >= 2000)] = 0

    # clean noise
    ct_arr = label(ct_arr)
    ct_arr = remove_small_objects(ct_arr, min_size=200, connectivity=24)

    # compute the largest connected component
    ct_arr = label(ct_arr)
    assert (ct_arr.max() != 0)  # assume at least 1 CC
    body_seg = ct_arr == np.argmax(np.bincount(ct_arr.flat)[1:]) + 1

    # new_nifti = nib.Nifti1Image(body_seg.astype(np.float), ct_scan.affine)
    # nib.save(new_nifti, f'test.nii.gz')

    return body_seg


def IsolateBS(body_seg):
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

    # new_nifti = nib.Nifti1Image(lungs_seg.astype(np.float), aff)
    # nib.save(new_nifti, f'test.nii.gz')

    xmin, xmax, ymin, ymax, zmin, zmax = utils.compute_seg_boundary_inds(lungs_seg)

    # inferior slice of lungs.
    bb = zmin
    # widest slice of lungs among y axis (sagittal view)
    cc = utils.compute_widest_slice(lungs_seg)

    return lungs_seg, cc, bb


def ThreeDBand(body_seg, lungs_seg, bb, cc):
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

    print(band)
    # aff = nib.load('/cs/casmip/public/for_aviv/MedicalImageProcessing/liver/test.nii.gz').affine
    # new_nifti = nib.Nifti1Image(band.astype(np.float), aff)
    # nib.save(new_nifti, f'band.nii.gz')
    return band


def MergedROI(ct_scan, aorta_arr):
    # fixme
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


def liver_roi(body_seg, ct_scan, aorta_nii):
    # fixme

    ct_arr = ct_scan.get_fdata()
    aorta_arr = aorta_nii.get_fdata()

    # th
    ct_arr[(ct_arr > -100) & (ct_arr < 200)] = 1
    ct_arr[(ct_arr <= -100) | (ct_arr >= 200)] = 0
    intersection = np.logical_and(body_seg, ct_arr)

    # find half of the aorta

    # find the largest label there

    ct_label = label(intersection)
    liver_seg = ct_label == np.argmax(np.bincount(ct_label.flat)[1:]) + 1

    # todo check this func by save the nifti. maybe we need to use the aorta seg and location.
    # save

    return roi


def findSeeds(ct_scan, roi, seeds_num=200):
    """
    Find sample seeds from roi on ct_scan.
    :param ct_scan: The scan
    :param roi: The segmentation to search for seeds.
    :param seeds_num: number of seeds.
    :return: seeds list from the roi on the scan.
    """
    seeds_list = []
    ct_arr = ct_scan.get_fdata()
    points = np.argwhere(roi)
    np.random.shuffle(points)

    for ind in points[0:seeds_num]:
        seeds_list.append(ct_arr[ind[0], ind[1], ind[2]])

    return seeds_list


def multipleSeedsRG(ct_scan, roi):
    # todo fixme
    seeds_list = findSeeds(ct_scan, roi)

    # Perform Seeded Region Growing with N initial points

    return liver_seg


def segmentLiver(ct_nii, aorta_nii, output_name):
    # todo fix this function
    ct_scan = ct_nii.get_fdata()

    body_seg = IsolateBody(ct_scan)

    roi = liver_roi(body_seg, ct_scan, aorta_nii)
    liver_seg = multipleSeedsRG(ct_scan, roi)
    new_nifti = nib.Nifti1Image(liver_seg.astype(np.float), ct_nii.affine)
    nib.save(new_nifti, f'{output_name}_LiverSeg.nii.gz')

    return liver_seg


if __name__ == '__main__':
    load = nib.load('/cs/casmip/public/for_aviv/MedicalImageProcessing/Targil1_data/Case2_CT.nii.gz')
    body_seg = IsolateBody(load)
    lungs_seg, cc, bb = IsolateBS(body_seg, load.affine)
    lungs_band = ThreeDBand(body_seg, lungs_seg, bb, cc)
