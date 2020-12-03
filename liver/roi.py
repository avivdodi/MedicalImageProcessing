import nibabel as nib
import numpy as np
from skimage.morphology import label, remove_small_objects, convex_hull_image, convex_hull_object
import utils


def IsolateBody(ct_scan):
    ct_arr = ct_scan.get_fdata()

    # th

    # ct_arr[ct_arr >= -500] = 1
    # ct_arr[ct_arr > 2000] = 0
    # ct_arr[ct_arr < -500] = 0

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


def IsolateBS(body_seg, aff):
    # find 2 large cavities

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
    # widest slice of lungs amond y axis (sagittal view)
    cc = utils.compute_widest_slice(lungs_seg)

    return lungs_seg, cc, bb


def ThreeDBand(body_seg, lungs_seg, bb, cc):
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


def MergedROI(ct_scan):
    ct_arr = ct_scan.get_fdata()
    # aorta gt
    # spine roi

    filename = ''
    new_nifti = nib.Nifti1Image(band.astype(np.float), ct_scan.affine)
    nib.save(new_nifti, f'{filename}_ROI.nii.gz')

    # return roi


def liver_roi(body_seg, ct_scan):
    ct_arr = ct_scan.get_fdata()

    # th
    ct_arr[(ct_arr > -100) & (ct_arr < 200)] = 1
    ct_arr[(ct_arr <= -100) | (ct_arr >= 200)] = 0
    intersection = np.logical_and(body_seg, ct_arr)

    # find half of the aorta
    # find the largest label there

    # save

    return


def findSeeds(ct_scan, roi):
    return seeds_list


def multipleSeedsRG(ct_scan, roi):
    return liver_seg


def segmentLiver(ct_nii, aorta_nii, output_name):
    new_nifti = nib.Nifti1Image(liver_seg.astype(np.float), ct_nii.affine)
    nib.save(new_nifti, f'{output_name}_LiverSeg.nii.gz')

    return liver_seg


if __name__ == '__main__':
    load = nib.load('/cs/casmip/public/for_aviv/MedicalImageProcessing/Targil1_data/Case2_CT.nii.gz')
    body_seg = IsolateBody(load)
    lungs_seg, cc, bb = IsolateBS(body_seg, load.affine)
    lungs_band = ThreeDBand(body_seg, lungs_seg, bb, cc)
