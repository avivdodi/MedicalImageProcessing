import nibabel as nib
import numpy as np
from skimage.morphology import label
from skimage.morphology import remove_small_objects



def IsolateBody(ct_scan):

    ct_arr = ct_scan.get_fdata()

    # th
    ct_arr[ct_arr < 500] = ?
    ct_arr[ct_arr > 2000] = ?

    #clean noise
    ct_arr = label(ct_arr)
    ct_arr = remove_small_objects(ct_arr, min_size=50, connectivity=24)

    # compute the largest connected component

    return body_seg


def IsolateBS(body_seg):
    # find 2 large cavities

    #the plane BB corresponds to the inferior slice of the lungs; the plane CC corresponds to the widest slice of the lungs
    # The plane CC is the last slice/slice in which the lungs slice does not change much (or close
    # to it).

    return lungs_seg, bb, cc


def ThreeDBand (body_seg, lungs_seg, bb, cc):




    return lungs_band