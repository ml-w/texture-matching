import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Union, Optional, List, Tuple, Any


def find_square_patches(binary_mask: sitk.Image, l: int) -> List[Tuple]:
    """
    Identifies all square patches of size l x l within an irregular boundary
    defined by a binary mask that only has one connected component.
    
    Args:
        binary_mask (np.array): 
            A 2D numpy array representing the binary mask with a single component.
        l (int): 
            The side length of the square patches to find.

    Returns:
        list of tuple: A list of tuples, where each tuple represents the (row, column) coordinates
                       of the top-left corner of a valid l x l square patch within the boundary.
    """
    if l <= 0:
        raise ValueError("The side length l must be a positive integer.")
    
    # Check that there is only one connected component in the binary mask
    labeled_array: sitk.LabelImage = sitk.ConnectedComponent(binary_mask)
    
    # Convert to numpy because it's much faster to sum
    binary_mask = sitk.GetArrayFromImage(binary_mask)

    # Get bounding box of the label
    bbox = sitk.LabelShapeStatisticsImageFilter()
    bbox.Execute(labeled_array != 0)
    bbox = bbox.GetBoundingBox(1) # [x, y, w, h]
    
    x, y, w, h = bbox
    x_end, y_end = x + w, y + h
    
    # Get coords that are non-zero
    coords_loop = np.argwhere(binary_mask != 0) # coords are y, x
    coords_loop = coords_loop[:, ::-1].tolist()
    coords_loop: set = set([tuple(c) for c in coords_loop if (c[0] <= x_end - l + 1) and (c[1] <= y_end - l + 1)])
    
    # List to store the top-left corners of valid l x l square patches
    valid_patches = []

    # Iterate from short_edge sized square back to l because bigger square must contain smaller ones
    while len(coords_loop):
        # brute force is pretty quick already
        i, j = coords_loop.pop()
        ip, jp = i+l, j+l

        patch = binary_mask[j:jp, i:ip]
        if np.all(patch != 0) and np.sum(patch) == l*l:
            if not (i, j) in valid_patches:
                valid_patches.append((i, j))

    return valid_patches


def sample_patches(sitkslice: sitk.Image, sitkmask: sitk.Image, l: int) -> sitk.Image:
    """Extracts square patches from a 2D image slice based on a segmentation mask.

    This function computes a bounding box from the given segmentation mask and extracts patches
    from the corresponding 2D image slice. The patches are centered around the middle of the bounding
    box and are of size `l` x `l` pixels.

    Args:
        sitkslice (sitk.Image): 
            A 2D image slice from which patches will be extracted.
        sitkmask (sitk.LabelImage): 
            The 2D segmentation mask that determines the region of interest in the image slice.
        l (int): 
            The side length in pixels of the square patches to be extracted.

    Returns:
        sitk.Image: An image object containing the stacked extracted patches. The origin is set to
        (0, 0, 0) and the spacing is set to (1, 1, 1).

    Note:
        - This function strips the spacing information off from the output image. The output
          will be an SimpleITK Image with (0, 0, 0) as the origin and (1, 1, 1) as the spacing.
        - This function assumes the input slice has already been preprocessed (e.g., with hole filling).
        - The function `find_square_patches` used within the code should provide a list of top-left
          coordinates for all square patches of size `l` x `l` that fit within the segmentation mask.
          This function is not defined within the provided code and should be implemented separately.
    """
    # * Get patches
    # Get bbox and select the center slice
    lab_stat_filter = sitk.LabelShapeStatisticsImageFilter()
    lab_stat_filter.Execute(sitkmask != 0)
    x, y, w, h= lab_stat_filter.GetBoundingBox(1)

    # Narrow down to middle slice
    im_slice  = sitkslice[x:x+w, y:y+h]
    seg_slice = sitkmask[x:x+w, y:y+h]

    # numpy version 
    np_im_slice = sitk.GetArrayFromImage(im_slice)

    # Get all possible squares with correct size
    patches = find_square_patches(seg_slice, l)
    patches = np.array(patches)
    
    if not len(patches):
        msg = f"Cannot find any squares with size {l} that can fit into the segmentation!"
        raise ValueError(msg)

    # Extract patches from original images
    #! Note the tranpose of x, y
    patch_stack = np.stack([np_im_slice[j:j+l, i:i+l] for i, j in patches])
    
    # * Convert patch_stack to sitk image
    patch_stack = sitk.GetImageFromArray(patch_stack)
    return patch_stack

