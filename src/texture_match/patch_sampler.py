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


def sample_patches(sitkslice: sitk.Image,
                   sitkmask: sitk.Image,
                   l: int,
                   return_coords: Optional[bool] = False) -> Union[sitk.Image, Tuple[sitk.Image, List[Tuple[int, int]]]]:
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
        return_coords (bool):
            If specified, return also the patch coords.

    Returns:
        sitk.Image:
            An image object containing the stacked extracted patches. The origin is set to (0, 0, 0) and
            the spacing is set to (1, 1, 1).
        List[(int, int)]:
            A list of tuple of two coordinates

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
    if return_coords:
        return patch_stack, [(p[0] + x, p[1] + y) for p in patches]
    else:
        return patch_stack


def sample_patches_exhaustive(sitkslice: sitk.Image,
                              sitkmask: sitk.Image,
                              l: int,
                              overlap: int,
                              return_coords: Optional[bool] = False,
                              drop_last: Optional[bool] = True) -> Union[sitk.Image, Tuple[sitk.Image, List[Tuple[int, int]]]]:
    if l < 1 or overlap < 1:
        raise ValueError(f"Argument `l` and `overlap` must be greater than 1.")

    if overlap >= l:
        raise ValueError(f"`Overlap must be smaller than `l`.")

    lab_stat_filter = sitk.LabelShapeStatisticsImageFilter()
    lab_stat_filter.Execute(sitkmask != 0)
    x, y, w, h= lab_stat_filter.GetBoundingBox(1)

    # Narrow down to middle slice
    im_slice  = sitkslice[x:x+w, y:y+h]
    seg_slice = sitkmask[x:x+w, y:y+h]

    vert_coords = np.arange(0, y+h, l - overlap).tolist()
    horz_coords = np.arange(0, x+w, l - overlap).tolist()

    # * prevents last patch falling out of map
    while (h - l) <= vert_coords[-1]:
        vert_coords.pop(-1)

    while (w - l) <= horz_coords[-1]:
        horz_coords.pop(-1)

    if not drop_last:
        vert_coords.append(h - l)
        horz_coords.append(w - l)

    xx, yy = np.meshgrid(horz_coords, vert_coords)
    coords  = np.stack([xx.ravel(),yy.ravel()], axis=1).tolist() # add back the bounding box coordinates

    # check if patch lies within segment entirely
    to_remove = []
    np_seg_slice = sitk.GetArrayFromImage(seg_slice)
    for c in coords:
        dx, dy = c
        patch = np_seg_slice[dy:dy+l, dx:dx+l]
        # catches all patches that has at least l non-zero pixels
        if np.sum(patch != 0) < l:
            to_remove.append([dx, dy])

    for c in to_remove:
        coords.remove(c)

    np_im_slice = sitk.GetArrayFromImage(im_slice)
    patch_stack = np.stack([np_im_slice[j:j + l, i:i + l] for i, j in coords])

    patch_stack = sitk.GetImageFromArray(patch_stack)
    if return_coords:
        return patch_stack, [(p[0] + x, p[1] + y) for p in coords]
    else:
        return patch_stack