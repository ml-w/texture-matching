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


def sample_patches_grid(sitkslice: sitk.Image,
                        sitkmask: sitk.Image,
                        l: int,
                        overlap: int,
                        return_coords: Optional[bool] = False,
                        drop_last: Optional[bool] = True) -> Union[sitk.Image, Tuple[sitk.Image, List[Tuple[int, int]]]]:
    """Extracts an exhaustive grid of patches from a SimpleITK image slice.

    Processes a given image slice and its associated mask to create an exhaustive
    grid of patches. The patches are of a specified size and are extracted with a
    specified overlap. This function includes patches having at least `l` pixels
    of segmentation in the patch. Optionally, it can also return the coordinates
    of the top-left corner of each patch within the original image slice.

    Args:
        sitkslice (sitk.Image):
            The SimpleITK image slice from which patches are to be extracted.
        sitkmask (sitk.Image):
            The SimpleITK mask image used to determine the region of interest for patch extraction.
        l (int):
            The side length of each square patch.
        overlap (int):
            The overlap between consecutive patches, in pixels.
        return_coords (bool, optional):
            If True, additionally returns the top-left coordinates of each patch within the original image slice.
            Defaults to False.
        drop_last (bool, optional):
            If True, patches that would fall partially outside the image bounds are dropped.
            Defaults to True.

    Returns:
        Union[sitk.Image, Tuple[sitk.Image, List[Tuple[int, int]]]]:
            A SimpleITK Image containing the patches if `return_coords` is False.
            If `return_coords` is True, a tuple containing the SimpleITK Image of patches and a list of tuples
            with each tuple containing the x and y coordinates of the top-left corner of each patch.

    Raises:
        ValueError:
            If `l` or `overlap` is less than 1, or if `overlap` is greater than or equal to `l`.

    Examples:
        To extract patches without coordinates:
            >>> patches = sample_patches_grid(image_slice, mask, 64, 32)

        To extract patches with coordinates:
            >>> patches, coords = sample_patches_grid(image_slice, mask, 64, 32, return_coords=True)

    .. note::
        - The patches are created by sliding a window of size `l` across the image with the given `overlap`.
        - The function calculates a bounding box from the mask to identify the region of interest.
        - Patches that do not contain a minimum number of non-zero pixels from the mask are excluded if
          `drop_last` is True. The minimum number is equal to the side length `l`.
        - The coordinates returned (if `return_coords` is True) are adjusted to account for the original
          position of the bounding box within the full image slice.
    """
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

    vert_coords = np.arange(0, h, l - overlap).tolist()
    horz_coords = np.arange(0, w, l - overlap).tolist()

    # * prevents last patch falling out of map
    try:
        while (h - l) <= vert_coords[-1]:
            vert_coords.pop(-1)

        while (w - l) <= horz_coords[-1]:
            horz_coords.pop(-1)

        if not drop_last:
            vert_coords.append(h - l)
            horz_coords.append(w - l)
    except IndexError:
        # warn about not enough space within segmentation
        msg = f"Square grid with size {l} cannot fit into the segmetnation. "
        raise IndexError(msg)
        


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


def sample_patches_random(sitkslice: sitk.Image,
                          sitkmask: sitk.Image,
                          l: int,
                          n: int,
                          return_coords: Optional[bool] = False) -> Union[sitk.Image, Tuple[sitk.Image, List[Tuple[int, int]]]]:
    """Extracts square patches from a 2D image slice based on a segmentation mask.

    Args:
        sitkslice (sitk.Image):
            A 2D image slice from which patches will be extracted.
        sitkmask (sitk.LabelImage):
            The 2D segmentation mask that determines the region of interest in the image slice.
        l (int):
            The side length in pixels of the square patches to be extracted.
        n (int):
            Number of samples.
        return_coords (bool):
            If specified, return also the patch coords.

    Returns:
        sitk.Image:
            An image object containing the stacked extracted patches. The origin is set to (0, 0, 0) and
            the spacing is set to (1, 1, 1).
        List[(int, int)]:
            A list of tuple of two coordinates
    """
    # * Get patches
    # numpy version
    np_im_slice = sitk.GetArrayFromImage(sitkslice)
    np_seg_slice = sitk.GetArrayFromImage(sitkmask)

    # Shrink segment to valid bound
    np_seg_slice[:l // 2+1, :] = 0
    np_seg_slice[-l // 2:, :] = 0
    np_seg_slice[:, -l // 2:] = 0
    np_seg_slice[:, :l // 2 + 1] = 0

    # Get all possible squares with correct size
    valid_coords = np.array(np.where(np_seg_slice != 0)).T - l // 2 # Shift the cord to corner
    # TODO: Need to add check check coords falling out of image


    patches = np.random.choice(np.arange(len(valid_coords)), size=n)
    patches = valid_coords[patches]

    if not len(patches):
        msg = f"Cannot find any squares with size {l} that can fit into the segmentation!"
        raise ValueError(msg)

    # Extract patches from original images
    #! Note there's no tranpose of i, j because numpy array was used to get the patch coords
    patch_stack = np.stack([np_im_slice[j:j+l, i:i+l] for j, i in patches])

    # * Convert patch_stack to sitk image
    patch_stack = sitk.GetImageFromArray(patch_stack)
    if return_coords:
        #! Transpose is here instead!
        return patch_stack, [(p[1], p[0]) for p in patches]
    else:
        return patch_stack