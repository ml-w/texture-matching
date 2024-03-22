from functools import partial
import SimpleITK as sitk
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any, Union, Tuple

from pexpect import EOF
from .patch_sampler import find_square_patches, sample_patches, sample_patches_grid, sample_patches_random
from mnts.mnts_logger import MNTSLogger
from mnts.utils import repeat_zip

import time
import warnings
import multiprocessing as mpi

warnings.filterwarnings('ignore', category=UserWarning)


def get_vicinity_segment_slice(seg_slice: sitk.Image,
                               dilate: Optional[int] = 16,
                               shrink: Optional[int] = None) -> sitk.Image:
    """Generates a vicinity segmentation slice from an input segmentation slice.

    The input segmentation slice is first dilated by a specified number of pixels to include adjacent
    tissue, and then the original segmentation is subtracted from this dilated segmentation to isolate
    the vicinity region. This region is then optionally shrunk to maintain a distance from the original
    tissue. The resulting segmentation is intended to capture tissues that are near but not part of 
    the target tissue, which can be crucial for distinguishing between cancerous and non-cancerous areas.

    Args:
        seg_slice (sitk.Image): 
            The input 2D segmentation slice as a SimpleITK Image.
        dilate (int, Optional): 
            The number of pixels by which to dilate the input segmentation. Defaults to 16.
        shrink (int, Optional): 
            The number of pixels by which to shrink the dilated vicinity segmentation. If not provided,
            it defaults to a quarter of the `dilate` value.

    Returns:
        sitk.Image: A SimpleITK Image representing the vicinity segmentation slice.

    .. notes::
        - The dilation and shrinkage are performed using a square kernel.
        - The result is a binary mask where the vicinity tissue is marked.
        - This function is typically used in medical image analysis to process tumor segmentation
          and can be critical in treatment planning and analysis.

    Raises:
        RuntimeError: If the dilation or erosion fails due to incompatible image dimensions or types.
    """
    logger = MNTSLogger['im_ops']
    
    # recommended setting
    shrink = shrink or dilate // 4
    logger.info(f"{shrink = }, {dilate = }")
    

    # * Dilate the patches for capturing vicinity non-cancer tissues
    seg_dilated = sitk.BinaryDilate(seg_slice, kernelRadius = [dilate, dilate])
    seg_non_cancer = seg_dilated - seg_slice
    
    # Shrinks it a bit so it stays further away from the cancer
    seg_non_cancer = sitk.BinaryErode(seg_non_cancer, kernelRadius = [shrink, shrink])
    return seg_non_cancer


def create_dummy_segment(image: sitk.Image) -> sitk.Image:
    """Create a dummy segmentation that covers the enture image"""
    return sitk.BinaryThreshold(image, 0, 0, 0, 1)


def get_features_from_patch_stack(stack: sitk.Image, pyrad_setting: Union[Path, str]) -> pd.DataFrame:
    """Extracts radiomic features from an image stack using PyRadiomics and MRI Radiomics Toolkit.

    This function processes a 3D SimpleITK image stack to extract radiomic features using the PyRadiomics
    library. A dummy segmentation is created for the entire image stack, and both the image stack and
    segmentation are temporarily saved to disk. MRI Radiomics Toolkit (mradtk) is then used to extract
    the features as per the PyRadiomics settings provided in `pyrad_setting`. The features are extracted
    by slice.

    Args:
        stack (sitk.Image): 
            A 3D SimpleITK image stack from which radiomic features are to be extracted.
        pyrad_setting (Union[Path, str]): 
            The path to the PyRadiomics settings file (YAML or JSON). This file specifies the
            configuration for feature extraction.

    Returns:
        pd.DataFrame: 
            A pandas DataFrame containing the extracted radiomic features. Shape is (n_patches, n_features).

    .. notes::
        - This function requires the `mradtk` library and PyRadiomics version 3.1.0 or above.
        - PyRadiomics logger functions will be overridden, and only errors will be reported.
        - The function `create_dummy_segment` creates a mask with all pixels set as foreground
        - The radiomic features are extracted by slice (2D) from the 3D stack.
        - The temporary files created during the feature extraction process are cleaned up automatically
          after the extraction is complete, but you need to make sure your system has enough space.

    Raises:
        ImportError: If `mradtk` or `radiomics` is not installed.
        IOError: If there is a problem reading the image stack or the settings file.
        RuntimeError: If feature extraction fails.
    """
    import warnings
    import radiomics, logging
    import mri_radiomics_toolkit as mradtk
    # warnings.filterwarnings('ignore', category=UserWarning)
    # radiomics.logger.setLevel(logging.ERROR) # Effectively multing it
    
    # create dummy segmentation
    dummy_seg_stack = create_dummy_segment(stack)
    
    # write both iamges to a temp directory
    with tempfile.TemporaryDirectory() as temp_dir: 
        temp_path = Path(temp_dir)
        temp_im_dir = temp_path / "IMG"
        temp_seg_dir = temp_path / "SEG"
        temp_im_dir.mkdir()
        temp_seg_dir.mkdir()

        # There's a limit in NIFTI's Dimension ~32000. If stack is over this number,
        sitk.WriteImage(stack, temp_im_dir / "Temp_Image.nii.gz")
        sitk.WriteImage(dummy_seg_stack, temp_seg_dir / "Temp_segment.nii.gz")
        
        fe = mradtk.FeatureExtractor(id_globber="Temp", param_file=str(pyrad_setting), by_slice=2)
        # pointless to set num worker > 1 because this reads one image only. Implementation within mradtk is
        # pooled for working on multiple images rather than multiple slices in the same image.
        df = fe.extract_features(temp_im_dir, temp_seg_dir, by_slice=2, num_workers=1)
        
    return df
    

def get_features_from_slice(im_slice: sitk.Image,
                            seg_slice: sitk.Image,
                            patch_size: int,
                            pyrad_setting: Union[Path, str],
                            grid_sampling: Optional[int] = 0,
                            grid_drop_last: Optional[bool] = False,
                            random_sampling: Optional[int] = 0) -> pd.DataFrame:
    """Get features from slice"""
    if grid_sampling < 0:
        warnings.warn(f"Incorrect setting for `grid_sampling`: {grid_sampling}")
    if random_sampling < 0:
        warnings.warn(f"Incorrect setting for `random_sampling`: {random_sampling}")

    # Get patch stack
    if grid_sampling > 0:
        patch_stack, patch_coords = sample_patches_grid(im_slice, seg_slice, patch_size, grid_sampling,
                                                        return_coords=True,
                                                        drop_last=grid_drop_last)
    elif random_sampling > 0:
        # Randomly select from mask image
        patch_stack, patch_coords = sample_patches_random(im_slice, seg_slice, patch_size, random_sampling,
                                                          return_coords=True)
    else:
        patch_stack, patch_coords = sample_patches(im_slice, seg_slice, patch_size, return_coords=True)
    
    # Extract features
    feats = get_features_from_patch_stack(patch_stack, pyrad_setting=pyrad_setting)

    # attach the corner indice of patches to features
    feats[('Extract Parameters', 'Index', 'Patch Corner Coordinate')] = patch_coords
    
    return feats


def get_features_from_image(im: sitk.Image,
                            seg: sitk.Image,
                            patch_size: int,
                            pyrad_setting: Union[Path, str],
                            include_vicinity: Optional[bool] = False,
                            num_workers: Optional[int] = 1,
                            grid_sampling: Optional[int] = 0,
                            grid_drop_last: Optional[bool] = False,
                            random_sampling: Optional[int] = 0,
                            **kwargs) -> pd.DataFrame:
    """Extracts features from an image using segmentation masks and PyRadiomics settings.

    Processes each slice of a 3D image and its corresponding segmentation to extract radiomic features.
    Optionally includes features from the vicinity of the segmented region.

    Args:
        im (sitk.Image): 
            The 3D image from which to extract features.
        seg (sitk.Image): 
            The 3D segmentation mask corresponding to the image.
        patch_size (int): 
            The size of the patches to use for feature extraction.
        pyrad_setting (Union[Path, str]): 
            The PyRadiomics settings file path or string.
        include_vicinity (bool, Optional): 
            Flag indicating whether to include features from the vicinity of the segmented region.
            Defaults to False.
        num_workers (int, Optional):
            Number of workers for MPI feature extracting. Each process will extract features from one
            patch stack. Default to 1, meaning no MPI.
        grid_sampling (int, Optional):
            Overlap setting grid patch generation, grid patch setting will be activated if this
            arguement > 0. Use it with pre-extraction tweak option to increase the sampling area.
            Include vicinity has no effect if this is > 0. Default to 0.
        grid_drop_last (bool, Optional):
            If `True`, the last patch will be dropped so that all patches are equally spaced.
        random_sampling (int, Optional):
            If not 0, will perform random sampling from vicinity slice, only effective when vicinity slice
            extraction is on.
        **kwargs:
            Additional keyword arguments to be passed to `get_vacinity_segment_slice`.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features for each image slice.

    Raises:
        ValueError: If the image and segmentation size, origin, or direction do not match.

    .. notes:
        - You should perform normalization prior to using this function.
        - The function assumes that the last axis of the image is the slice axis.
        - This function checks the `Size`, `Origin` and `Direction` of the input image and segmentation
        - The output DataFrame will have a multi-index if `include_vicinity` is True, and a single
          index otherwise.

    See Also:
        :func:`texture_match.patch_sampler.sample_patches_grid`

    """
    logger = MNTSLogger['extract-features']
    
    # Assume the last axis is the target axis
    num_slice = im.GetSize()[-1]
    
    # Get bounding box of the label
    bbox = sitk.LabelShapeStatisticsImageFilter()
    bbox.Execute(seg != 0)
    bbox = bbox.GetBoundingBox(1) 
    x, y, z, w, h, d = bbox
    
    
    # Check if image and segmentation have the same size, origin, and direction
    if not np.allclose(im.GetSize(),seg.GetSize()):
        raise ValueError(f"Image and segmentation size mismatch: {im.GetSize()} != {seg.GetSize()}")
    if not np.allclose(im.GetOrigin(), seg.GetOrigin()):
        raise ValueError(f"Image and segmentation origin mismatch: {im.GetOrigin()} != {seg.GetOrigin()}")
    if not np.allclose(im.GetDirection(), seg.GetDirection()):
        raise ValueError(f"Image and segmentation direction mismatch: {im.GetDirection()} != {seg.GetDirection()}")

    # Set segmentation to have the same metadata as the image
    seg.CopyInformation(im)
    
    # Initialize an empty list to store the DataFrame rows
    rows = []
    
    # Loop through each slice and extract features
    if num_workers == 1:
        logger.info("Not using MPI, this might take a while.")
        for idx in range(z, z+d + 1):
            # select the slice
            _im_slice = im[:, :,idx]  
            _seg_slice = seg[:, :,idx] 
            
            o = _extract_features(idx, _im_slice, _seg_slice, patch_size=patch_size,
                                  grid_sampling=grid_sampling,
                                  grid_drop_last=grid_drop_last,
                                  random_sampling=random_sampling,
                                  include_vicinity=include_vicinity, pyrad_setting=pyrad_setting, **kwargs)
            if not o is None:    
                rows.append(o)
    else:
        logger.info(f"Using MPI with {num_workers} workers.")
        pool = mpi.Pool(num_workers)
        
        idxs = range(z, z + d + 1)
        args = [idxs, [im[:, :, idx] for idx in idxs], [seg[:, :, idx] for idx in idxs]]
        func = partial(_extract_features, patch_size = patch_size,
                       grid_sampling=grid_sampling,
                       grid_drop_last=grid_drop_last,
                       random_sampling=random_sampling,
                       include_vicinity = include_vicinity, pyrad_setting=pyrad_setting, **kwargs)
        
        p = pool.starmap_async(func, repeat_zip(*args))
        pool.close()
        
        while not p.ready():
            # ? Leave room for progress bar?
            # pbar.n = progress.value
            # pbar.refresh(nolock=False)
            time.sleep(0.1)
            
        o = p.get()
        pool.join()
        
        # remove None from o
        while type(None) in [type(oo) for oo in o]:
            type_list = [type(oo) for oo in o]
            none_idx = type_list.index(type(None))
            o.pop(none_idx)
        rows.extend(o)
    if len(rows) == 0:
        raise ValueError("No features returned, check your segmentation!")
    
    # Concatenate all feature rows into a DataFrame
    out_df = pd.concat(rows, axis=0)
    logger.info(f"Output_dimension: {out_df.shape}")
    return out_df
            

def _extract_features(i: int,
                      im_slice: sitk.Image,
                      seg_slice: sitk.Image,
                      patch_size: int=None,
                      include_vicinity: bool=None,
                      pyrad_setting: Union[str, Path]=None,
                      grid_sampling: Optional[int] = 0,
                      grid_drop_last: Optional[bool] = False,
                      random_sampling: Optional[int] = 0,
                      **kwargs):
    """This is a helper function that is intended for mpi"""
    logger = MNTSLogger['texture-match.extract_features']
    logger.info(f"Current thread: {mpi.current_process().name}")
    
    # Skip if found no labels
    if np.sum(sitk.GetArrayFromImage(seg_slice)) == 0:
        return None
    
    # Get features from the current slice
    try:
        feat_target = get_features_from_slice(im_slice, seg_slice, patch_size, pyrad_setting,
                                              grid_sampling=grid_sampling,
                                              grid_drop_last=grid_drop_last)
        feat_target[('Extract Parameters', 'Index', 'Slice Index')] = i
        feat_target[('Extract Parameters', 'Flags', 'Vicinity')] = False
        feat_target[('Extract Parameters', 'Settings', 'Grid Overlap')] = grid_sampling
        o = feat_target
    except ValueError as e:
        # when the segmentation is too small and there are no square that can be fitted, skip this slice
        logger.info(f"Segmentation cannot fit any squares of patch size: {patch_size}")
        logger.debug(f"Original error {e}")
        o = None
        
    
    # # If vicinity features are to be included
    if include_vicinity and o is not None and grid_sampling <= 0:
        logger.info("Performing extraction for vicinity.")
        
        # Get vicinity segmentation slice
        vic_seg_slice = get_vicinity_segment_slice(seg_slice, **kwargs)
        if np.sum(sitk.GetArrayFromImage(vic_seg_slice)) == 0:
            logger.warning("Nothing returned from vicinity dilate and shrink.")
            return  o 
            
        # Get features from the vicinity of the current slice
        try:
            feat_vic = get_features_from_slice(im_slice, vic_seg_slice, patch_size, pyrad_setting,
                                               random_sampling=random_sampling)
            feat_vic[('Extract Parameters', 'Index', 'Slice Index')] = i
            feat_vic[('Extract Parameters', 'Flags', 'Vicinity')] = True
            feat_vic[('Extract Parameters', 'Settings', 'Grid Overlap')] = 0
            o = pd.concat([feat_target, feat_vic], axis=0)
        except ValueError as e:
            logger.info(f"Vicinity cannot fit any squares of patch size: {patch_size}")
            logger.debug(f"Original error: {e}")
    
    # reconstruct index output
    if isinstance(o, pd.DataFrame):
        # Update index level names
        o.index.names = [n if n != 'Slice number' else 'Patch Number' for n in o.index.names]
        # Update Study number to idx
        original_index = [list(oo) for oo in zip(*o.index)]
        new_index = pd.MultiIndex.from_tuples([tuple(oo) for oo in zip([f"Slice {i}"] * len(original_index[0]), *original_index[1:])])
        o.index = new_index
    
    return o