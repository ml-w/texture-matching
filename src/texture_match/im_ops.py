from unittest import case
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from pandas import value_counts
from functools import partial


def resample_seg(im: sitk.Image, seg: sitk.Image) -> sitk.Image:
    NEED_RESAMPLE = not all([
        np.allclose(im.GetSize(), seg.GetSize()), 
        np.allclose(im.GetDirection(), seg.GetDirection()), 
        np.allclose(im.GetSpacing(), seg.GetSpacing())
    ])
    if NEED_RESAMPLE:
        seg = sitk.Resample(seg, referenceImage=im)
    return seg


        
def slicewise_operation(im: sitk.Image, func: Callable, axis: Optional[int]=-1) -> sitk.Image:
    if not callable(func):
        raise TypeError("Input is not callable.")
    
    if not isinstance(im, sitk.Image):
        raise TypeError(f"Image must be `sitk.Image`. Got {type(im)} instead.")
    
    if axis > im.GetDimension():
        raise ValueError("Axis specified larger than dimension")
    axis = axis % im.GetDimension()
    num_slice = im.GetSize()[axis]
    
        # Assuems the data type is uint8
    new_im = sitk.Image(im)
    new_im[:] = 0
    for i in range(num_slice):
        s = tuple([slice(None) if j != axis else i for j in range(im.GetDimension())])
        new_s: sitk.Image = func(im[s])
        new_im[s] = new_s
    return new_im
    

def slicewise_hole_fill(seg: sitk.Image, axis=-1) -> sitk.Image:
    return slicewise_operation(seg, sitk.BinaryFillhole, axis=axis)
        
        
def slicewise_binary_opening(seg: sitk.Image,axis=-1, **kwargs) -> sitk.Image:
    return slicewise_operation(seg, partial(sitk.BinaryMorphologicalOpening, **kwargs), axis=axis)


def slicewise_binary_closing(seg: sitk.Image, axis=-1, **kwargs) -> sitk.Image:
    return slicewise_operation(seg, partial(sitk.BinaryMorphologicalClosing, **kwargs), axis=axis)