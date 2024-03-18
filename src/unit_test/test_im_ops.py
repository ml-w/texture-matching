import sys
sys.path.append("../")
import unittest
import SimpleITK as sitk
from texture_match.im_ops import *




class TestImOps(unittest.TestCase):
    def test_resample_img(self):
        im = sitk.ReadImage("./test_data/images/MRI_01.nii.gz")
        seg = sitk.ReadImage("./test_data/segment/MRI_02.nii.gz")
        
        seg = resample_seg(im, seg)
        self.assertTupleEqual(seg.GetSize(), im.GetSize())
        
    def test_slicewise_hold_fill(self):
        seg = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ], [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ]
        ], dtype='uint8')
        sitk_seg = sitk.GetImageFromArray(seg)
        
        target = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ], [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ]
        ], dtype='uint8')
        
        out = slicewise_hole_fill(sitk_seg)
        out = sitk.GetArrayFromImage(out)
        
        self.assertTrue(np.allclose(target, out), f"{target = }, {out = }")
        
    def test_slicewise_closing(self):
        seg = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ], [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 0, 0, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ]
        ], dtype='uint8')
        sitk_seg = sitk.GetImageFromArray(seg)
        
        target = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ], [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ]
        ], dtype='uint8')
        
        out = slicewise_binary_closing(sitk_seg)
        out = sitk.GetArrayFromImage(out)
        
        self.assertTrue(np.allclose(target, out), f"{target = }, {out = }")
        
    def test_slicewise_opening(self):
        seg = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 1, 0, 1, 0, 0], 
            [0, 0, 0, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ], [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ]
        ], dtype='uint8')
        sitk_seg = sitk.GetImageFromArray(seg)
        
        target = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ], [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ]
        ], dtype='uint8')
        
        out = slicewise_binary_opening(sitk_seg)
        out = sitk.GetArrayFromImage(out)
        
        self.assertTrue(np.allclose(target, out), f"{target = }, {out = }")