import sys
sys.path.append('..')
import unittest
import SimpleITK as sitk
import numpy as np
from texture_match.extract_features import *
from pathlib import Path
import radiomics, logging
from mnts.mnts_logger import MNTSLogger

class TestPatchPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.binary_image = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        self.binary_image_sitk = sitk.GetImageFromArray(self.binary_image)
        self.pyrad_setting_file = Path("../assets/pyrad_settings.yml")
        if not self.pyrad_setting_file.is_file():
            raise FileNotFoundError(f"Cannot local setting file: {self.pyrad_setting_file}")
        
        # mute radiomics
        radiomics.logger.setLevel(logging.ERROR)
        MNTSLogger.set_global_log_level('debug')
        
        return super().setUp()
    
    def test_get_vacinity_segment(self):
        x = get_vacinity_segment_slice(self.binary_image_sitk, dilate=3, shrink = 1)
        x = sitk.GetArrayFromImage(x)
        
        target = np.array([
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        equal = target == x
        self.assertTrue(np.all(equal.ravel()), msg=f"{x = } NOT equal {target = }")
        pass
    
    def test_extract_features(self):
        """Test feature extraction"""
        dummy_image = np.random.randn(10, 16, 16) * 32
        dummy_image = sitk.GetImageFromArray(dummy_image.astype('uint32'))
        
        df = get_features_from_patch_stack(dummy_image, pyrad_setting=self.pyrad_setting_file)
        self.assertGreater(df.shape[0], 1)
        
        
    def test_extract_features_from_slice(self):
        """Test extraction from slice"""
        # Load image
        img = sitk.ReadImage("./test_data/images/MRI_01.nii.gz")
        seg = sitk.ReadImage("./test_data/segment/MRI_01.nii.gz", outputPixelType=sitk.sitkUInt8)
        seg[:, :, :] = 0
        seg[100:120, 100:120, 100] = 1
        
        df = get_features_from_slice(img[..., 100], seg[..., 100] != 0, 16, self.pyrad_setting_file)
        self.assertEqual(df.shape[0], 25, f"{df.shape = }")
        self.assertGreater(df.shape[0], 1)
    
    def test_extract_features_from_image(self):
        # Load image
        img = sitk.ReadImage("./test_data/images/MRI_01.nii.gz")
        seg = sitk.ReadImage("./test_data/segment/MRI_01.nii.gz", outputPixelType=sitk.sitkUInt8)
        
        # Remake the segmentation 
        seg[:, :, :] = 0
        seg[100:120, 100:120, 95:100] = 1
        
        df = get_features_from_image(img, seg, patch_size=16, include_vicinity=True, 
                                     pyrad_setting=self.pyrad_setting_file, dilate=20, shrink=2)
        self.assertGreater(df.shape[0], 1)
        
    def test_extract_features_from_image_mpi(self):
        
        # Load image
        img = sitk.ReadImage("./test_data/images/MRI_01.nii.gz")
        seg = sitk.ReadImage("./test_data/segment/MRI_01.nii.gz", outputPixelType=sitk.sitkUInt8)
        
        # Remake the segmentation 
        seg[:, :, :] = 0
        seg[100:120, 100:120, 95:100] = 1

        df = get_features_from_image(img, seg, patch_size=16, include_vicinity=True, 
                                     pyrad_setting=self.pyrad_setting_file, dilate=20, shrink=2, 
                                     num_workers=8)
        self.assertGreater(df.shape[0], 25 * 5) # There's 25 16x16 patches per slice and 5 slices to process
        
        
    