import unittest
import numpy as np
import sys
sys.path.append('..')
from texture_match.patch_sampler import *


class TestPatchSampling(unittest.TestCase):
    def setUp(self) -> None:
        # Test this function
        self.binary_mask = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        self.binary_mask_itk = sitk.GetImageFromArray(self.binary_mask)
        return super().setUp()
    
    def test_find_square_patches(self) -> None:
        """Test getting square patches from segmentation"""
        patches = find_square_patches(self.binary_mask_itk, 2)
        answer = [(4, 3), (4, 6), (9, 2), (8, 3), (2, 2), (2, 5), (1, 3), 
                  (6, 2), (4, 5), (3, 6), (5, 3), (8, 2), (9, 4), (1, 2), 
                  (7, 3), (3, 5), (5, 2), (4, 4), (8, 4), (9, 3), (2, 6), 
                  (7, 2), (6, 3)]
        
        self.assertTupleEqual(tuple(patches), tuple(answer))
        
    def test_sample_patches(self) -> None:
        """Test getting the square patches from the original images using the coordinates found"""
        patches_stack = sample_patches(self.binary_mask_itk, self.binary_mask_itk, 2)
        self.assertIsInstance(patches_stack, sitk.Image)

        with self.assertRaises(ValueError):
            sample_patches(self.binary_mask_itk, self.binary_mask_itk, 4)
        

        
        