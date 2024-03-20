import unittest
import numpy as np
import cv2
import sys
from pprint import pprint, pformat
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
        

    def test_sample_pathces_exhaustive(self):
        interpolated_mask = cv2.resize(self.binary_mask, np.array(self.binary_mask.shape) * 2, interpolation=cv2.INTER_NEAREST)
        sitk_mask = sitk.GetImageFromArray(interpolated_mask[:, :])
        filled_hole = sitk.BinaryFillhole(sitk_mask, True)

        patches, coords = sample_patches_exhaustive(filled_hole, filled_hole,  4, 1,
                                                    return_coords=True, drop_last=False)
        answer = [(2, 3) , (5, 3)  , (8, 3) , (11, 3) , (13, 3),
                  (2, 6) , (5, 6)  , (8, 6) , (11, 6) , (13, 6),
                  (2, 9) , (5, 9)  , (8, 9) , (11, 9) , (13, 9),
                  (2, 12), (5, 12) , (8, 12), (11, 12), (13, 12),
                  (2, 15), (5, 15) , (8, 15), (13, 15), (2 , 18),
                  (5, 18), (8, 18)]
        self.assertTrue(set(coords) == set(answer), pformat(f"{coords = },\n {answer = }", indent=2))
