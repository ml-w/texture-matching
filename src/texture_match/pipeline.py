import SimpleITK as sitk
import re
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Any, List, Tuple

PathLike = Union[str, Path]

class TextureMatchPipeline(object):
    r"""This pipeline offers a stremlined texture extraction interface"""
    def __init__(self, 
                 prad_setting: PathLike, 
                 patch_size: int, 
                 norm_graph: Optional[PathLike] = None, 
                 norm_state: Optional[PathLike] = None) -> None:
        pass
    
    def extract_features(self, img: sitk.Image, seg: sitk.Image) -> pd.DataFrame:
        pass
    
        