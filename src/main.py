from multiprocessing import Value
from os import path
import click
from matplotlib.pylab import rand
import pandas as pd
import radiomics
from pathlib import Path
from mnts.utils import load_supervised_pair_by_IDs, get_unique_IDs
from mnts.mnts_logger import MNTSLogger
from typing import Optional, List, Tuple, Any, Union

from texture_match.extract_features import *
from texture_match.im_ops import *
from functools import partial

PathLike = Union[Path, str]

# mute radiomics logger
radiomics.logger.setLevel(40)

# Some parameter settings to make click more friendly
class IDListParamType(click.ParamType):
    name = 'idlist'

    def convert(self, value, param, ctx) -> List[str]:
        try:
            if value is None:
                return None

            if value.startswith('[') and value.endswith(']'):
                id_list = value[1:-1].split(',')
                # Strip whitespace and filter out empty strings
                return [id_str.strip() for id_str in id_list if id_str.strip()]
            else:
                self.fail('ID list must start with "[" and end with "]" and contain IDs separated by commas.',
                          param, ctx)
        except ValueError as e:
            self.fail(f'Could not parse ID list: {e}', param, ctx)


@click.command()
@click.argument('input_dir', nargs=1, type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument('segment_dir', nargs=1, type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument('output_file', nargs=1, type=click.Path(path_type=Path))
@click.option('-p', '--patch-size', default=16, type=click.IntRange(0, 128), help="Patch size")
@click.option('--pyrad-setting-file', type=click.Path(exists=True, dir_okay=False),
              help='Pyradiomics settings yaml file.')
@click.option('--id-globber', default=r"^\w+\d+")
@click.option('--include-vicinity', default=False, is_flag=True,
              help="If true, extract features from vicinity as well.")
@click.option('--vic-dilate-px', default=None, type=int,
              help='If not specified, this is calculated from patch_size.')
@click.option('--vic-shrink-px', default=None, type=int,
              help="If not specified, this is calcualted from patch_size.")
@click.option('--vic-random-sampling', default=0, type=click.IntRange(0), 
              help="If > 0, randomly sample from vicinity segment instead of exhaustively.")
@click.option('--with-normalization', default=False, is_flag=True,
              help="Normalize input before extraction of features.")
@click.option('--norm-graph', default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to normalization graph yaml file.")
@click.option('--norm-states', default=None, type=click.Path(file_okay=False),
              help="Path to normalization state folder if requred.")
@click.option('-e', '--grid-sampling', default=0, type=click.IntRange(0, 128),
              help="Patch grid overlap setting. If this > 0, patch grid with uniform overlap will be generate "
                   "rather than exhaust all possible points.")
@click.option('--pre-extraction-tweak', default=0, type=click.IntRange(-128, 128),
              help="Dilate or shrink mask prior to feature extraction.")
@click.option('-n', '--num-workers', default=1, type=click.IntRange(0, mpi.cpu_count()),
              help="Number of workers. Default to 1.")
@click.option('--idlist', default=None, type=IDListParamType(),
              help="Specify the ids. Default behavior glob all viable IDs.")
@click.option('--overwrite', default=0, is_flag=True,
              help="If true, overwrite the key in the output HDF5 if it exists. Default behavior is skipping.")
@click.option('--log-dir', default=None, type=click.Path(path_type=Path),
              help="If specified, a log file will be written here.")
@click.option('--extract-class', default=None, type=click.IntRange(0, 255),
              help="If specified, the specified segmentation label will be extracted from segmentation prior to "
                   "calculating the feature values.")
@click.option('--debug', default=False, is_flag=True, help="If specified, only work on 1 case.")
def main(input_dir: Path, 
         segment_dir: Path, 
         output_file: Path, 
         patch_size: int,
         pyrad_setting_file: Optional[PathLike],
         id_globber: Optional[str], 
         include_vicinity: Optional[bool], 
         vic_dilate_px: Optional[int],
         vic_shrink_px: Optional[int],
         vic_random_sampling: Optional[int],
         with_normalization: Optional[bool], 
         norm_graph: Optional[PathLike],
         norm_states: Optional[PathLike], 
         num_workers: Optional[int],
         idlist: Optional[Tuple[str]],
         overwrite: Optional[bool],
         grid_sampling: Optional[int],
         pre_extraction_tweak: Optional[int],
         extract_class: Optional[int],
         log_dir: Optional[PathLike], 
         debug: Optional[bool]):
    
    # * setup
    main_logger = MNTSLogger['texture-match'] if log_dir is None else MNTSLogger(str(log_dir), logger_name='texture-match', keep_file=True)
    output_file.parent.mkdir(exist_ok=True)
    if include_vicinity:
        vic_dilate_px = vic_dilate_px or int(patch_size * 2)
        vic_shrink_px = vic_shrink_px or vic_dilate_px // 4
    if debug:
        main_logger.set_global_log_level('debug')
    
    # Get IDs
    input_nifties = [l.name for l in input_dir.glob("*nii.gz")]
    seg_nifties = [l.name for l in segment_dir.glob("*nii.gz")]
    input_ids = get_unique_IDs(input_nifties, globber=id_globber)
    seg_ids = get_unique_IDs(seg_nifties, globber=id_globber)

    # * Operate only on IDs that has overlap
    intersection = list(set(input_ids) & set(seg_ids))
    intersection.sort()
    if idlist is not None:
        idlist = set(idlist)
        idlist = list(set(intersection) & idlist)
    else:
        idlist = intersection
    main_logger.info(f"Working on ID list: {idlist}")
    
    # * pairs to process
    pairs: Tuple[Path, Path] = load_supervised_pair_by_IDs(input_dir, segment_dir, globber=id_globber, return_pairs=True, idlist=idlist)
    pairs = {k: v for k, v in zip(idlist, pairs)} # Note this works only if id list is sorted
    
    main_logger.info("{:=^120}".format(" Start Feature Extraction "))
    for idx, (im, seg) in pairs.items():
        main_logger.info(f"Working on: {idx:^10} {str(im.name)[-50:]:>50} <-> {str(seg.name)[-50:]:<50}")

        # * check if file is already processed
        out_name = output_file.with_suffix('.h5')
        if out_name.is_file() and not overwrite:
            main_logger.info("Found existing HDF5 store, trying to continue from existing work.")
            with pd.HDFStore(out_name) as hdfstore:
                keys = list(hdfstore.keys())
                main_logger.info(f"Existing keys: {keys}")
                if f'/{idx}' in keys:
                    main_logger.info(f"Found key in output file, skipping {idx}")
                    continue


        # * load image
        sitk_im, sitk_seg = sitk.ReadImage(im), sitk.ReadImage(seg)
        
        # * preprocessing
        # normalization
        if with_normalization:
            if norm_graph is None or norm_states is None:
                raise ValueError("Must specify `norm_graph` and `norm_states` for `with_normalization`.")
            # Not implemented yet
            pass
        # extract sub-calss
        if extract_class is not None:
            main_logger.info(f"Extracting class {extract_class} in segmentation.")
            sitk_seg = sitk_seg == extract_class
        sitk_seg = sitk_seg != 0
        sitk_seg = sitk.Cast(sitk_seg, sitk.sitkUInt8)
        # dilate or shrink
        if not pre_extraction_tweak == 0:
            if pre_extraction_tweak < 0:
                tweak_func = partial(sitk.BinaryErode, kernelRadius=(pre_extraction_tweak, pre_extraction_tweak))
            else:
                tweak_func = partial(sitk.BinaryDilate, kernelRadius=(pre_extraction_tweak, pre_extraction_tweak))
            sitk_seg = slicewise_operation(sitk_seg, tweak_func)
        sitk_seg = slicewise_binary_opening(sitk_seg, kernelRadius=(3, 3))
        
        
        # * get features
        try:
            if not np.all([
                np.isclose(sitk_im.GetSize(), sitk_seg.GetSize()),
                np.isclose(sitk_im.GetSpacing(), sitk_seg.GetSpacing()),
                np.isclose(sitk_im.GetOrigin(), sitk_seg.GetOrigin()),
            ]):
                # Check origin distance greater than 15cm
                if np.linalg.norm(np.array(sitk_im.GetOrigin()) - np.array(sitk_seg.GetOrigin())) > 150:
                    main_logger.warn("There is a large origin displacement, please check your image and segmentation.")

                #! This needs a unit test
                main_logger.warning("Found discripency between image and segmentation pair, resampling segmentation.")
                main_logger.debug(f"{sitk_im.GetSize() = } -> {sitk_seg.GetSize() = } "
                                  f"{sitk_im.GetSpacing() = } -> {sitk_seg.GetSpacing() = } "
                                  f"{sitk_im.GetOrigin() = } -> {sitk_seg.GetOrigin() = } ")
                sitk_seg = sitk.Resample(sitk_seg, referenceImage=sitk_im,
                                         interpolator=sitk.sitkNearestNeighbor,
                                         outputPixelType=sitk.sitkUInt8)
            df = get_features_from_image(sitk_im,
                                        sitk_seg, 
                                        patch_size, 
                                        pyrad_setting=pyrad_setting_file, 
                                        include_vicinity=include_vicinity,
                                        grid_sampling=grid_sampling,
                                        num_workers=num_workers,
                                        random_sampling=vic_random_sampling,
                                        dilate = vic_dilate_px,
                                        shrink = vic_shrink_px)
            
            # * save features
            main_logger.info(f"Saving to: {out_name}")
            with pd.HDFStore(out_name) as hdf_file:
                if f'/{idx}' in hdf_file.keys():
                    if not overwrite:
                        main_logger.warning(f"Key {idx} already in output H5, this might lead to unecessary inflation"
                                            f"of file size.")
                    else:
                        main_logger.info(f"Key {idx} already in output H5, overwriting it")
                # TODO: Implement update function
                df.to_hdf(hdf_file, key=f'{idx}')
        except Exception as e:
            main_logger.exception(e)
            main_logger.error(f"Failed to process pair {(im, seg)}")
            
        if debug:
            break
    pass



if __name__ == '__main__':
    main()