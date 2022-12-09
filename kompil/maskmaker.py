import os
import torch

from typing import Union, List

from kompil.utils.resources import get_video
from kompil.data.video import FullVideoLoader
from kompil.utils.args import KeyvalType, arg_keyval_list_to_dict
from kompil.maskmakers import maskmaker_factory, instantiate_maskmaker


def run_maskmaker(
    name: str,
    video: str,
    resolution: Union[str, None],
    colorspace: str,
    params: KeyvalType,
    output: Union[str, None],
):
    # Ensure good arguments
    assert maskmaker_factory().has(name), f"{name} maskmaker unknown"
    colorspace = colorspace if colorspace is not None else "yuv420"
    params = arg_keyval_list_to_dict(params)

    # Read video
    print("Reading the video...")
    video_loader = FullVideoLoader(
        video_file_path=get_video(video),
        colorspace=colorspace,
        storage_device=torch.device("cpu"),
        target_device=torch.device("cuda"),
        pin_memory=True,
        resolution=resolution,
    )

    # Process maskmaker
    maskmaker = instantiate_maskmaker(name, **params)
    mask = maskmaker.generate(video_loader)

    mask = torch.nan_to_num(mask, nan=0, posinf=0, neginf=0)
    mask = torch.clamp(mask.float(), 0, 1).half()

    # Save output
    if output is not None:
        output_fpath = os.path.expanduser(output)
        print("Saving in", output_fpath)
        torch.save(mask, output_fpath)


def run_maskmerger(
    masks_fpath: List[str],
    method: Union[str, None],
    output: Union[str, None],
):
    assert len(masks_fpath) > 1

    method = method if method is not None else "add"
    assert method in ["add", "max"]

    merge_mask = None

    for mask_fpath in masks_fpath:
        assert os.path.exists(mask_fpath)

        loaded_mask = torch.load(mask_fpath, map_location="cuda")

        print(f"Applying ({method}) mask from {mask_fpath}")

        if merge_mask is None:
            merge_mask = torch.clone(loaded_mask)
        else:
            assert (
                merge_mask.shape == loaded_mask.shape
            ), "All masks must have the same shape / nb frames"
            if method == "max":
                merge_mask = torch.maximum(merge_mask, loaded_mask)
            elif method == "add":
                merge_mask += loaded_mask

    merge_mask = torch.nan_to_num(merge_mask, nan=0, posinf=0, neginf=0)
    merge_mask = torch.clamp(merge_mask.float(), 0, 1).half()

    # Save output
    if output is not None:
        output_fpath = os.path.expanduser(output)
        print("Saving in", output_fpath)
        torch.save(merge_mask, output_fpath)
