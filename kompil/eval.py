import os
import json
import torch
from typing import Optional, List, Union

import kompil.data.video as video
from kompil.data.data import VideoDataset
from kompil.data.section import get_frame_location_mapping, mapping_to_index_table
from kompil.profile.bench import bench_model, Benchmark
from kompil.utils.video import chw_to_resolution
from kompil.utils.resources import get_video
from kompil.utils.colorspace import convert_shape_to_colorspace
from kompil.nn.models.model import VideoNet
from kompil.packers.utils import generic_load


def bench(
    model_file: str,
    video_path: str,
    device_name: str,
    batch_size: int,
    fp32: bool,
    cluster: List[Union[str, int]],
    resolution: Optional[str],
):
    device = torch.device(device_name)
    # Load model
    model: VideoNet = generic_load(model_file)
    model = model.to(device)
    if fp32:
        model = model.float()
    else:
        model = model.half()

    # Find final resolution
    rgb8_shape = convert_shape_to_colorspace(model.frame_shape, model.colorspace, "rgb8")
    model_resolution = chw_to_resolution(rgb8_shape)
    resolution = resolution if resolution is not None else model_resolution

    # Handle clusters
    if cluster is not None:
        nb_frame = len(torch.load(cluster[0]))

        frames_mapping = get_frame_location_mapping(nb_frame, cluster=cluster)
        index_table = mapping_to_index_table(frames_mapping)
    else:
        index_table = None

    # Load video reference
    video_loader = video.StreamVideoLoader(
        video_path, model.colorspace, device, resolution, index_table=index_table, half=not fp32
    )

    # Load dataset
    dataset = VideoDataset(video_loader, half=not fp32)

    # Run
    def cb(frame_id, frames_len):
        print(f"Frame {frame_id}/{frames_len - 1}", end="\r")

    benchmark = bench_model(dataset, model, batch=batch_size, callback=cb, resolution=resolution)
    print()
    return benchmark


def write_output_file(output_file: str, benchmark: Benchmark):
    # Check if target is a file or a folder to add file name
    output_file = os.path.expanduser(output_file)
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, "eval.json")
    # Check if the target already exists
    # TODO: uncomment this after adding an option to override
    # if os.path.exists(output_file):
    #     res = input(f"file {output_file} exists. Do you want to erase it? [n/y] ")
    #     if res.lower() not in ["y", "yes"]:
    #         return
    # Build eval data
    data = {
        metric_name: {
            "min": metric.min,
            "max": metric.max,
            "mean": metric.mean,
            "data": metric.data.tolist(),
        }
        for metric_name, metric in benchmark.items()
    }
    # Write the datafile
    with open(output_file, "w+") as file:
        json.dump(data, file, indent=4)


def evaluate(
    model_file: str,
    video_name: str,
    device_name: str,
    batch_size: int,
    fp32: bool,
    cluster: List[Union[str, int]],
    output_file: Optional[str],
    resolution: Optional[str],
):
    # Get the paths
    model_file = os.path.expanduser(model_file)
    video_path = get_video(video_name)
    # PSNR & SSIM & VMAF
    benchmark = bench(model_file, video_path, device_name, batch_size, fp32, cluster, resolution)
    # Print result
    print()
    print(benchmark.to_table())
    # Feed report
    if output_file is not None:
        write_output_file(output_file, benchmark)
