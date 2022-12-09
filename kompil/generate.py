import os
import av
import torch

from typing import List, Union, Optional

from kompil.nn.models.model import VideoNet, model_load
from kompil.data.section import get_frame_location_mapping, mapping_to_index_table
from kompil.data.timeline import create_timeline
from kompil.utils.video import tensor_to_numpy, save_frame, resolution_to_chw
from kompil.utils.ffmpeg import create_video_from_frames
from kompil.utils.colorspace import convert_to_colorspace
from kompil.utils.y4m import write_y4m


def __torch_img_to_av(img: torch.Tensor):
    img = tensor_to_numpy(img)
    return av.VideoFrame.from_ndarray(img, format="rgb24")


def _iter_batch_opt(length: int, batch: int):
    """
    Generator to give id list or slice list.
    """
    if batch is None:
        batch = 1

    for start in range(0, length, batch):
        stop = min(start + batch, length)
        yield slice(start, stop, 1)


def _get_model_according_storage(filename: str, gpu_if_possible: bool) -> VideoNet:
    data = torch.load(filename, map_location="cpu")

    model = model_load(filename)

    if gpu_if_possible:
        model = model.cuda()

    return model


def generate_frames(
    model_file: str,
    output: str,
    cluster: List[Union[str, int]],
    prefix: str,
    batch_size: int,
    cpu: bool,
    resolution: Optional[str],
):
    """
    Read a model and generate the frames as images in the target folder.
    """
    # Get target resolution
    if resolution is not None:
        tc, th, tw = resolution_to_chw(resolution)

    # Check output folder
    output = os.path.expanduser(os.path.abspath(output))
    parent_folder = os.path.dirname(output)
    if not os.path.exists(parent_folder) or not os.path.isdir(parent_folder):
        raise AttributeError(f"Bad parent folder {parent_folder}")
    if not os.path.exists(output):
        os.mkdir(output)
    if os.listdir(output):
        raise AttributeError(f"Folder {output} already contains files")

    # Load the model from a file and move it to cuda (required tu run it)
    print(f"Openning model {model_file}")
    model: VideoNet = _get_model_according_storage(model_file, gpu_if_possible=not cpu)
    model.eval()

    # Cluster to index table
    if cluster is not None:
        nb_frame = len(torch.load(cluster[0]))
        frames_mapping = get_frame_location_mapping(nb_frame, cluster=cluster)
        index_table = mapping_to_index_table(frames_mapping)
    else:
        index_table = list(range(model.nb_frames))
    index_table = torch.LongTensor(index_table).to(model.device)

    # Generate frames
    os.makedirs(output, exist_ok=True)
    time_frame = create_timeline(model.nb_frames, device=model.device)
    counter = 0
    last_frame = len(index_table)
    with torch.no_grad():
        for index_ids in _iter_batch_opt(len(index_table), batch_size):
            # Run model on target indexes
            frames_ids = index_table[index_ids]
            tin = torch.index_select(time_frame, 0, frames_ids)
            tout = model.forward_rgb8(tin)
            # Resize to target resolution
            if resolution is not None:
                tout = torch.nn.functional.interpolate(tout, size=(th, tw), mode="bilinear")
            # Save frames
            for i in range(len(frames_ids)):
                frames_id = frames_ids[i]
                filepath = os.path.join(output, f"{prefix}{frames_id}.png")
                save_frame(tout[i], filepath)
                counter += 1
                print(
                    f"Generation: {counter}/{last_frame} ({counter/last_frame * 100:0.1f}%)",
                    end="\r",
                )
    print()
    print("Generation end.")


def generate_video(model_file: str, output: str, codec: str, cpu: bool):
    """
    Read a model and generate its video.
    """
    # Incompatibilities for now:
    if codec not in ["mpeg4"]:
        raise NotImplementedError("Only codec mpeg4 is available for now")

    # TODO: Make it compatible with more codecs, specifically libx264

    # Load the model from a file and move it to cuda (required tu run it)
    print(f"Openning model {model_file}")
    model: VideoNet = _get_model_according_storage(model_file, gpu_if_possible=not cpu)
    c, h, w = model.frame_shape
    fps = round(model.fps)
    video_len = model.nb_frames

    # Open video stream
    print(f"Openning file {output}")
    container = av.open(output, mode="w")

    stream = container.add_stream(codec, rate=fps)
    stream.width = w
    stream.height = h

    for i, img in enumerate(model.generate_video()):
        print(f"frame {i + 1}/{video_len} ...", end="\r")
        frame = __torch_img_to_av(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    print()
    print("Flushing...")

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()

    print("Video written")


def run_model(model: VideoNet):
    time_frame = create_timeline(model.nb_frames, device=model.device).to(model.dtype)
    with torch.no_grad():
        for id_frame in range(model.nb_frames):
            image = model.forward(time_frame[id_frame].unsqueeze(0))[0]
            yield image.contiguous()


def generate_y4m(model_file: str, output: str, cpu: bool):
    """
    Read a model and generate its video.
    """
    # Load the model from a file and move it to cuda (required tu run it)
    print(f"Openning model {model_file}")
    model: VideoNet = _get_model_according_storage(model_file, gpu_if_possible=not cpu)
    c, h, w = model.frame_shape
    fps = round(model.fps)

    assert model.colorspace in ["ycbcr420", "ycbcr420shift"]

    with write_y4m(output, w * 2, h * 2, (fps, 1)) as y4m:
        for image in run_model(model):
            image = convert_to_colorspace(image, model.colorspace, "ycbcr420")
            image.clip_(0.0, 1.0)
            image = (image * 255.0).to(torch.uint8)
            y4m.write_frame(image)

    print("Video written")


def generate_video_from_frames(path: str, output: str, codec: str, fps: int):
    """
    Read a folder to get the frames and build the video.
    """
    folder, prefix = os.path.split(path)

    create_video_from_frames(output, folder, prefix, codec, fps)
