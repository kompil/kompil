import torch
import tempfile

from kompil.utils.resources import get_video
from kompil.cli_defaults import SectionSplitDefaults as defaults


def _make_hard_split(video_fpath: str, nb_section: int):
    import decord

    video = decord.VideoReader(video_fpath, ctx=decord.cpu(), num_threads=0)

    nb_frame = len(video)
    nb_frame_per_section = int(nb_frame / nb_section)
    all_section_idx = torch.empty(nb_frame, dtype=torch.int16)

    sect_idx = 0
    for i in range(0, nb_frame, nb_frame_per_section):
        if i + nb_frame_per_section >= nb_frame:
            nb_frame_per_section = nb_frame - i

        all_section_idx[i : i + nb_frame_per_section] = sect_idx
        sect_idx += 1

    return all_section_idx


def _make_cluster_split(video_fpath: str, nb_section: int):
    from kompil.section_cluster import gen_data, find

    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp_data:
        tmp_data_fpath = tmp_data.name
        gen_data(video_fpath, tmp_data_fpath)
        all_section_idx = find(tmp_data_fpath, nb_cluster=nb_section, output_file=None)

    return all_section_idx


def _make_dynamic_size_split(video_fpath: str):
    raise NotImplementedError


def split(
    video_fpath: str,
    nb_section: int,
    section_idx_fpath: str,
    method: str = defaults.SEC_METHOD,
):
    assert nb_section > 0

    video_fpath = get_video(video_fpath)

    if method == "hard":
        section_idx = _make_hard_split(video_fpath, nb_section)
    elif method == "dynamic":
        section_idx = _make_dynamic_size_split(video_fpath)
    elif method == "cluster":
        section_idx = _make_cluster_split(video_fpath, nb_section)
    else:
        raise NotImplementedError

    torch.save(section_idx, section_idx_fpath)
