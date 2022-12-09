import torch
import os

from typing import Union, List


def get_frame_location_mapping(
    nb_frame: int,
    start_frame: Union[int, List[int], None] = None,
    frames_count: Union[int, List[int], None] = None,
    cluster: Union[List[Union[str, int]], None] = None,
):
    """
    Returns an integer tensor (nb_frame sized) that specified which frame must processed
    If a value at index i is -1, it means that the frame i must not be considered for training;
    Any other value at index i means the original section/cluster of the frame i and that the frame must be considered for training
    """
    # If nothing specified : get the whole video
    if not cluster and not start_frame and not frames_count:
        return torch.zeros(nb_frame, dtype=torch.int16)

    # Initialize mapping, by default : -1 to revoke all frame
    mapping = -torch.ones(nb_frame, dtype=torch.int16)

    if start_frame or frames_count:
        # Start frames and frames count to sections
        start_frame = start_frame if start_frame else [0]
        start_frame = start_frame if isinstance(start_frame, list) else [start_frame]
        frames_count = frames_count if frames_count else [nb_frame]
        frames_count = frames_count if isinstance(frames_count, list) else [frames_count]
        assert len(start_frame) == len(
            frames_count
        ), "Sections lists have not the same number of elements."
        sections = list(zip(start_frame, frames_count))

        # Use section index as mapping index
        for i in range(len(sections)):
            start, count = sections[i]
            mapping[start : start + count] = i

    if cluster:
        # Format : file, [idx_0, idx_1..]
        assert len(cluster) >= 2

        cluster_file = cluster[0]

        assert os.path.exists(cluster_file)

        # File must be generated via kompil cluster CLI
        clusters_idx = torch.load(cluster_file)

        assert clusters_idx is not None

        authorized_cluster_idx = list(map(int, cluster[1:]))

        # Use cluster index as mapping index
        for frame_idx in range(nb_frame):
            cluster_idx = clusters_idx[frame_idx]
            if cluster_idx in authorized_cluster_idx:
                mapping[frame_idx] = cluster_idx

    return mapping


def mapping_to_index_table(frames_mapping: torch.Tensor) -> list:
    # Just keep selected frames (from selected cluster or bounding)
    kept_frames = frames_mapping > -1

    # Remove unwanted frame indexes
    index_table = torch.nonzero(kept_frames).squeeze(1).tolist()

    return index_table
