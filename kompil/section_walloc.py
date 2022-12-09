import os
import torch

from typing import Union, List, Any


def walloc(
    section_idx_fpath: str,
    total_mp: float,
    walloc_fpath: str,
    constraints_sect_mp: Union[List[float], None],
):
    assert os.path.exists(section_idx_fpath)

    assert total_mp > 0

    all_section_idx = torch.load(section_idx_fpath)

    nb_section = int(max(all_section_idx.max().item() + 1, 0))
    weight_allocation = torch.zeros(nb_section)
    remaining_mp = total_mp
    remaining_frame = len(all_section_idx)

    # First : apply weight contraints if provided
    # constraint format is : idx_sect_x mp_x idx_sect_y mp_y ....
    if constraints_sect_mp:
        assert len(constraints_sect_mp) % 2 == 0

        pair_sect_mp = [
            (int(constraints_sect_mp[i]), float(constraints_sect_mp[i + 1]))
            for i in range(0, len(constraints_sect_mp), 2)
        ]

        for pair in pair_sect_mp:
            assert remaining_mp >= 0

            sect_idx = pair[0]
            nb_mp = pair[1]

            assert sect_idx >= 0 and nb_mp > 0

            nb_frame_in_section = torch.count_nonzero(all_section_idx == sect_idx).item()

            assert nb_frame_in_section > 0

            weight_allocation[sect_idx] = nb_mp
            remaining_mp -= nb_mp
            remaining_frame -= nb_frame_in_section

    # Second : apply remaining mp to the other sections according nb of frame
    for sect_idx in range(nb_section):
        if weight_allocation[sect_idx] != 0:
            print(sect_idx, "section :", weight_allocation[sect_idx].item(), "mp (forced)")
        else:
            assert remaining_mp >= 0

            nb_frame_in_section = torch.count_nonzero(all_section_idx == sect_idx).item()

            assert nb_frame_in_section > 0

            nb_mp = (nb_frame_in_section / remaining_frame) * remaining_mp
            weight_allocation[sect_idx] = nb_mp
            remaining_mp -= nb_mp
            remaining_frame -= nb_frame_in_section
            print(sect_idx, "section :", nb_mp, "mp")

    print("Remaining :", remaining_mp, "mp")

    weight_allocation *= 1e6

    torch.save(weight_allocation, walloc_fpath)
