import torch
import torch.nn.functional

from typing import List, Tuple
from torch import Tensor
from torch.nn.modules.module import Module


class _Switch(Module):
    """Torch module to switch between different networks according to the input code."""

    ModuleList = List[torch.nn.Module]

    __constants__ = ["__modules"]

    def __init__(self, modules: "_Switch.ModuleList"):
        super().__init__()
        self.__modules = modules
        for i, module in enumerate(modules):
            self.add_module(f"module_{i}", module)

    def _frame_to_module(self, id_frame: int) -> Tuple[int, int]:
        """
        :return: id_module, frame_inside_module
        """
        raise NotImplementedError()

    def __resolve(self, x: Tensor) -> Tuple[int, int, int]:
        """
        Find the actual module running a specific input.

        :return: id_frame, id_module, frame_inside_module
        """
        id_frame = x.long().item()

        id_module, frame_inside_module = self._frame_to_module(id_frame)

        return id_frame, id_module, frame_inside_module

    def __one_at_the_time(self, x: Tensor) -> Tensor:
        _, id_module, frame_inside_module = self.__resolve(x)
        module = self.__modules[id_module]
        new_input = torch.Tensor([frame_inside_module]).to(x.device).to(x.dtype)
        return module(new_input.unsqueeze(0))

    def forward(self, batch_in: Tensor) -> Tensor:
        batch_size = batch_in.shape[0]
        inter_size = torch.Size(batch_in.shape[1:-1])

        # Flat the batch_size and optional dimensions, treat them likewise
        batch_in_flatten = batch_in.view(-1, 1)

        # Apply the switch for each elements
        output = torch.cat(
            [
                self.__one_at_the_time(batch_in_flatten[idx])
                for idx in range(batch_in_flatten.shape[0])
            ]
        )

        # Restore the batches and optional dimensions
        output_shape = output.shape[1:]
        return output.view(batch_size, *inter_size, *output_shape)


class Switch(_Switch):
    """
    Torch module to switch between different networks according to predefined sections.

    Example:
    layer = Switch(
        sections=[(0, 10, 1200), (1, 250, 1200)],
        modules=[(my_module_1, 11), (my_module_2, 11)]
    )
    In the first section, 0 is the index of the module, 10 is the start frame in the module and
    1200 is the duration.

    In the first module, 11 the number of inputs of the module.

    Note: every modules have to have the same output size

    """

    __constants__ = ["__sections", "__modules"]

    def __init__(self, sections: List[Tuple[int, int, int]], modules: _Switch.ModuleList):
        super().__init__(modules)
        self.__sections = sections

    def _frame_to_module(self, id_frame: int) -> Tuple[int, int]:
        """
        :return: id_module, frame_inside_module
        """
        id_counter = id_frame
        for id_module, init_frame, duration in self.__sections:
            if id_counter < duration:
                frame_inside_module = id_counter + init_frame
                return id_module, frame_inside_module
            id_counter -= duration

        raise RuntimeError(f"Did not find sequence for frame {id_frame}")


class SwitchIndexed(_Switch):
    """
    Torch module to switch between different networks according to an indexed table.

    Note 1: It works like a load layer, it loads the index from an external file when asked, then
    store the indexes as part of the parameters.
    Note 2: every modules have to have the same output size
    """

    __constants__ = ["__frame_index"]

    def __init__(self, nb_frames: int, index_file: str, modules: _Switch.ModuleList):
        super().__init__(modules)
        self.__nn_indexes = torch.nn.parameter.Parameter(
            torch.empty(nb_frames, dtype=torch.int16), requires_grad=False
        )
        # self.add_module("nn_indexes", self.__nn_indexes)
        self.fill_indexes(index_file)
        # frame index need to be lazy since the values of nn_indexes might be modified after init
        self.__frame_index = None

    def fill_indexes(self, index_file: str) -> torch.Tensor:
        print("Loading indexes from", index_file, "...")
        self.__nn_indexes.copy_(torch.load(index_file))

    def create_frame_index(self, nn_indexes: torch.Tensor) -> torch.LongTensor:
        minimum = nn_indexes.min()
        maximum = nn_indexes.max()
        frame_index = torch.zeros_like(nn_indexes, dtype=torch.long, device=nn_indexes.device)
        for i in range(minimum, maximum + 1):
            bin_mask = nn_indexes == i
            current_frame_index = torch.cumsum(bin_mask.long(), 0) - 1
            current_frame_index.masked_fill_(~bin_mask, 0)
            frame_index += current_frame_index
        return frame_index

    def _frame_to_module(self, id_frame: int) -> Tuple[int, int]:
        """
        :return: id_module, frame_inside_module
        """
        if self.__frame_index is None:
            self.__frame_index = self.create_frame_index(self.__nn_indexes)

        return self.__nn_indexes[id_frame], self.__frame_index[id_frame]
