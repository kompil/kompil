import abc
import copy
import numpy
import torch
import random
import threading
import collections

import kompil.data.video as video

from torch.utils.data import IterableDataset
from typing import Union
from kompil.data.timeline import create_timeline


class SequenceSelector(abc.ABC):
    @abc.abstractmethod
    def select(self, index_list: list) -> list:
        """
        Manipulate an index list and return the modified list. The returned list might be shorter,
        shuffled or whatever changes to the input index list.
        """

    @abc.abstractmethod
    def estimate_length(self, max_len: int) -> int:
        """
        Get the estimated length of the data that select will output.
        """


class SequenceStraight(SequenceSelector):
    """
    Keep the list intact.
    """

    def select(self, index_list: list) -> list:
        return copy.copy(index_list)

    def estimate_length(self, max_len: int) -> int:
        return max_len


class SequenceShuffle(SequenceSelector):
    """
    Just shuffle the list.
    """

    def select(self, index_list: list) -> list:
        copied_list = copy.copy(index_list)
        random.shuffle(copied_list)
        return copied_list

    def estimate_length(self, max_len: int) -> int:
        return max_len


class SequenceProbaCutoff(SequenceSelector):
    """
    Select an index list based on a probability list (called weights) and a cutoff parameter, then
    shuffle the list.
    """

    def __init__(self, weights: numpy.ndarray = None, cutoff: int = None):
        self.__weights = weights
        self.__cutoff = cutoff
        self.__lock = threading.Lock()

    def set_cutoff_ratio(self, cutoff_ratio: float, data_length: int):
        assert cutoff_ratio > 0.0 and cutoff_ratio <= 1.0
        cutoff = int(data_length * cutoff_ratio)
        assert cutoff > 0
        with self.__lock:
            self.__cutoff = cutoff

    def set_equal_weights(self, data_length: int):
        equal_weight = 1.0 / data_length
        weights = [equal_weight for _ in range(data_length)]
        self.set_weights(numpy.array(weights))

    def set_weights(self, weights: numpy.ndarray):
        with self.__lock:
            self.__weights = weights

    def select(self, index_list: list) -> list:

        with self.__lock:
            cutoff = self.__cutoff
            weights = self.__weights

        return numpy.random.choice(index_list, p=weights, size=(cutoff,))

    def estimate_length(self, _: int) -> int:
        with self.__lock:
            return self.__cutoff


class _ReindexIterator:
    def __init__(self, loader: video.VideoLoader, sequence: list):
        assert len(loader) >= len(sequence), "data must be longer than sequence"

        self.__loader = loader
        self.__index = 0
        self.__sequence = sequence

    def __len__(self):
        return len(self.__loader)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.__sequence):
            raise StopIteration

        frame_id = self.__sequence[self.__index]
        self.__index += 1
        return frame_id, self.__loader.get_raw(frame_id)


class VideoDataset(IterableDataset):
    MAX_QUEUE = 5

    def __init__(
        self,
        loader: video.VideoLoader,
        shuffle: bool = False,
        pin_memory: bool = False,
        half: bool = False,
    ):
        super().__init__()

        # Build data
        self.__half = half
        self.__loader = loader
        self.__max_len = len(self.__loader)
        timeline = create_timeline(self.__max_len, device=loader.device)
        self.__timeline = timeline.pin_memory() if pin_memory else timeline
        if self.__half:
            self.__timeline = self.__timeline.half()

        # Properties
        self.__sequence = list(range(self.__max_len))
        self.__sequence_selector = SequenceShuffle() if shuffle else SequenceStraight()

        # Thread
        self.__cv = threading.Condition()
        self.__thread = None
        self.__thread_quit = False
        self.__image_queue = collections.deque()
        self.__end = False

    def set_sequence_selector(self, selector: SequenceSelector):
        assert isinstance(selector, SequenceSelector)
        with self.__cv:
            self.__sequence_selector = selector

    @property
    def video_loader(self) -> video.VideoLoader:
        return self.__loader

    @property
    def timeline(self) -> torch.Tensor:
        return self.__timeline

    @property
    def device(self):
        return self.__loader.device

    @property
    def fps(self) -> int:
        return self.__loader.fps

    def __len__(self):
        return self.__sequence_selector.estimate_length(self.__max_len)

    def __getitem__(self, idx: Union[int, slice]):
        return self.__timeline[idx], self.__loader[idx]

    def __iter__(self):
        self.__stop_thread()
        self.__start_thread()
        return self

    def __next__(self):
        with self.__cv:
            while not self.__image_queue and not self.__end:
                self.__cv.wait()

            if self.__end:
                raise StopIteration

            time_vec, raw_frame = self.__image_queue.popleft()

            self.__cv.notify_all()

        frame = self.__loader.convert_raw(raw_frame)

        return time_vec, frame

    def __del__(self):
        self.__stop_thread()

    def __run(self):
        # Create a new sequence
        with self.__cv:
            sequence = self.__sequence_selector.select(self.__sequence)
        # Iterate over the sequence
        for frame_id, raw_frame in _ReindexIterator(self.__loader, sequence):
            time_vec = self.__timeline[frame_id]

            with self.__cv:
                while len(self.__image_queue) >= self.MAX_QUEUE:
                    self.__cv.wait()

                    if self.__thread_quit:
                        self.__end = True
                        self.__cv.notify_all()
                        return

                time_vec = self.__timeline[frame_id]
                self.__image_queue.append((time_vec, raw_frame))
                self.__cv.notify_all()

        with self.__cv:
            self.__end = True
            self.__cv.notify_all()

    def __start_thread(self):
        if self.__thread is not None:
            raise RuntimeError("Thread already launched")
        self.__end = False
        self.__thread_quit = False
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.start()

    def __stop_thread(self):
        if self.__thread is None:
            return
        with self.__cv:
            self.__thread_quit = True
            self.__cv.notify_all()
        self.__thread.join()
        self.__thread = None
