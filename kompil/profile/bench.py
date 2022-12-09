import abc
import torch
import tabulate
from typing import List, Union, Dict, Optional

from kompil.nn.models.model import VideoNet
from kompil.data.data import VideoDataset
from kompil.utils.y4m import Y4MReader
from kompil.utils.video import discretize, resolution_to_chw
from kompil.utils.colorspace import convert_to_colorspace
from kompil.metrics.metrics import factory as metric_factory
from kompil.data.timeline import create_timeline


class _TestData:
    def __init__(self, min: float, max: float, mean: float, data: Union[None, torch.Tensor]):
        self.__min = min
        self.__max = max
        self.__mean = mean
        self.__data = data

    def to_dict(self) -> Dict[str, float]:
        return {"min": self.__min, "max": self.__max, "mean": self.__mean}

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "_TestData":
        return _TestData(t.min().item(), t.max().item(), t.mean().item(), t)

    @staticmethod
    def from_dict(data: Dict[str, float]) -> "_TestData":
        return _TestData(data["min"], data["max"], data["mean"], None)

    @property
    def data(self) -> torch.Tensor:
        if self.__data is None:
            raise RuntimeError("The data in the TestData has been cleared")
        return self.__data

    @property
    def min(self) -> float:
        return self.__min

    @property
    def max(self) -> float:
        return self.__max

    @property
    def mean(self) -> float:
        return self.__mean

    def to_list(self) -> List[float]:
        return [self.min, self.max, self.mean]

    @property
    def average(self) -> float:
        return self.__mean

    def cleanup_data(self):
        """Clean the list, keep the min, max, mean"""
        if self.__data is None:
            return
        data = self.__data
        self.__data = None
        del data


def _str2(val):
    return f"{val:0.2f}"


def _str3(val):
    return f"{val:0.3f}"


class Benchmark:
    KNOWN_CONVERSION = {
        "psnr": ["PSNR", _str2],
        "ssim": ["SSIM", _str3],
        "vmaf": ["VMAF", _str2],
    }

    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            assert isinstance(value, _TestData), f"{key} should be of type _TestData."
            assert not hasattr(self, key), f"{self} has already an attribute called {key}"

            setattr(self, key, value)

        self.__data = kwargs

    def items(self):
        return self.__data.items()

    def get(self, name: str) -> _TestData:
        bench = self.__data.get(name)
        if bench is None:
            raise RuntimeError(f"{name} is not benched")
        return bench

    def cleanup_data(self):
        """Clean the data, keep the min, max, mean"""
        for _, data in self.__data.items():
            data.cleanup_data()

    def to_dict(self) -> str:
        return {key: data.to_dict() for key, data in self.__data.items()}

    def to_table(self) -> str:
        headers = ["Metric", "min", "avg", "max"]

        table = []

        for key, data in self.__data.items():
            conv = self.KNOWN_CONVERSION.get(key, None)
            if conv is None:
                table.append([key, data.min, data.average, data.max])
            else:
                nb = conv[1]
                table.append([conv[0], nb(data.min), nb(data.average), nb(data.max)])

        return tabulate.tabulate(table, headers, tablefmt="fancy_grid")

    @staticmethod
    def from_tensors(**kwargs) -> "Benchmark":
        data = {key: _TestData.from_tensor(datalist) for key, datalist in kwargs.items()}
        return Benchmark(**data)

    @staticmethod
    def from_dict(dict_data) -> "Benchmark":
        data = {key: _TestData.from_dict(datalist) for key, datalist in dict_data.items()}
        return Benchmark(**data)


class BenchmarkEpochStorage:
    """
    Automatic storage that run tests if not already for a specific epoch and cleanup most of the
    data of old epochs
    """

    def __init__(
        self, dataset: VideoDataset, bench_batch: Union[int, None], eval_metrics: List[str]
    ):
        self.__saved_test = dict()
        self.__dataset = dataset
        self.__last_run_epoch = None
        self.__bench_batch = bench_batch
        self.__eval_metrics = eval_metrics

    def __getitem__(self, epoch) -> Benchmark:
        return self.__saved_test[epoch]

    def __len__(self):
        return len(self.__saved_test)

    @property
    def last(self) -> Union[None, Benchmark]:
        return self.__saved_test.get(self.__last_run_epoch, None)

    @property
    def last_run_epoch(self):
        epoch = self.__last_run_epoch

        return epoch if epoch else 0

    def items(self):
        return self.__saved_test.items()

    def run_test(self, epoch: int, model: VideoNet) -> Benchmark:
        # Don't run twice for the same epoch
        if epoch in self.__saved_test:
            return self.__saved_test[epoch]
        # Cleanup the list from previous runs
        last_run = self.__saved_test.get(self.__last_run_epoch)
        if last_run:
            last_run.cleanup_data()
        self.__last_run_epoch = epoch
        # Run test
        test = bench_model(
            self.__dataset, model, batch=self.__bench_batch, metrics=self.__eval_metrics
        )
        # Store test
        self.__saved_test[epoch] = test

        return test


class MultiMetrics:
    def __init__(self, *metrics):
        self.__metrics = {}

        for metric_name in metrics:
            assert metric_name not in self.__metrics, "Twice the same metric"
            cls, args, kwargs = metric_factory().get(metric_name)
            self.__metrics[metric_name] = cls(*args, **kwargs)

    def add_frame(self, ref: torch.Tensor, dist: torch.Tensor):
        for metric in self.__metrics.values():
            metric.add_frame(ref, dist)

    def compute(self) -> float:
        for metric in self.__metrics.values():
            metric.compute()

    def to_benchmark(self) -> Benchmark:
        tensors = {
            metric_name: metric.get_score_list() for metric_name, metric in self.__metrics.items()
        }
        return Benchmark.from_tensors(**tensors)


class BenchInput(abc.ABC):
    @abc.abstractmethod
    def open(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def count(self) -> int:
        pass

    @abc.abstractmethod
    def read_frame(self) -> torch.Tensor:
        pass


class DatasetBenchInput(BenchInput):
    def __init__(self, dataset: VideoDataset, device: Optional[torch.device] = None):
        super().__init__()
        self.__dataset = dataset
        self.__device = device

    def open(self) -> None:
        self.__frame = 0

    def close(self) -> None:
        pass

    def count(self) -> int:
        return len(self.__dataset)

    def read_frame(self) -> torch.Tensor:
        if self.__frame >= len(self.__dataset):
            return EOFError()
        colorspace = self.__dataset.video_loader.colorspace
        _, f = self.__dataset[self.__frame]
        self.__frame += 1
        if self.__device is not None:
            f = f.to(self.__device)

        rgb8 = convert_to_colorspace(f, src=colorspace, dst="rgb8")
        return discretize(torch.clamp(rgb8, min=0.0, max=1.0))


class ModelBenchInput(BenchInput):
    def __init__(self, model: VideoNet, batch: int = 1):
        super().__init__()
        self.__model = model
        self.__batch = batch
        self.__loaded = {}
        self.__prev_grad = None
        self.__timeline = create_timeline(model.nb_frames, device=model.device).to(model.dtype)

    def open(self) -> None:
        self.__frame = 0
        self.__prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def close(self) -> None:
        torch.set_grad_enabled(self.__prev_grad)

    def count(self) -> int:
        return self.__model.nb_frames

    def read_frame(self) -> torch.Tensor:
        if self.__frame >= self.__model.nb_frames:
            return EOFError()
        # Preloaded case
        if self.__frame in self.__loaded:
            f = self.__loaded[self.__frame]
        # To be calculated
        else:
            self.__loaded.clear()
            vec_timeline = self.__timeline[self.__frame : self.__frame + self.__batch]
            res_frames = self.__model(vec_timeline)
            for i in range(min(self.__batch, self.__model.nb_frames - self.__frame)):
                self.__loaded[i + self.__frame] = res_frames[i]
            f = res_frames[0]
        # Convert
        self.__frame += 1
        rgb8 = convert_to_colorspace(f, src=self.__model.colorspace, dst="rgb8")
        return discretize(torch.clamp(rgb8, min=0.0, max=1.0))


class Y4mBenchInput(BenchInput):
    def __init__(
        self,
        fpath: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float,
    ):
        super().__init__()
        self.__file = Y4MReader(fpath)
        self.__device = device
        self.__dtype = dtype

    def open(self) -> None:
        self.__file.seek(0)

    def close(self) -> None:
        self.__file.close()

    def count(self) -> int:
        return self.__file.frames_count

    def read_frame(self) -> torch.Tensor:
        f = self.__file.read_frame()
        if self.__device is not None:
            f = f.to(self.__device)
        f = f.to(self.__dtype) / 255.0
        rgb8 = convert_to_colorspace(f, src=self.__file.colorspace, dst="rgb8")
        return discretize(torch.clamp(rgb8, min=0.0, max=1.0))


def bench(
    ref: BenchInput,
    dist: BenchInput,
    callback: callable = None,
    resolution: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> Benchmark:
    """Process test by comparing to sets."""
    ref_count = ref.count()
    assert ref_count == dist.count()

    if resolution is not None:
        _, th, tw = resolution_to_chw(resolution)

    metrics_names = metrics if metrics is not None else metric_factory().keys()
    metrics_compute = MultiMetrics(*metrics_names)

    ref.open()
    dist.open()
    for img_id in range(ref_count):
        # Get images
        img_ref = ref.read_frame()
        img_dist = dist.read_frame()

        # Resize to target resolution
        if resolution is not None:
            img_ref.unsqueeze_(0)
            img_dist.unsqueeze_(0)
            img_ref = torch.nn.functional.interpolate(img_ref, size=(th, tw), mode="bilinear")[0]
            img_dist = torch.nn.functional.interpolate(img_dist, size=(th, tw), mode="bilinear")[0]

        # Compute
        metrics_compute.add_frame(img_ref, img_dist)

        # Callback
        if callback:
            callback(img_id, ref_count)

    ref.close()
    dist.close()

    # Get the results
    metrics_compute.compute()
    return metrics_compute.to_benchmark()


def bench_model(
    dataset: VideoDataset,
    model: VideoNet,
    batch: int = 1,
    callback: callable = None,
    resolution: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> Benchmark:
    """
    Process test by comparing the dataset with the generated images from the model.

    :return: benchmark
    :rtype: Benchmark
    """
    ref = DatasetBenchInput(dataset, model.device)
    dist = ModelBenchInput(model, batch)
    return bench(ref, dist, callback, resolution, metrics)
