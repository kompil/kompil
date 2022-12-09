import sys
import time
import pytorch_lightning as pl
from typing import Union, Tuple, List

from kompil.nn.models.model import VideoNet
from kompil.data.data import VideoDataset
from kompil.utils.time import now_str
from kompil.utils.static_config import get_mac_address
from kompil.profile.bench import BenchmarkEpochStorage, Benchmark, bench_model
from kompil.profile.criteria import Criteria
from kompil.train.optimizers import CustomScheduler
from kompil.train.training import TrainingParams
from kompil.profile.report import EncodingReport, Curve, build_git_info
from kompil.profile.profiling_callbacks import QuitCallback


class ReportCallback(pl.callbacks.Callback):
    def __init__(
        self,
        criteria: Criteria,
        dataset: VideoDataset,
        training_params: TrainingParams,
        tester: BenchmarkEpochStorage,
        quit_callback: QuitCallback,
        batch_size: int,
        topology_builder: str,
        model_extra: list,
        video_name: str,
        take_best: bool,
        pruning: Union[List[Tuple[int, float]], None],
        fine_tuning: Union[int, None],
        quality_metric: str,
        eval_metrics: List[str],
        grad_clipping: Union[float, None],
        batch_acc: Union[int, None],
    ):
        super().__init__()
        self.__start_time = None
        self.__report = None
        self.__quit_callback = quit_callback
        self.__criteria = criteria
        self.__dataset = dataset
        self.__training_params = training_params
        self.__tester = tester
        self.__topology_builder = topology_builder
        self.__model_extra = model_extra
        self.__batch_size = batch_size
        self.__video_name = video_name
        self.__pruning = pruning
        self.__fine_tuning = fine_tuning
        self.__take_best = take_best
        self.__quality_metric = quality_metric
        self.__eval_metrics = eval_metrics
        self.__grad_clipping = grad_clipping
        self.__batch_acc = batch_acc

        self.__lr_curve = Curve(("epoch", int), ("learning_rate", float))

        self.__base_learning_rate = training_params.optimizer_params["lr"]
        # Build git info as earliest as possible so it reflects the state at start.
        self.__git_info = build_git_info()

    @property
    def report(self):
        return self.__report

    def setup(self, trainer: pl.Trainer, pl_module: VideoNet, stage):
        self.__start_time = time.time()

    def teardown(self, trainer: pl.Trainer, pl_module: VideoNet, stage):
        # Look for status
        if self.__quit_callback.criteria_reached:
            status = "reach_criteria"
        else:
            status = "reach_last_epoch"

        # Make benchmark
        self.make_report(status, pl_module, True, trainer.current_epoch + 1)

    def __benchmark(self, pl_module: VideoNet) -> Benchmark:
        pl_module = pl_module.cuda()

        def cb(frame_id, frames_len):
            print(f"Frame {frame_id}/{frames_len - 1}", end="\r")

        benchmark = bench_model(
            self.__dataset,
            pl_module,
            batch=self.__batch_size,
            callback=cb,
            metrics=self.__eval_metrics,
        )
        print()

        return benchmark

    def make_report(self, status: str, pl_module: VideoNet, bench: bool, nb_epochs: int):
        # Compute time
        compute_time = time.time() - self.__start_time

        # Bench
        if bench:
            print("Benching...")
            benchmark = self.__benchmark(pl_module)
            print("test results:")
            print(benchmark.to_table())
        else:
            benchmark = None

        # Curves
        _headers = [("epoch", int), ("min", float), ("max", float), ("mean", float)]
        metric_curves = {}
        for epoch, test in self.__tester.items():
            for metric_name, data in test.items():
                if metric_name not in metric_curves:
                    metric_curves[metric_name] = Curve(*_headers)
                metric_curves[metric_name].append(epoch, *data.to_list())

        # Build the report
        self.__report = EncodingReport(
            timestamp=str(now_str()),
            model_name=pl_module.name,
            model_params=pl_module.to_meta_dict(),
            model_nb_parameters=pl_module.nb_params,
            model_extra=self.__model_extra,
            model_pattern=self.__topology_builder,
            training_params=self.__training_params.to_dict(),
            criteria=self.__criteria,
            status=status,
            compute_time=compute_time,
            benchmark=benchmark,
            epochs=nb_epochs,
            mac_address=get_mac_address(),
            command_line=" ".join(sys.argv),
            original_video_path=self.__dataset.video_loader.file_path,
            metric_curves=metric_curves,
            learning_rate_curve=self.__lr_curve,
            model_resume=str(pl_module),
            git_info=self.__git_info,
            batch_size=self.__batch_size,
            video_name=self.__video_name,
            take_best=self.__take_best,
            pruning=self.__pruning,
            fine_tuning=self.__fine_tuning,
            quality_metric=self.__quality_metric,
            grad_clipping=self.__grad_clipping,
            batch_acc=self.__batch_acc,
        )

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: VideoNet):
        scheduler = pl_module.scheduler
        if isinstance(scheduler, CustomScheduler):
            abs_lr = self.__base_learning_rate * scheduler.learning_rate_ratio
            self.__lr_curve.append(trainer.current_epoch, abs_lr)
