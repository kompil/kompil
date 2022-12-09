import os
import time
import torch
import pytorch_lightning as pl

from typing import Union
from torch.utils.tensorboard import SummaryWriter

from kompil.utils.dynamic_config import BaseDynamicConfig
from kompil.nn.models.model import VideoNet, model_save
from kompil.profile.bench import BenchmarkEpochStorage, Benchmark
from kompil.profile.criteria import Criteria
from kompil.train.optimizers import CustomScheduler
from kompil.metrics.metrics import metric_higher_is_better


class OnBenchmarkCallback(pl.callbacks.Callback):
    def __init__(self, tester: BenchmarkEpochStorage, epoch_step: int = 1, first_epoch: int = 0):
        super().__init__()
        self.__epoch_step = epoch_step
        self.__tester = tester
        self.__first_epoch = first_epoch

    def set_epoch_step(self, epoch_step: int):
        self.__epoch_step = epoch_step

    def on_epoch_end(self, trainer: pl.Trainer, model: VideoNet):
        # Pass if not triggered by epoch
        if trainer.current_epoch < self.__first_epoch:
            return
        # Pass if not amongs tests epochs
        if trainer.current_epoch % self.__epoch_step != 0:
            return
        # Run test
        benchmark = self.__tester.run_test(trainer.current_epoch, model)
        # The callbacks have been triggered, run them
        self.on_benchmark(trainer, model, benchmark)

    def on_benchmark(self, trainer: pl.Trainer, model: VideoNet, benchmark: Benchmark):
        """
        Callback called after a benchmark occured.
        """
        pass


class TensorboardCallback(OnBenchmarkCallback):
    def __init__(
        self, tester: BenchmarkEpochStorage, epoch_step: int, path_logs: str, exframes: list = []
    ):
        super().__init__(tester=tester, epoch_step=epoch_step)
        self.exframes = exframes
        self.writer = SummaryWriter(path_logs)

    def teardown(self, trainer, pl_module, stage):
        self.writer.flush()
        self.writer.close()
        del self.writer

    def on_epoch_end(self, trainer, model):
        super().on_epoch_end(trainer, model)

        m_introspection = model.introspection
        if m_introspection is not None:
            for key, value in m_introspection.items():
                self.writer.add_scalars(key, value, trainer.current_epoch)

        if model.scheduler:
            scheduler = model.scheduler
            if isinstance(scheduler, CustomScheduler):
                self.writer.add_scalars("scheduler", scheduler.to_dict(), trainer.current_epoch)

    def on_benchmark(self, trainer: pl.Trainer, model: VideoNet, benchmark: Benchmark):

        for name, bench in benchmark.items():
            conv = benchmark.KNOWN_CONVERSION.get(name, None)
            if conv is None:
                self.writer.add_scalars(name, bench.to_dict(), trainer.current_epoch)
            else:
                self.writer.add_scalars(conv[0], bench.to_dict(), trainer.current_epoch)

        if not self.exframes:
            return

        c, h, w = model.frame_shape

        tmp_video = torch.zeros(1, 60, c, h, w, device=model.device, dtype=model.dtype)

        for idx_img in self.exframes:
            if idx_img + 30 >= model.nb_frames:
                continue

            for id_frame, res_frame in enumerate(model.generate_video(idx_img - 30, idx_img + 30)):
                tmp_video[0, id_frame] = res_frame
                del res_frame

            self.writer.add_video(f"Predict/Videos/{idx_img}", tmp_video, fps=30)

        del tmp_video


class QuitCallback(OnBenchmarkCallback):
    def __init__(
        self,
        tester: BenchmarkEpochStorage,
        max_epochs: int,
        criteria: Criteria,
        config: BaseDynamicConfig,
        epoch_step: int,
    ):
        super().__init__(tester=tester, epoch_step=epoch_step)
        self.__max_epochs = max_epochs
        self.__criteria = criteria
        self.__criteria_reached = False
        self.__config_max_epochs = config.handle("max_epochs", max_epochs)

    @property
    def criteria_reached(self):
        return self.__criteria_reached

    def setup(self, trainer: pl.Trainer, pl_module: VideoNet, stage):
        super().setup(trainer, pl_module, stage)
        self.__criteria_reached = False

    def __read_max_epochs(self):
        max_epochs = self.__config_max_epochs.read()
        assert isinstance(max_epochs, int)
        return max_epochs

    def on_epoch_end(self, trainer, model):
        super().on_epoch_end(trainer, model)

        # Update max epoch based on dynamic config
        max_epochs = self.__read_max_epochs()
        if max_epochs != self.__max_epochs:
            print("New max epochs:", max_epochs)
        self.__max_epochs = max_epochs

        # Quit if the max epoch has been reached
        if trainer.current_epoch >= self.__max_epochs - 1:
            trainer.should_stop = True

    def on_benchmark(self, trainer: pl.Trainer, model: VideoNet, benchmark: Benchmark):
        if self.__criteria.active and self.__criteria.reached(benchmark):
            trainer.should_stop = True
            self.__criteria_reached = True


class SaveLastCallback(pl.callbacks.Callback):
    FOLDER = "/dev/shm/kompil_last"

    @classmethod
    def get_path(cls, model_name: str) -> str:
        return os.path.join(cls.FOLDER, f"{model_name}.last.pth")

    def __init__(self, epoch_step: int):
        super().__init__()
        os.makedirs(self.FOLDER, exist_ok=True)
        self.__epoch_step = epoch_step

    def on_epoch_end(self, trainer: pl.Trainer, model: VideoNet):
        # Pass if not amongs tests epochs
        if trainer.current_epoch == 0 or trainer.current_epoch % self.__epoch_step != 0:
            return
        # Save the model
        model_copy = model.clean_clone()
        if model_copy is not None:
            model_save(model_copy, self.get_path(model.name))


class SaveBestCallback(OnBenchmarkCallback):
    FOLDER = "/dev/shm/kompil_best"

    @classmethod
    def get_path(cls, model_name: str) -> str:
        return os.path.join(cls.FOLDER, f"{model_name}.best.pth")

    def __init__(
        self,
        tester: BenchmarkEpochStorage,
        epoch_step: int,
        save_on_disk: bool,
        quality_metric: str,
    ):
        super().__init__(tester=tester, epoch_step=epoch_step)
        self.__save_on_disk = save_on_disk
        self.__last_quality = 0.0
        self.__best_model = None
        self.__quality_metric = quality_metric
        self.__quality_hib = metric_higher_is_better(quality_metric)
        if self.__save_on_disk:
            os.makedirs(self.FOLDER, exist_ok=True)

    @property
    def best(self) -> Union[VideoNet, None]:
        """Get the best saved model"""
        return self.__best_model

    def on_benchmark(self, trainer: pl.Trainer, model: VideoNet, benchmark: Benchmark):
        # Skip if not best
        quality = benchmark.get(self.__quality_metric).average
        if self.__best_model is not None:
            if self.__quality_hib and self.__last_quality >= quality:
                return
            if not self.__quality_hib and self.__last_quality <= quality:
                return
        self.__last_quality = quality
        # Save best for progress bar
        model.update_progress_bar_dict({self.__quality_metric: quality})
        # Make a clean copy of the model
        best_model = model.clean_clone()
        # If the clean copy failed don't do anything
        if best_model is None:
            return
        # Save the best model for later iterations
        self.__best_model = best_model
        # If required, save it on disk
        if self.__save_on_disk:
            model_save(self.__best_model, self.get_path(model.name))


class RemainingTimeEstCallback(pl.callbacks.Callback):
    def __init__(self, max_epochs: int):
        super().__init__()
        self.__start_training_time = None
        self.__max_epochs = max_epochs

    def on_train_start(self, trainer: pl.Trainer, model: VideoNet):
        self.__start_training_time = time.time()

    def on_epoch_end(self, trainer: pl.Trainer, model: VideoNet):
        # Don't show if useless
        if trainer.max_epochs is None:
            return
        if trainer.current_epoch < 20:
            return
        # Estimate the remaining time
        time_spent = time.time() - self.__start_training_time
        remaining_epochs = self.__max_epochs - trainer.current_epoch
        remaining = time_spent * remaining_epochs / trainer.current_epoch
        # Format to string
        formatted = ""
        if remaining > 3600:
            hour = remaining // 3600
            remaining = remaining % 3600
            formatted += f"{int(hour):d}h:"
        if remaining > 60:
            minutes = remaining // 60
            remaining = remaining % 60
            formatted += f"{int(minutes)}min:"
        formatted += f"{int(remaining)}s"
        # Save time for progress bar
        model.update_progress_bar_dict({"remaining": formatted})
