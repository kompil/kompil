import torch
import pytorch_lightning as pl

from typing import Dict
from kompil.nn.models.model import VideoNet
from kompil.data.data import VideoDataset, SequenceProbaCutoff
from kompil.profile.bench import BenchmarkEpochStorage, Benchmark
from kompil.profile.profiling_callbacks import OnBenchmarkCallback
from kompil.train.optimizers import CustomScheduler
from kompil.metrics.metrics import metric_higher_is_better
from kompil.utils.numbers import to_scale
from kompil.nn.layers.prune import (
    prune,
    recursive_unprune,
    recursive_prune_step,
    recursive_save_params,
    count_prunable_params,
)


class RobinHoodCallback(OnBenchmarkCallback):
    def __init__(
        self,
        tester: BenchmarkEpochStorage,
        dataset: VideoDataset,
        cutoff_ratio: float,
        epoch_step: int = 1,
        start_epoch: int = 0,
    ):
        assert isinstance(dataset, VideoDataset)
        assert cutoff_ratio > 0.0
        assert cutoff_ratio < 1.0

        super().__init__(tester=tester, epoch_step=epoch_step, first_epoch=start_epoch)
        self.tester = tester
        self.dataset = dataset

        # Init sequence selector
        self.sequence_selector = SequenceProbaCutoff()
        self.sequence_selector.set_cutoff_ratio(cutoff_ratio, len(dataset))
        self.sequence_selector.set_equal_weights(len(dataset))

    def on_benchmark(self, trainer: pl.Trainer, model: VideoNet, benchmark: Benchmark):
        # Get weights
        probs = 1 / benchmark.psnr.data.cpu().numpy()
        probs /= probs.sum()

        # Apply sequence selector
        self.dataset.set_sequence_selector(self.sequence_selector)
        self.sequence_selector.set_weights(probs)


class ArouseWeightsCallback(pl.callbacks.Callback):
    def __init__(
        self, epoch_step: int = 1, start_epoch: int = 0, last_epoch: int = 3000, std: float = 0.02
    ):
        super().__init__()

        self.__epoch_step = epoch_step
        self.__start_epoch = start_epoch
        self.__last_epoch = last_epoch
        self.__std = std

    def set_epoch_step(self, epoch_step: int):
        self.__epoch_step = epoch_step

    def on_epoch_end(self, trainer: pl.Trainer, model: VideoNet):
        # Pass if not triggered by epoch
        if trainer.current_epoch < self.__start_epoch or trainer.current_epoch > self.__last_epoch:
            return

        # Pass if not amongs tests epochs
        if trainer.current_epoch % self.__epoch_step != 0:
            return

        def fn(t: torch.Tensor) -> torch.Tensor:
            gauss = torch.normal(torch.zeros_like(t), torch.ones_like(t) * self.__std)

            t.add_(gauss)

            return t

        model._apply(fn)


class PruningCallback(pl.callbacks.Callback):

    TYPE_TO_METHOD = {
        "identity": prune.Identity,
        "random_unstructured": prune.RandomUnstructured,
        "l1_unstructured": prune.L1Unstructured,
        "random_structured": prune.RandomStructured,
        "ln_structured": prune.LnStructured,
    }

    def __init__(
        self,
        pruning_type: str,
        pruning_dict: Dict[int, float],
        lottery_ticket: bool,
    ):
        super().__init__()

        self.__pruning_type = self.TYPE_TO_METHOD[pruning_type]
        self.__pruning_dict = pruning_dict
        self.__lottery_ticket = lottery_ticket
        self.__prunable_params = None
        self.__total_params = None
        self.__already_pruned_params = 0

    def on_train_start(self, trainer: pl.Trainer, model: VideoNet):
        # Save initial params for lottery tickets
        if self.__lottery_ticket:
            recursive_save_params(model)
        # Count params concerned by pruning
        prunable_params = count_prunable_params(model)
        total_params = sum(p.numel() for p in model.parameters())
        self.__prunable_params = prunable_params
        self.__total_params = total_params
        self.__already_pruned_params = 0
        # Verify we can prune enough param to make the total count

    def on_epoch_end(self, trainer: pl.Trainer, model: VideoNet):

        amount = self.__pruning_dict.get(trainer.current_epoch, None)

        if amount is None:
            return

        # If the amount is negatif, it applies the learning
        if amount < 0.0:
            print(f"removing every pruning from model.")
            recursive_unprune(model)
            self.__already_pruned_params = 0
            return

        # Adjust the amount to fit whats already pruned
        remaining_prunable = self.__prunable_params - self.__already_pruned_params
        absolute_prune_target = int(amount * self.__total_params)
        prune_target = absolute_prune_target / self.__prunable_params
        adjusted_amount = absolute_prune_target / remaining_prunable
        self.__already_pruned_params = remaining_prunable - absolute_prune_target

        # Report to the terminal
        print(f"Pruning:")
        print(f" - {amount * 100:0.1f}% of all parameters.")
        print(f" - {to_scale(absolute_prune_target)} parameters.")
        print(f" - {prune_target * 100:0.1f}% of prunable parameters.")
        print(f" - {adjusted_amount * 100:0.1f}% of remaining prunable parameters.")

        # Clip to the max
        if adjusted_amount > 1.0:
            print("WARNING: Not enough parameters to prune, pruning all.")
            adjusted_amount = 1.0

        # Apply the pruning
        recursive_prune_step(model, adjusted_amount, self.__pruning_type, self.__lottery_ticket)

    def on_train_end(self, trainer: pl.Trainer, model: VideoNet):
        recursive_unprune(model)

    def on_save_checkpoint(self, model: VideoNet, *args):
        recursive_unprune(model)


class FineTuningCallback(OnBenchmarkCallback):
    """
    This will save the best average psnr model to restore it at the end of the learning and fine
    tune it a little.
    """

    def __init__(
        self,
        tester: BenchmarkEpochStorage,
        epochs: int,
        start: int,
        quality_metric: str,
        epoch_step: int = 1,
        start_epoch: int = 0,
    ):
        super().__init__(tester=tester, epoch_step=epoch_step, first_epoch=start_epoch)
        self.__last_quality = 0.0
        self.__best_model = None
        self.__epochs = epochs
        self.__start = start
        self.__quality_metric = quality_metric
        self.__quality_hib = metric_higher_is_better(quality_metric)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: VideoNet):
        assert isinstance(
            pl_module.scheduler, CustomScheduler
        ), "Fine tuning is only allowed with custom schedulers."

    def on_epoch_end(self, trainer: pl.Trainer, model: VideoNet):
        super().on_epoch_end(trainer, model)
        # Safety to check the scheduler is still acting as a fine tuner
        scheduler: CustomScheduler = model.scheduler
        if trainer.current_epoch > self.__start:
            assert scheduler.fine_tuning
        # Act only at the right epoch
        if trainer.current_epoch != self.__start:
            return
        # Restore the best model found when the fine tuning kicks in.
        print(f"Fine tune starting for {self.__epochs} epochs")
        assert self.__best_model is not None
        model.restore(self.__best_model)
        # Set the scheduler flag
        scheduler.set_fine_tuning(self.__start)

    def on_benchmark(self, trainer: pl.Trainer, model: VideoNet, benchmark: Benchmark):
        # no need to store the best after fine tuning kicks in
        if trainer.current_epoch >= self.__start:
            return
        # Skip if not best
        quality = benchmark.get(self.__quality_metric).average
        if self.__best_model is not None:
            if self.__quality_hib and self.__last_quality >= quality:
                return
            if not self.__quality_hib and self.__last_quality <= quality:
                return
        self.__last_quality = quality
        # Save the current model by making a clean copy
        self.__best_model = model.clean_clone()
        if self.__best_model is None:
            raise RuntimeError("Failed to save best model, cannot fine tune")
