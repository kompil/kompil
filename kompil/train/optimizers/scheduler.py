import abc
import math
import torch
import numpy
import queue
import random
from typing import Union, List, Optional

from kompil.utils.factory import Factory
from kompil.utils.dynamic_config import BaseDynamicConfig


__FACTORY = Factory("scheduler")


def factory() -> Factory:
    global __FACTORY
    return __FACTORY


def register_scheduler(name: str):
    return factory().register(name)


class SchedulerItems:
    def __init__(
        self,
        nb_frames: int,
        tester,
        dyn_conf: BaseDynamicConfig,
        learning_epochs: int,
        fine_tuning_epochs: Optional[int],
    ):
        self.nb_frames = nb_frames
        self.tester = tester
        self.dyn_conf = dyn_conf
        self.learning_epochs = learning_epochs
        self.fine_tuning_epochs = fine_tuning_epochs


def build_scheduler(
    name: str,
    optimizer,
    items: SchedulerItems,
    params: dict,
):
    return factory()[name](optimizer, items, **params)


@register_scheduler("step")
def step_scheduler(optimizer, items: SchedulerItems, gamma=0.5, step_size=200, verbose=False):
    return torch.optim.lr_scheduler.StepLR(
        optimizer, gamma=gamma, step_size=step_size, verbose=verbose
    )


@register_scheduler("multistep")
def multistep_scheduler(
    optimizer, items: SchedulerItems, gamma=0.5, milestones=[350, 400, 450, 500, 550], verbose=False
):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        gamma=gamma,
        milestones=milestones,
        verbose=verbose,
    )


class CustomScheduler(torch.optim.lr_scheduler.LambdaLR, metaclass=abc.ABCMeta):
    def __init__(self, optimizer, items: SchedulerItems, verbose=False):
        self.__dynamic_config = items.dyn_conf.handle("value", "scheduler")
        self.__current_lrr = 1.0
        self.__last_update_epoch = -1
        self.__fine_tuning = None
        super().__init__(optimizer, lr_lambda=self.__callback, last_epoch=-1, verbose=verbose)

    def __read_dynamic_value(self) -> Union[None, float]:
        from_remote = self.__dynamic_config.read()
        if from_remote != "scheduler":
            if isinstance(from_remote, int):
                return float(from_remote)

            if isinstance(from_remote, float):
                return from_remote

            print(f"Warning, {self.__dyn_conf} bad value. Resetting to scheduler...")
            self.__dyn_conf.write("scheduler")

        return None

    def __callback(self, epoch: int) -> float:
        # BUG ? Pytorch call twice the scheduler by epoch
        # Add protection
        if epoch == self.__last_update_epoch:
            return self.__current_lrr

        self.__last_update_epoch = epoch
        self.__current_lrr = self.__update(epoch)
        return self.__current_lrr

    def __update(self, epoch: int) -> float:
        # We need to make sure the LR is updated anyway to keep momentum.
        std_lr = self.update_learning_rate(epoch)

        # Manual learning rate is P0
        dyn_value = self.__read_dynamic_value()

        if dyn_value is not None:
            return dyn_value

        # Then comes the learning rate from the scheduler
        return std_lr

    def set_fine_tuning(self, start_epoch: int):
        self.__fine_tuning = start_epoch

    def stop_fine_tuning(self):
        self.__fine_tuning = None

    def to_dict(self):
        return {"learning_rate": self.__current_lrr}

    @property
    def learning_rate_ratio(self):
        return self.__current_lrr

    @property
    def fine_tuning(self) -> bool:
        return self.__fine_tuning is not None

    @property
    def fine_tuning_start_epoch(self) -> int:
        return self.__fine_tuning

    @abc.abstractmethod
    def update_learning_rate(self, epoch: int) -> float:
        pass


@register_scheduler("none")
class NoneScheduler(CustomScheduler):
    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        ft_high_lr=0.3,
        ft_low_lr=0.1,
        verbose=False,
    ):
        self.__ft_high_lr = ft_high_lr
        self.__ft_low_lr = ft_low_lr
        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        if not self.fine_tuning:
            return 1.0

        epoch_ratio = (epoch - self.fine_tuning_start_epoch) / self.fine_tuning_start_epoch
        current_lr = self.__ft_high_lr - (self.__ft_high_lr - self.__ft_low_lr) * epoch_ratio
        return current_lr


@register_scheduler("vuality")
class VualityScheduler(CustomScheduler):
    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        target: tuple = None,
        mul_factor: tuple = (0.26, 0.275, 0.2, 0.265),
        min_lr: float = 0.01,
        verbose=False,
    ):
        if target is None:
            base_lr = optimizer.param_groups[0]["lr"]
            target = (34, 41, int(5 / base_lr), 1)

        self.__tester = items.tester
        self.__target = target
        self.__min_lr = min_lr
        self.__last_lr = 1.0
        self.__mul_factor = mul_factor
        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:

        # Auto behavior
        last_bench = self.__tester.last

        if last_bench is None:
            return 1.0

        self.__last_lr = min(
            self.__last_lr,
            max(
                1
                - (
                    self.__mul_factor[3] * min(1, last_bench.ssim.mean / self.__target[3])
                    + self.__mul_factor[0] * min(1, last_bench.psnr.min / self.__target[0])
                    + self.__mul_factor[1] * min(1, last_bench.psnr.mean / self.__target[1])
                    + self.__mul_factor[2] * min(1, epoch / self.__target[2])
                ),
                self.__min_lr,
            ),
        )

        return self.__last_lr


@register_scheduler("tycho")
class TychoScheduler(CustomScheduler):
    """
    Scheduler that continuously decay according to a factor, but can accelerate if video quality is
    higher than expected.
    """

    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        target_mean_psnr=37.5,
        progression_min=0.1,
        progression_pow=2.0,
        decay_factor=0.9995,
        verbose=False,
    ):
        self.__tester = items.tester
        self.__last_lr_ratio = 1.0

        self.__target_mean_psnr = target_mean_psnr
        self.__progression_min = progression_min
        self.__progression_pow = progression_pow
        self.__decay_factor = decay_factor

        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        # Find the updated value
        new_lr = self.__update_lr(self.__tester, self.__last_lr_ratio)
        # Take the min to ensure it won't be higher
        self.__last_lr_ratio = min(self.__last_lr_ratio, new_lr)
        return self.__last_lr_ratio

    def __update_lr(self, tester, last_lr) -> float:
        # Calculate progression toward target.
        last_bench = tester.last
        progression = 0.0
        if last_bench is not None:
            progression = last_bench.psnr.mean / self.__target_mean_psnr
            # Progression must not be superior to 1
            progression = min(1.0, progression)
            # The pow factor is to compensate for the logarithmique behavior of the PSNR
            progression = math.pow(progression, self.__progression_pow)
        lr_progression = 1.0 - progression * (1.0 - self.__progression_min)
        # calculate lr based on a decay factor
        lr_epoch = last_lr * self.__decay_factor
        # Next learning rate is the minimal between both
        return min(lr_epoch, lr_progression)


@register_scheduler("tycho-vmaf")
class TychoVMAFScheduler(CustomScheduler):
    """
    Scheduler that continuously decay according to a factor, but can accelerate if video quality is
    higher than expected.
    """

    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        vmaf_start=75.0,
        progression_min=0.1,
        decay_factor=0.99955,
        verbose=False,
    ):
        self.__tester = items.tester
        self.__last_lr_ratio = 1.0

        self.__progression_min = progression_min
        self.__vmaf_start = vmaf_start
        self.__decay_factor = decay_factor

        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        # Find the updated value
        new_lr = self.__update_lr(self.__tester, self.__last_lr_ratio)
        # Take the min to ensure it won't be higher
        self.__last_lr_ratio = min(self.__last_lr_ratio, new_lr)
        return self.__last_lr_ratio

    def __update_lr(self, tester, last_lr) -> float:
        # Calculate progression toward target.
        last_bench = tester.last
        progression = 0.0
        if last_bench is not None:
            progression = (last_bench.vmaf.mean - self.__vmaf_start) / (100.0 - self.__vmaf_start)
            # Progression must be between 0 and 1.0
            progression = max(0.0, min(1.0, progression))
        lr_progression = max(self.__progression_min, 1.0 - progression)
        # calculate lr based on a decay factor
        lr_epoch = last_lr * self.__decay_factor
        # Next learning rate is the minimal between both
        return min(lr_epoch, lr_progression)


@register_scheduler("tarkin")
class TarkinScheduler(CustomScheduler):
    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        verbose=False,
    ):
        self.__tester = items.tester
        self.__min_lr = 0.05
        self.__max_lr = 2
        self.__last_lr = 1.0
        self.__prev_bench = None
        self.__decay = 0.99995
        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        last_bench = self.__tester.last

        if last_bench is None:
            return self.__last_lr

        if self.__prev_bench is None:
            self.__prev_bench = last_bench
            return self.__last_lr

        if epoch % 10 != 0:
            return self.__last_lr

        vmaf_diff = last_bench.vmaf.mean - self.__prev_bench.vmaf.mean
        psnr_diff = last_bench.psnr.mean - self.__prev_bench.psnr.mean

        lr = self.__last_lr + (
            max(min(vmaf_diff * 0.1, 0.075), -0.1)
            + max(min(psnr_diff * 0.25, 0.075), -0.1)
            - 1 / (vmaf_diff**2 + psnr_diff**2 + 2000)
        )

        lr *= self.__decay

        self.__last_lr = min(max(lr, self.__min_lr), self.__max_lr)

        self.__prev_bench = last_bench

        return self.__last_lr


@register_scheduler("forge")
class StarForgeScheduler(CustomScheduler):
    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        verbose=False,
    ):
        # Scheduler with dynamic learning rate ratio adaptation based
        # on Adam optimization : https://arxiv.org/pdf/1412.6980.pdf

        self.__tester = items.tester
        self.__min_lr = 5e-2
        self.__max_lr = 2
        self.__last_lr = 1.0
        self.__m = 0
        self.__v = 0
        self.__b1 = 0.9
        self.__b2 = 0.999
        self.__alpha = 2.5e-2
        self.__eps = 1e-8
        self.__belief = True
        self.__prev_bench = None
        self.__decay = 0.99995
        self.__sinus_on_decay = False
        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        last_bench = self.__tester.last

        if last_bench is None:
            return self.__last_lr

        if self.__prev_bench is None:
            self.__prev_bench = last_bench
            return self.__last_lr

        if epoch % 10 != 0:
            return self.__last_lr

        psnr_diff = max(min(self.__prev_bench.psnr.mean - last_bench.psnr.mean, 0.5), -0.5)
        vmaf_diff = max(min(self.__prev_bench.vmaf.mean - last_bench.vmaf.mean, 0.5), -0.5)
        stagnation_malus = min(2e-5 / (psnr_diff**2 + (vmaf_diff / 5) ** 2 + self.__eps), 0.1)

        # Gradient is based as : negative => go ahend / positive => not the good direction
        # Second part of the equation is to add a positive gradient value if there is a quality stagnation
        # ==> LR must be reduced
        grad = psnr_diff + stagnation_malus
        self.__m = self.__b1 * self.__m + (1 - self.__b1) * grad

        if self.__belief:
            # AdaBelief version : https://juntang-zhuang.github.io/adabelief/
            self.__v = self.__b2 * self.__v + (1 - self.__b2) * (grad - self.__m) ** 2 + self.__eps
        else:
            self.__v = self.__b2 * self.__v + (1 - self.__b2) * grad**2

        mt = self.__m / (1 - self.__b1**epoch)
        vt = self.__v / (1 - self.__b2**epoch)
        lr = self.__last_lr - self.__alpha * (mt / (math.sqrt(vt) + self.__eps))

        if lr < self.__last_lr and self.__sinus_on_decay:
            sinus = math.sin(epoch / 10.0) / 50
            lr += sinus

        lr *= self.__decay

        self.__last_lr = min(max(lr, self.__min_lr), self.__max_lr)

        self.__prev_bench = last_bench

        return self.__last_lr


@register_scheduler("mir")
class MirScheduler(CustomScheduler):
    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        verbose=False,
    ):
        # Mir will natively up learning rate ratio but it will be
        # balanced with high lr ratio, epoch and average psnr stability

        self.__tester = items.tester
        self.__min_lr = 0.05
        self.__max_lr = 2
        self.__max_epoch = items.learning_epochs
        self.__last_lr = 1.0

        self.__global_up = 0.18  # Up value at each call
        self.__epoch_ratio = 0.225  # Epoch ratio is effective
        self.__lr_ratio = 0.143  # Naturally add weight for high ratio value
        self.__stab_ratio = 0.25  # Stabilization is based on standart deviation on last avg psnr
        self.__random_ratio = 0.1  # To add some fun in that scheduler

        self.__hist_size = 4
        self.__prev_psnr = queue.Queue(maxsize=self.__hist_size)
        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        last_bench = self.__tester.last

        if last_bench is None:
            return self.__last_lr

        if epoch % 10 != 0:
            return self.__last_lr

        if self.__prev_psnr.full():
            self.__prev_psnr.get()

        self.__prev_psnr.put(last_bench.psnr.mean)

        if epoch >= 100:
            list_p = list(self.__prev_psnr.queue)
            stability_weight = min(numpy.std(list_p), 1)
        else:
            stability_weight = 0

        epoch_weight = epoch / self.__max_epoch
        lr_weight = self.__last_lr / self.__max_lr
        random_weight = random.randrange(0, 10) / 10.0

        lr = (
            self.__last_lr
            + self.__global_up
            + self.__random_ratio * random_weight
            - self.__epoch_ratio * epoch_weight
            - self.__lr_ratio * lr_weight
            - self.__stab_ratio * stability_weight
        )

        self.__last_lr = min(max(lr, self.__min_lr), self.__max_lr)

        return self.__last_lr


@register_scheduler("saliout")
class SalioutScheduler(CustomScheduler):
    def __init__(
        self,
        optimizer,
        items: SchedulerItems,
        verbose=False,
    ):
        # Mir based scheduler, with some modifications :
        # - Aranged ratios
        # - Add pseudo a adam weight to keep mean psnr growing
        # - Minimal lr : 0.05 ==> 0.075
        # - lr weight : linear ==> pseudo-exponential based to avoid to high lr

        self.__tester = items.tester
        self.__min_lr = 0.075
        self.__max_lr = 2
        self.__max_epoch = items.learning_epochs
        self.__last_lr = 1.0

        self.__global_up = 0.18
        self.__epoch_ratio = 0.225
        self.__lr_ratio = 0.143
        self.__stab_ratio = 0.175
        self.__random_ratio = 0.1
        self.__adam_ratio = 0.107

        self.__m = 0
        self.__v = 0
        self.__b1 = 0.9
        self.__b2 = 0.99
        self.__eps = 1e-8

        self.__prev_psnr = queue.Queue(maxsize=4)

        super().__init__(optimizer, items=items, verbose=verbose)

    def update_learning_rate(self, epoch: int) -> float:
        last_bench = self.__tester.last

        if last_bench is None:
            return self.__last_lr

        if epoch % 10 != 0:
            return self.__last_lr

        if self.__prev_psnr.full():
            self.__prev_psnr.get()

        self.__prev_psnr.put(last_bench.psnr.mean)

        stability_weight = 0
        adam_weight = 0

        if epoch >= 50:
            list_psnr = list(self.__prev_psnr.queue)
            grad = max(min(list_psnr[-2] - list_psnr[-1], 0.25), -0.25)
            self.__m = self.__b1 * self.__m + (1 - self.__b1) * grad
            self.__v = self.__b2 * self.__v + (1 - self.__b2) * grad**2

            mt = self.__m / (1 - self.__b1**epoch)
            vt = self.__v / (1 - self.__b2**epoch)

            adam_weight = mt / (math.sqrt(vt) + self.__eps)

            if epoch >= 100:
                stability_weight = min(numpy.std(list_psnr), 1)

        epoch_weight = epoch / self.__max_epoch
        lr_weight = (self.__last_lr * math.exp(self.__last_lr / 8.5)) / self.__max_lr
        random_weight = random.randrange(0, 10) / 10.0

        lr = (
            self.__last_lr
            + self.__global_up
            + self.__random_ratio * random_weight
            - self.__epoch_ratio * epoch_weight
            - self.__lr_ratio * lr_weight
            - self.__stab_ratio * stability_weight
            - self.__adam_ratio * adam_weight
        )

        self.__last_lr = min(max(lr, self.__min_lr), self.__max_lr)

        return self.__last_lr
