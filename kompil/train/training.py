import json
import numpy

from kompil.train.optimizers import ADAM_OPT
from kompil.train.loss.base import factory as loss_factory


class TrainingParams:
    def __init__(
        self,
        loss_name: str,
        loss_params: dict = None,
        optimizer_name: str = ADAM_OPT,
        optimizer_params: dict = None,
        scheduler_name: str = None,
        scheduler_params: dict = None,
    ):
        self.loss_name = loss_name
        self.loss_params = loss_params if loss_params else {}
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params if scheduler_params else {}

    def to_dict(self) -> dict:
        return {
            "loss": self.loss_name,
            "loss_params": self.loss_params,
            "optimizer": self.optimizer_name,
            "optimizer_params": self.optimizer_params,
            "scheduler": self.scheduler_name,
            "scheduler_params": self.scheduler_params,
        }

    @staticmethod
    def from_dict(data: dict) -> "TrainingParams":
        return TrainingParams(
            loss_name=data.get("loss"),
            loss_params=data.get("loss_params"),
            optimizer_name=data.get("optimizer"),
            optimizer_params=data.get("optimizer_params"),
            scheduler_name=data.get("scheduler"),
            scheduler_params=data.get("scheduler_params"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def __str__(self):
        return f"TrainingParams<{self.loss_name}, {self.loss_params}, {self.optimizer_name}, {self.optimizer_params}, {self.scheduler_name}, {self.scheduler_params}>"

    @staticmethod
    def __params_to_repr(params):
        text = str()
        for key, value in params.items():
            text += f"{key}={value} "
        return text

    @property
    def description(self) -> str:
        return str(
            f"loss: {self.loss_name} {self.__params_to_repr(self.loss_params)}\n"
            f"optimizer: {self.optimizer_name} {self.__params_to_repr(self.optimizer_params)}\n"
            f"scheduler: {self.scheduler_name} {self.__params_to_repr(self.scheduler_params)}\n"
        )

    @staticmethod
    def load(file_path: str):
        with open(file_path, "r") as f:
            content = json.load(f)
            loss_name = content["loss"]
            loss_params = content["loss_params"]
            optimizer_name = content["optimizer"]
            optimizer_params = content["optimizer_params"]
            scheduler_name = content["scheduler"]
            scheduler_params = content["scheduler_params"]

        params = TrainingParams(
            loss_name,
            loss_params,
            optimizer_name,
            optimizer_params,
            scheduler_name,
            scheduler_params,
        )

        return params


class TrainingStep:
    def __init__(self, epoch: int, last_notes: list):
        assert epoch

        self.epoch = epoch
        self.last_notes = last_notes

    def __str__(self):
        return f"TrainingStep<{self.epoch}, {self.last_notes}>"


class ActivableTrainingParams:
    def __init__(self, params: TrainingParams, activation, minimal_epoch: int = 100):
        assert activation
        assert params

        self.minimal_epoch = max(2, minimal_epoch)
        self.params = params
        self.activation = activation

    def valid(self, training_step: TrainingStep) -> bool:
        if training_step.epoch < self.minimal_epoch:
            return False

        return self.activation(training_step)

    def __str__(self):
        return f"ActivableTrainingParams<{self.params}, {self.minimal_epoch}, {self.activation.__name__}>"

    @staticmethod
    def loss_rstd_activation(training_step: TrainingStep, **kwargs) -> bool:
        losses = [i[0] for i in training_step.last_notes]
        # Relative standart deviation
        return (
            numpy.std(losses) / (numpy.mean(losses) + 1e-4) < 0.04
        )  # TODO: parametrable activation

    @staticmethod
    def psnr_rstd_activation(training_step: TrainingStep, **kwargs) -> bool:
        psnrs = [i[1] for i in training_step.last_notes]
        # Relative standart deviation
        return numpy.std(psnrs) / (numpy.mean(psnrs) + 1e-4) < 0.04  # TODO: parametrable activation


# v1 with dummy parsing for easy parameters
# Input format ==> psnr/adapt,200/adapt,epoch:1000/bpsnr,bound:34,epoch:200/bpsnr,bound:36
# TODO: Use an official format like json for futher need
def build_activable_training_params(args: str) -> list:
    atp = []
    assert args

    splitted_training_params = args.split("/")  # split atp

    for training_params in splitted_training_params:
        if not training_params:
            continue

        loss_params = training_params.split(",")  # split atp params

        nb_loss_params = len(loss_params)

        assert nb_loss_params > 0

        atp_loss_name = loss_params[0]

        assert loss_factory().has(atp_loss_name)  # make sure the loss exists

        atp_loss_params = {}
        atp_epoch = 500  # default minimal epoch for activation

        for i in range(1, nb_loss_params):  # iterate over params
            loss_param = loss_params[i]

            if not loss_param:
                continue

            splitted_param = loss_param.split(":")

            if splitted_param[0] == "epoch":
                assert splitted_param[1]
                atp_epoch = int(splitted_param[1])
            elif len(splitted_param) == 1:  # if no name is specified ==> atp epoch
                atp_epoch = int(splitted_param[0])
            else:
                arg_name = splitted_param[0]
                arg_val = splitted_param[1]
                assert arg_name
                assert float(arg_val)
                atp_loss_params[arg_name] = float(arg_val)

        atp.append(
            ActivableTrainingParams(
                params=TrainingParams(loss_name=atp_loss_name, loss_params=atp_loss_params),
                activation=ActivableTrainingParams.psnr_rstd_activation,  # default activation for now
                minimal_epoch=atp_epoch,
            )
        )

    return atp
