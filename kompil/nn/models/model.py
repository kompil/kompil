import copy
import torch
import pytorch_lightning as pl

from typing import Tuple, Union, Optional, Dict, Any, List
from torch.utils.data import IterableDataset

from kompil.nn.topology.builder import build_topology_from_list
from kompil.train.loss.base import factory as loss_factory
from kompil.train.optimizers import build_scheduler, get_optimizer, SchedulerItems
from kompil.train.training import TrainingParams
from kompil.data.timeline import create_timeline
from kompil.utils.colorspace import convert_to_colorspace


class VideoNet(pl.LightningModule):
    VERSION = 1

    def __init__(
        self,
        name: str,
        nb_frames: int,
        fps: int,
        frame_shape: Tuple[int, int, int],  # c,h,w
        topology: list,
        colorspace: str,
        training_params: TrainingParams = None,
        checkpoint_data=None,
    ):
        super().__init__()

        # Meta info
        self.name = name
        self._nb_frames = nb_frames
        self._fps = fps
        self._frame_shape = frame_shape
        self.nb_outputs = frame_shape[0] * frame_shape[1] * frame_shape[2]
        self.topology = topology
        self._colorspace = colorspace
        self.__progress_bar_dict = {}

        self.training_params = None
        self.loss_lambda = None
        self.__checkpoint_data = None

        self.__optimizer = None
        self.__scheduler = None
        self.__scheduler_items = None

        if training_params or checkpoint_data:
            self._configure_training(training_params, checkpoint_data)

        self.inference_context = dict()
        self.sequence, _ = build_topology_from_list(
            sequence=self.topology,
            context=self.inference_context,
        )

    def set_scheduler_items(self, items: SchedulerItems):
        self.__scheduler_items = items

    def update_progress_bar_dict(self, pg: dict):
        self.__progress_bar_dict.update(pg)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.update(self.__progress_bar_dict)
        return items

    def clone(self) -> "VideoNet":
        model_copy = VideoNet(
            name=self.name,
            nb_frames=self.nb_frames,
            fps=self.fps,
            frame_shape=self.frame_shape,
            topology=self.topology,
            colorspace=self.colorspace,
            training_params=self.training_params,
            checkpoint_data=self.__checkpoint_data,
        )
        model_copy.__optimizer = copy.deepcopy(self.__optimizer)
        model_copy.__scheduler = self.__scheduler
        model_copy.sequence = copy.deepcopy(self.sequence)
        model_copy.loss_lambda = self.loss_lambda
        return model_copy

    def clean_clone(self) -> "VideoNet":
        """
        Clone the model and remove all the imperfections such as pruning and weight normalisation.
        """
        from kompil.nn.layers.prune import recursive_unprune
        from kompil.nn.layers.weightnorm import rm_norm_weight_norm_layers

        try:
            model_copy = self.clone()
            rm_norm_weight_norm_layers(model_copy)
            recursive_unprune(model_copy)
            return model_copy
        except Exception as e:
            print(f"/!\\ Failed to copy model: {e}")

    def restore(self, other: "VideoNet"):
        """Restore the model at another saved point"""
        assert self.name == other.name
        assert self._nb_frames == other._nb_frames
        assert self._fps == other._fps
        assert self._frame_shape == other._frame_shape
        assert self.nb_outputs == other.nb_outputs
        assert self.topology == other.topology
        assert self._colorspace == other._colorspace
        assert self.training_params == other.training_params
        assert self.loss_lambda == other.loss_lambda
        self.__checkpoint_data = other.__checkpoint_data
        self.sequence.load_state_dict(other.sequence.state_dict())
        self.__optimizer.load_state_dict(other.__optimizer.state_dict())

    @property
    @torch.jit.unused
    def full_name(self):
        return f"{self.name}_{self.nb_frames}f_{self.nb_params}p"

    @property
    @torch.jit.unused
    def nb_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    @torch.jit.unused
    def fps(self):
        return self._fps

    @property
    @torch.jit.unused
    def frame_shape(self):
        return self._frame_shape

    @property
    @torch.jit.unused
    def nb_frames(self):
        return self._nb_frames

    @property
    @torch.jit.unused
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    @property
    @torch.jit.unused
    def colorspace(self):
        return self._colorspace

    @property
    @torch.jit.unused
    def scheduler(self):
        return self.__scheduler

    @property
    @torch.jit.unused
    def introspection(self) -> Optional[Dict[str, Any]]:
        return None

    @torch.jit.export
    def get_meta(self):
        # Method name is parsed "as it" in Android side, don't change it
        # Needed for torchscript conversion in order to keep meta data

        # /!\ if put double underscore *__* as prefix, the torchscript conversion      /!\
        # /!\ won't keep the attribute and you will get a RuntimeError "missing attribute" /!\
        return self._fps, self._frame_shape, self._nb_frames, self._colorspace

    def forward(self, x: torch.Tensor):
        # Initialize default context
        self.inference_context.clear()
        self.inference_context["frames_idx"] = x.long()
        # Run model
        return self.sequence(x)

    def forward_rgb8(self, x: torch.Tensor):
        frame = self.forward(x)
        return convert_to_colorspace(frame, src=self.colorspace, dst="rgb8")

    @torch.jit.unused
    def training_step(self, batch, batch_idx):
        x, y_ref = batch

        y_pred = self.forward(x)

        return self.loss_lambda(y_pred, y_ref, context=self.inference_context)

    @torch.jit.unused
    def configure_optimizers(self):
        """Required by pytorch-lightning"""
        self.__optimizer = None
        self.__scheduler = None

        if not self.training_params:
            return

        self.__optimizer = get_optimizer(
            self.training_params.optimizer_name,
            self.parameters(),
            self.training_params.optimizer_params,
        )

        if self.__optimizer_state:
            self.__optimizer.load_state_dict(self.__optimizer_state)

        if not self.training_params.scheduler_name:
            return self.__optimizer

        assert (
            self.__scheduler_items is not None
        ), "Set scheduler items first via method set_scheduler_items"

        self.__scheduler = build_scheduler(
            self.training_params.scheduler_name,
            self.__optimizer,
            items=self.__scheduler_items,
            params=self.training_params.scheduler_params,
        )

        return [self.__optimizer], [self.__scheduler]

    @torch.jit.unused
    def _configure_training(self, training_params: TrainingParams, checkpoint_data):
        assert training_params or checkpoint_data
        assert training_params.loss_name

        self.training_params = training_params
        self.__checkpoint_data = checkpoint_data
        self.loss_lambda = loss_factory()[training_params.loss_name]

        self.__optimizer_state = None
        if checkpoint_data is not None:
            self.__optimizer_state = checkpoint_data["optimizer_state_dict"]

        self.__checkpoint_data = checkpoint_data

        assert self.loss_lambda

    @torch.jit.unused
    def prepare_training(self, dataset: IterableDataset):
        """In-house preparation"""
        provided_params = self.training_params.loss_params
        provided_params = provided_params if provided_params else {}

        self.loss_lambda.prepare(dataset, **provided_params)

    @torch.jit.unused
    def to_meta_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.frame_shape[2],
            "height": self.frame_shape[1],
            "channels": self.frame_shape[0],
            "fps": self.fps,
            "nb_frames": self.nb_frames,
            "topology": self.topology,
            "colorspace": self.colorspace,
        }

    @torch.jit.unused
    def generate_video(self, start: Union[int, None] = None, stop: Union[int, None] = None):
        self.eval()

        start = start if start else 0
        stop = stop if stop else self.nb_frames - 1

        time_frame = create_timeline(start, stop + 1, device=self.device)

        with torch.no_grad():
            for id_frame in range(start, stop + 1):
                res_frame = self.forward_rgb8(time_frame[id_frame].unsqueeze(0))[0]
                yield torch.clamp(res_frame, min=0.0, max=1.0)

    @torch.jit.unused
    def run_once(self, callback: Optional[callable] = None):
        self.eval()
        time_frame = create_timeline(self.nb_frames, device=self.device)
        with torch.no_grad():
            for id_frame in range(self.nb_frames):
                img = self.forward(time_frame[id_frame].unsqueeze(0))
                if callback is not None:
                    callback(frame=id_frame, image=img)

    @property
    @torch.jit.unused
    def checkpoint_epoch(self) -> Union[int, None]:
        """
        Get the epoch from last checkpoint. None if it was not a checkpoint.
        """
        if self.__checkpoint_data is None:
            return None
        return self


def meta_to_protomodel(meta: dict, training_params, checkpoint) -> VideoNet:
    # Read meta
    video_width = meta["width"]
    video_height = meta["height"]
    video_channels = meta["channels"]
    nb_frames = meta["nb_frames"]
    topology = meta["topology"]
    fps = meta["fps"]
    name = meta["name"]
    colorspace = meta.get("colorspace", "rgb8")

    # Create model
    model = VideoNet(
        name=name,
        nb_frames=nb_frames,
        fps=fps,
        frame_shape=(video_channels, video_height, video_width),
        topology=topology,
        training_params=training_params,
        checkpoint_data=checkpoint,
        colorspace=colorspace,
    )

    # Disable batch norm / drop out modules...
    model.eval()

    return model


def model_save(model: VideoNet, file_path: str):
    """
    Standard model save procedure
    """
    data = {
        "version": VideoNet.VERSION,
        "model_state_dict": model.state_dict(),
        "model_meta_dict": model.to_meta_dict(),
    }
    torch.save(data, file_path)


def model_load_from_data(data: dict) -> VideoNet:
    version = data.get("version")

    assert version in [VideoNet.VERSION], f"packed version {version} not supported"

    if version != VideoNet.VERSION:
        print("WARNING: The model version is not the last one.")
        print("         Some temporary updates will be made to the internal data.")
        print("         Use packer CLI to make it permanent.")

    # Check if checkpoint
    if "checkpoint" in data:
        raise RuntimeError("This is a checkpoint, read it with checkpoint_load instead.")

    # Read meta
    model = meta_to_protomodel(data["model_meta_dict"], None, None)

    # Load parameters data into the model
    model.load_state_dict(data["model_state_dict"])

    # Back to float32 as a standard format
    # TODO: see if this conversion is necessary
    model.to(torch.float32)

    return model


def model_load(file_path: str) -> VideoNet:
    """
    Standard model save procedure
    """
    data = torch.load(file_path, map_location="cpu")
    return model_load_from_data(data)


def checkpoint_save(model: VideoNet, file_path: str, epoch=None):
    """
    Checkpoint save procedure
    """
    data = {
        "version": VideoNet.VERSION,
        "model_state_dict": model.state_dict(),
        "model_meta_dict": model.to_meta_dict(),
        "checkpoint": {
            "epoch": epoch,
            "optimizer_state_dict": model.optimizer.state_dict(),
            "training_params": model.training_params.to_dict(),
        },
    }
    torch.save(data, file_path)


def checkpoint_load(file_path: str, training_params) -> VideoNet:
    """
    Checkpoint save procedure
    """
    data = torch.load(file_path, map_location="cpu")
    version = data.get("version")

    assert version == VideoNet.VERSION, f"packed version {version} not supported"

    # Check if checkpoint
    if "checkpoint" not in data:
        raise RuntimeError("This is not a checkpoint, read it with model_load instead.")

    # Read meta
    checkpoint = data["checkpoint"]
    loaded_training_params = TrainingParams.from_dict(checkpoint["training_params"])
    # TODO: merge loaded_training_params with training_params
    model = meta_to_protomodel(data["model_meta_dict"], loaded_training_params, checkpoint)

    # Load parameters data into the model
    model.load_state_dict(data["model_state_dict"])

    # Back to float32 as a standard format
    # TODO: see if this conversion is necessary
    model.to(torch.float32)

    return model
