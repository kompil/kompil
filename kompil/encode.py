import os
import ast
import sys
import json
import torch
import textwrap
import traceback
import pytorch_lightning as pl
import kompil.data.video as video

from typing import Union, Tuple, List, Any, Optional, Dict
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from kompil.data import VideoDataset
from kompil.data.section import get_frame_location_mapping, mapping_to_index_table
from kompil.train.training import TrainingParams
from kompil.train.learning_rate import factory as lr_factory
from kompil.train.optimizers import SchedulerItems
from kompil.train.training_callbacks import (
    RobinHoodCallback,
    ArouseWeightsCallback,
    PruningCallback,
    FineTuningCallback,
)
from kompil.nn.models.model import VideoNet, model_save, checkpoint_load
from kompil.nn.topology.pattern import pattern_to_topology
from kompil.profile.report import EncodingReport
from kompil.profile.criteria import Criteria
from kompil.profile.profiling_callbacks import (
    TensorboardCallback,
    SaveLastCallback,
    SaveBestCallback,
    BenchmarkEpochStorage,
    QuitCallback,
    RemainingTimeEstCallback,
)
from kompil.profile.report_callback import ReportCallback
from kompil.utils.args import arg_keyval_list_to_dict, KeyvalType
from kompil.utils.resources import get_video
from kompil.utils.time import now_str, setup_start_time
from kompil.utils.dynamic_config import DynamicConfigFile
from kompil.utils.paths import (
    PATH_BUILD,
    PATH_LOGS,
    clear_dir,
)
from kompil.utils.static_config import (
    get_allowed_memory,
    get_name,
)
from kompil.utils.backup import save_local_model_and_report
from kompil.utils.video import get_video_frame_info
from kompil.nn.layers.save_load import save_save_layers, load_load_layers
from kompil.nn.layers.weightnorm import norm_weight_norm_layers, rm_norm_weight_norm_layers
from kompil.cli_defaults import EncodingDefaults as defaults

TEST_EPOCH_STEP = 10
START_ROBIN_HOOD = 400
ROBIN_HOOD_CUTOFF = 0.8
DEFAULT_AROUSE_STD = 0.025
DEFAULT_AROUSE_STEP = 200


def record(
    model: VideoNet,
    report: EncodingReport,
    output_folder_opt: str,
    report_path: Union[str, None],
):
    if not model and not report:
        raise Exception("Model and report are none, something wen't wrong during encoding !")

    if output_folder_opt:
        save_local_model_and_report(model, report, output_folder_opt)

    if report_path:
        # Save report
        report_path = os.path.expanduser(report_path)
        os.makedirs(report_path, exist_ok=True)
        report.save_to(report_path)
        # Copy model into the report folder
        target_model_path = os.path.join(report_path, "model.pth")
        model_save(model, target_model_path)
        print("Report and model saved into:", report_path)

    print(f"End of recording !")


def __get_video_loader(
    mode: str,
    video_path: str,
    resolution: str,
    colorspace: str,
    start_frame: Union[int, List[int], None],
    frames_count: Union[int, List[int], None],
    cluster: Union[List[Union[str, int]], None],
):
    import decord

    nb_frames = len(decord.VideoReader(uri=video_path, ctx=decord.cpu()))

    # Make a mapping for each frame to the target section
    frames_mapping = get_frame_location_mapping(nb_frames, start_frame, frames_count, cluster)
    index_table = mapping_to_index_table(frames_mapping)

    print(f"{len(index_table)} frames to encode")

    # Devices
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda")

    # Manage auto loading mode
    if mode == "auto":
        mode = __get_loading_mode(video_path, resolution)
        print("Auto loading mode set to:", mode)

    # Get video loaders
    if mode == "full-gpu":
        return video.FullVideoLoader(
            video_path, colorspace, cuda_device, cuda_device, False, resolution, index_table
        )

    if mode == "full-ram":
        return video.FullVideoLoader(
            video_path, colorspace, cpu_device, cuda_device, True, resolution, index_table
        )

    return video.StreamVideoLoader(
        video_path,
        colorspace,
        cuda_device,
        resolution,
        index_table,
        False,
    )


def __get_loading_mode(video_path: str, resolution: str):
    max_allowed_ram, max_allowed_vram = get_allowed_memory()

    frames, c, h, w = get_video_frame_info(video_path, resolution)

    estimate_uint8_weight = frames * c * h * w

    if estimate_uint8_weight < max_allowed_vram:
        return "full-gpu"

    if estimate_uint8_weight < max_allowed_ram:
        return "full-ram"

    return "stream"


def __get_actual_final_epoch(autoquit_epoch: int, fine_tuning: Optional[int]):
    return autoquit_epoch + fine_tuning if fine_tuning is not None else autoquit_epoch


def __get_callbacks(
    train_dataset: video.VideoLoader,
    tester: BenchmarkEpochStorage,
    robin_hood: bool,
    no_models_on_ram: bool,
    arouse: List[float],
    pruning: Union[Tuple[str, List[Tuple[int, float]]], None],
    lottery_ticket: bool,
    autoquit_epoch: int,
    criteria: Criteria,
    remote: DynamicConfigFile,
    training_params: TrainingParams,
    batch_size: int,
    topology_builder: str,
    model_extra: list,
    video_name: str,
    fine_tuning: Union[int, None],
    take_best: bool,
    quality_metric: str,
    eval_metrics: List[str],
    grad_clipping: Union[float, None],
    batch_acc: Union[int, None],
) -> Dict[str, pl.callbacks.Callback]:

    callbacks = {
        "tensorboard": TensorboardCallback(
            tester=tester, epoch_step=TEST_EPOCH_STEP, path_logs=PATH_LOGS
        ),
    }

    if not no_models_on_ram:
        callbacks["save_last"] = SaveLastCallback(epoch_step=TEST_EPOCH_STEP)

    if pruning is not None:
        callbacks["prunning"] = PruningCallback(
            pruning_type=pruning[0],
            pruning_dict={epoch: val for epoch, val in pruning[1]},
            lottery_ticket=lottery_ticket,
        )

    if robin_hood:
        callbacks["robin_hood"] = RobinHoodCallback(
            tester=tester,
            dataset=train_dataset,
            cutoff_ratio=ROBIN_HOOD_CUTOFF,
            epoch_step=TEST_EPOCH_STEP,
            start_epoch=START_ROBIN_HOOD,
        )

    if arouse:
        nb_param = len(arouse)

        assert nb_param in [
            2,
            3,
            4,
        ], "Arouse must be in format : first_epoch last_epoch [epoch_step] [std]"

        step = int(arouse[2]) if nb_param >= 3 else DEFAULT_AROUSE_STEP
        std = int(arouse[3]) if nb_param >= 4 else DEFAULT_AROUSE_STD

        callbacks["arouses"] = ArouseWeightsCallback(
            start_epoch=int(arouse[0]), last_epoch=int(arouse[1]), epoch_step=step, std=std
        )

    quit_cb = QuitCallback(
        tester=tester,
        max_epochs=__get_actual_final_epoch(autoquit_epoch, fine_tuning),
        criteria=criteria,
        config=remote.handle("quit_callback"),
        epoch_step=TEST_EPOCH_STEP,
    )
    callbacks["quit"] = quit_cb

    callbacks["report"] = ReportCallback(
        criteria=criteria,
        dataset=train_dataset,
        training_params=training_params,
        tester=tester,
        quit_callback=quit_cb,
        batch_size=batch_size,
        topology_builder=topology_builder,
        model_extra=model_extra,
        video_name=video_name,
        pruning=pruning,
        fine_tuning=fine_tuning,
        take_best=take_best,
        quality_metric=quality_metric,
        eval_metrics=eval_metrics,
        grad_clipping=grad_clipping,
        batch_acc=batch_acc,
    )

    callbacks["save_best"] = SaveBestCallback(
        tester=tester,
        epoch_step=TEST_EPOCH_STEP,
        save_on_disk=not no_models_on_ram,
        quality_metric=quality_metric,
    )

    if fine_tuning is not None:
        callbacks["fine_tuning"] = FineTuningCallback(
            tester=tester,
            epochs=fine_tuning,
            start=autoquit_epoch,
            quality_metric=quality_metric,
            epoch_step=TEST_EPOCH_STEP,
        )

    callbacks["remaining_time_est"] = RemainingTimeEstCallback(
        __get_actual_final_epoch(autoquit_epoch, fine_tuning)
    )

    return callbacks


def __get_learning_rate(
    learning_rate: Union[float, str], dataset: VideoDataset, batch_size: int, params: dict
) -> float:

    name = str(learning_rate)

    if lr_factory().has(name):
        lr_fct = lr_factory()[name]

        return lr_fct(
            nb_frames=len(dataset),
            frame_shape=dataset.video_loader.frame_shape,
            batch_size=batch_size,
            **params,
        )
    else:
        return float(learning_rate)


def get_model(
    model_name: str,
    checkpoint: Union[str, None],
    topology_builder: str,
    training_params: TrainingParams,
    nb_frames: int,
    frame_shape: tuple,
    fps: int,
    colorspace: str,
    model_extra: List[Any],
) -> VideoNet:
    # Load from checkpoint
    if checkpoint is not None:
        print(f"Loading checkpoint model from file {checkpoint}")
        return checkpoint_load(checkpoint, training_params)

    # Builder
    else:
        topo_dict = pattern_to_topology(
            pattern=topology_builder,
            out_shape=torch.Size(frame_shape),
            nb_frames=nb_frames,
            model_extra=model_extra,
        )

    # Generic
    model: VideoNet = VideoNet(
        name=model_name,
        nb_frames=nb_frames,
        fps=fps,
        frame_shape=frame_shape,
        topology=topo_dict,
        training_params=training_params,
        colorspace=colorspace,
    )

    return model


def encode(
    model_name: Union[str, None],
    video_name: str,
    autoquit_epoch: int = defaults.MAX_EPOCH,
    fine_tuning: int = defaults.FINE_TUNING,
    batch_size: int = defaults.BATCH_SIZE,
    learning_rate: float = defaults.LEARNING_RATE,
    params_lr: KeyvalType = None,
    output_folder: Optional[str] = None,
    checkpoint: Optional[str] = None,
    loading_mode: str = defaults.LOADING_MODE,
    topology_builder: str = defaults.TOPO_BUILDER,
    loss: str = defaults.LOSS,
    params_loss: KeyvalType = None,
    opt: str = defaults.OPT,
    robin_hood: bool = False,
    criteria: KeyvalType = None,
    keep_logs: bool = False,
    scheduler: str = defaults.SCHEDULER,
    params_scheduler: KeyvalType = None,
    resolution: Optional[str] = None,
    no_models_on_ram: bool = False,
    precision: int = defaults.PRECISION,
    model_extra: List[Any] = defaults.MODEL_EXTRA,
    frames_count: Union[int, List[int], None] = None,
    start_frame: Union[int, List[int], None] = None,
    cluster: Optional[List[Union[str, int]]] = None,
    arouse: Optional[List[float]] = None,
    colorspace: str = defaults.COLORSPACE,
    pruning: Optional[Tuple[str, List[Tuple[int, float]]]] = None,
    lottery_ticket: bool = False,
    take_best: bool = False,
    quality_metric: str = defaults.QUALITY_METRIC,
    eval_metrics: List[str] = defaults.EVAL_METRICS,
    grad_clipping: float = defaults.GRAD_CLIPPING,
    batch_acc: int = defaults.BATCH_ACC,
) -> Tuple[VideoNet, EncodingReport]:
    setup_start_time(force=True)

    assert (
        quality_metric in eval_metrics
    ), f"quality metric ({quality_metric}) has to be included in the evaluate metrics."

    if cluster and (start_frame or frames_count):
        raise Exception(f"Cannot specify a cluster mask and a specific time internal")

    if video_name.startswith("mock_"):
        video_path = video_name
        resolution = video_path[len("mock_") :]
        video_loader = video.MockVideoLoader(
            resolution=resolution,
            colorspace=colorspace,
            target_device=torch.device("cuda"),
        )
        model_name = video_path
        output_folder = output_folder if output_folder else f"{PATH_BUILD}/{model_name}"

    else:
        video_path = get_video(video_name)
        if not os.path.exists(video_path):
            raise Exception(f"Video path does not exists: {video_path}")

        if model_name is None:
            model_name = os.path.splitext(os.path.split(video_path)[1])[0]
            print(f"Model name automatically set to {model_name}")

        output_folder = output_folder if output_folder else f"{PATH_BUILD}/{model_name}"

        video_loader = __get_video_loader(
            loading_mode, video_path, resolution, colorspace, start_frame, frames_count, cluster
        )

    train_dataset = VideoDataset(loader=video_loader, shuffle=True)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
    )

    nb_frame = len(train_dataset)

    params_lr = arg_keyval_list_to_dict(params_lr)

    base_learning_rate = __get_learning_rate(
        learning_rate=learning_rate, dataset=train_dataset, batch_size=batch_size, params=params_lr
    )

    if params_loss is not None:
        params_loss = arg_keyval_list_to_dict(params_loss)

    if params_scheduler is not None:
        params_scheduler = arg_keyval_list_to_dict(params_scheduler)

    training_params = TrainingParams(
        loss_name=loss,
        loss_params=params_loss,
        optimizer_name=opt,
        optimizer_params={"lr": base_learning_rate},
        scheduler_name=scheduler,
        scheduler_params=params_scheduler,
    )
    print("Training parameters:")
    print(textwrap.indent(training_params.description, " " * 4))

    if criteria is None:
        criteria = Criteria(None)
    else:
        criteria = Criteria.from_keywords(**arg_keyval_list_to_dict(criteria))

    # Get or create model
    model = get_model(
        model_name=model_name,
        checkpoint=checkpoint,
        topology_builder=topology_builder,
        training_params=training_params,
        nb_frames=nb_frame,
        frame_shape=train_dataset.video_loader.frame_shape,
        fps=train_dataset.fps,
        colorspace=colorspace,
        model_extra=model_extra,
    )
    load_load_layers(model)
    norm_weight_norm_layers(model)

    model.prepare_training(train_dataset)
    model.train()
    model.to(video_loader.device)

    if not keep_logs:
        clear_dir(PATH_LOGS)

    # Configure learning
    tester = BenchmarkEpochStorage(train_dataset, bench_batch=batch_size, eval_metrics=eval_metrics)

    remote = DynamicConfigFile(file_name="kompil_remote.json")

    scheduler_items = SchedulerItems(
        nb_frames=nb_frame,
        tester=tester,
        dyn_conf=remote.handle("scheduler"),
        learning_epochs=autoquit_epoch,
        fine_tuning_epochs=fine_tuning,
    )
    model.set_scheduler_items(scheduler_items)

    logger = TensorBoardLogger(save_dir=PATH_LOGS, version=now_str(), name="lightning")
    callbacks = __get_callbacks(
        train_dataset=train_dataset,
        tester=tester,
        robin_hood=robin_hood,
        no_models_on_ram=no_models_on_ram,
        arouse=arouse,
        pruning=pruning,
        lottery_ticket=lottery_ticket,
        autoquit_epoch=autoquit_epoch,
        criteria=criteria,
        remote=remote,
        training_params=training_params,
        batch_size=batch_size,
        topology_builder=topology_builder,
        model_extra=model_extra,
        video_name=video_name,
        fine_tuning=fine_tuning,
        take_best=take_best,
        quality_metric=quality_metric,
        eval_metrics=eval_metrics,
        grad_clipping=grad_clipping,
        batch_acc=batch_acc,
    )

    quit_cb = callbacks["quit"]
    report_cb = callbacks["report"]
    save_best_cb = callbacks["save_best"]

    trainer = pl.Trainer(
        max_epochs=sys.maxsize,
        callbacks=list(callbacks.values()),
        benchmark=True,  # auto-speedup
        logger=logger,
        checkpoint_callback=False,
        gpus=1,
        precision=precision,
        amp_backend="native",
        amp_level="O2",
        terminate_on_nan=True,
        gradient_clip_val=grad_clipping,
        accumulate_grad_batches=batch_acc,
    )

    if model.checkpoint_epoch is not None:
        trainer.current_epoch = model.checkpoint_epoch

    # Learn
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Start encoding {model.name}...")
        trainer.fit(model, train_dataloader)

        if quit_cb is not None and quit_cb.criteria_reached:
            print(f"Encoding {model.name} stopped by reaching the quality criteria")
        else:
            print(f"Encoding {model.name} stopped by reaching the epoch {trainer.current_epoch}.")

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(f"Error during encoding of {model.name}: {e}")
        status = f"failure: {e}"
        report_cb.make_report(status, model, False, trainer.current_epoch)
    finally:
        print("Free up encode memory...")
        del train_dataloader
        del train_dataset
        del trainer

        torch.cuda.empty_cache()

    final_model = model
    if take_best:
        if save_best_cb.best is None:
            print("WARNING: best model not found, save last instead.")
        else:
            final_model = save_best_cb.best

    rm_norm_weight_norm_layers(model)
    save_save_layers(model)

    return final_model, report_cb.report


def args_encode(args):
    """Regroup all functions to be called for a proper encoding based on the argparse arguments"""

    # Add mock encode
    if args.video == "mock":
        args.autoquit_epoch = 1

    # Pruning args to list of pair
    pruning = None
    if args.pruning is not None:
        assert len(args.pruning) % 2 == 1
        assert len(args.pruning) >= 3
        assert isinstance(args.pruning[0], str)
        pruning = (
            str(args.pruning[0]),
            [
                (int(args.pruning[i]), float(args.pruning[i + 1]))
                for i in range(1, len(args.pruning), 2)
            ],
        )

    model, report = encode(
        model_name=args.name,
        video_name=args.video,
        autoquit_epoch=args.autoquit_epoch,
        fine_tuning=args.fine_tuning,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        params_lr=args.params_lr,
        output_folder=args.output_folder,
        checkpoint=args.checkpoint,
        loading_mode=args.loading_mode,
        topology_builder=args.topology_builder,
        loss=args.loss,
        params_loss=args.params_loss,
        opt=args.opt,
        robin_hood=args.robin_hood,
        criteria=args.criteria,
        keep_logs=args.keep_logs,
        scheduler=args.scheduler,
        params_scheduler=args.params_scheduler,
        resolution=args.resolution,
        no_models_on_ram=args.no_models_on_ram,
        precision=args.precision,
        model_extra=args.model_extra,
        frames_count=args.frames_count,
        start_frame=args.start_frame,
        cluster=args.cluster,
        arouse=args.arouse,
        colorspace=args.colorspace,
        pruning=pruning,
        lottery_ticket=args.lottery_ticket,
        take_best=args.take_best,
        quality_metric=args.quality_metric,
        eval_metrics=args.eval_metrics,
        grad_clipping=args.gradient_clipping,
        batch_acc=args.accumulate_batches,
    )

    if report is not None:
        record(model, report, args.output_folder, args.report_path)
