#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# -*- coding: utf-8 -*-

import argparse
import argcomplete

import kompil.cli_defaults


def pair(arg):
    # For simplity, assume arg is a pair of integers
    # separated by a ':'. If you want to do more
    # validation, raise argparse.ArgumentError if you
    # encounter a problem.
    return [int(x) for x in arg.split(":")]


def find_optimizers_names(**kwargs):
    from kompil.train.optimizers.optimizer import factory

    return factory().keys()


def find_losses_names(**kwargs):
    from kompil.train.loss.base import factory

    return factory().keys()


def find_topology_names(**kwargs):
    from kompil.nn.topology.pattern import factory

    return factory().keys()


def find_scheduler_names(**kwargs):
    from kompil.train.optimizers.scheduler import factory

    return factory().keys()


def find_resolutions_names(**kwargs):
    from kompil.utils.video import RESOLUTION_MAP

    return ["-1", *RESOLUTION_MAP.keys()]


def find_packers(**kwargs):
    from kompil.packers import factory_packer

    return factory_packer().keys()


def find_learning_rates(**kwargs):
    from kompil.train.learning_rate import factory

    return factory().keys()


def find_metrics(**kwargs):
    from kompil.metrics.metrics import factory

    return factory().keys()


def find_colorspace(**kwargs):
    from kompil.utils.colorspace import COLORSPACE_LIST

    return COLORSPACE_LIST


def find_maskmakers(**kwargs):
    from kompil.maskmakers import maskmaker_factory

    return maskmaker_factory().keys()


def find_player_decoders(**kwargs):
    from kompil.player.decoders.decoder import decoder_factory

    return decoder_factory().keys()


def find_quantizers(**kwargs):
    from kompil.quant import factory

    return factory().keys()


def find_correctors(**kwargs):
    from kompil.corr import factory

    return factory().keys()


def find_quantization_blacklists(**kwargs):
    from kompil.quant.mapping import get_blacklist_mapping

    return get_blacklist_mapping().keys()


def setup():
    import numpy
    import torch
    import decord
    from kompil.utils.time import setup_start_time

    setup_start_time()

    decord.bridge.set_bridge("torch")

    torch.manual_seed(4242)
    numpy.random.seed(4242)


def fill_encode(subparser):
    dft = kompil.cli_defaults.EncodingDefaults

    parser = subparser.add_parser("encode", help="Encode the target video file.")
    parser.add_argument("video", type=str, help="Path to the video to encode.")
    parser.add_argument(
        "--batch-size", default=dft.BATCH_SIZE, type=int, help="Size of the batches."
    )
    parser.add_argument(
        "--learning-rate",
        default=dft.LEARNING_RATE,
        help="Specify the learning rate. Specify dynamic for nb frames based learning rate.",
    ).completer = find_learning_rates
    parser.add_argument(
        "--name",
        default=None,
        type=str,
        help="Specify the name of the model, it will be the name of the video if not specified.",
    )
    parser.add_argument(
        "--output-folder",
        default="build",
        type=str,
        help="Specify the output folder for the model saving.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        type=str,
        help="Target path to a copy of the report joined with a copy of the final model.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Restart the learning by loading a model file.",
    )
    parser.add_argument(
        "--topology-builder",
        default=dft.TOPO_BUILDER,
        type=str,
        help="Specify a topology to start with.",
    ).completer = find_topology_names
    parser.add_argument(
        "--loading-mode",
        default=dft.LOADING_MODE,
        type=str,
        choices=["auto", "full-ram", "full-gpu", "stream"],
        help="Loading strategy for the input video.",
    )
    parser.add_argument(
        "--loss",
        default=dft.LOSS,
        type=str,
        help="Loss function applied during the learning.",
    ).completer = find_losses_names
    parser.add_argument(
        "--params-loss",
        default=None,
        type=str,
        nargs="+",
        help="Parameterization for the selected loss. Goes with key/value pairs.",
    )
    parser.add_argument(
        "--opt",
        "--optimizer",
        default=dft.OPT,
        type=str,
        help="Optimizer used during the learning.",
    ).completer = find_optimizers_names
    parser.add_argument(
        "--robin-hood",
        action="store_true",
        help="Activate the Robin Hood training optimization. Steal the best for the good of the few !",
    )
    parser.add_argument(
        "--autoquit-epoch", default=dft.MAX_EPOCH, type=int, help="Stop after N epochs."
    )
    parser.add_argument(
        "--fine-tuning", default=dft.FINE_TUNING, type=int, help="Fine tune during N epochs."
    )
    parser.add_argument(
        "--criteria",
        default=None,
        type=str,
        nargs="+",
        help=(
            "Stop when the quality target is reached. Goes with key/value pairs. "
            "Example: min_psnr 30.0 mean_vmaf 98.0"
        ),
    )
    parser.add_argument(
        "-m",
        "--quality-metric",
        type=str,
        default=dft.QUALITY_METRIC,
        help="Specify the quality metric that will identify the best models.",
    ).completer = find_metrics
    parser.add_argument(
        "--eval-metrics",
        type=str,
        nargs="+",
        default=dft.EVAL_METRICS,
        help="Specify the quality metrics that will processed during evaluation.",
    ).completer = find_metrics
    parser.add_argument(
        "--scheduler",
        default=dft.SCHEDULER,
        type=str,
        help="Specify a scheduler with a set of default params.",
    ).completer = find_scheduler_names
    parser.add_argument(
        "--params-scheduler",
        default=None,
        type=str,
        nargs="+",
        help="Parameterization for the selected scheduler. Goes with key/value pairs.",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Don't clear logs directory before running encoding.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default="-1",
        type=str,
        help="Specify the output resolution (height in pixel) of the encoded video. Downscale at loading time. -1 for same resolution.",
    ).completer = find_resolutions_names
    parser.add_argument(
        "--no-models-on-ram",
        action="store_true",
        help="Disable the automatic saving of the last and best model on the disk during training.",
    )
    parser.add_argument(
        "--take-best",
        action="store_true",
        help="At the end of the encode, take the best model based on its average VMAF score.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=[16, 32],
        default=dft.PRECISION,
        help="Floating point precision.",
    )
    parser.add_argument(
        "-x",
        "--model-extra",
        "--params-topology",
        nargs="+",
        default=dft.MODEL_EXTRA,
        help="Additional info to build the topology.",
    )
    parser.add_argument(
        "--params-lr",
        nargs="+",
        default=None,
        help="Additional info to generate the learning rate.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        nargs="+",
        default=None,
        help="Index (or list) of the first frame of the sequence to encode.",
    )
    parser.add_argument(
        "--frames-count",
        type=int,
        nargs="+",
        default=None,
        help="Number (or list) of frames to encode starting from corresponding start-frame.",
    )
    parser.add_argument(
        "-c",
        "--cluster",
        type=str,
        nargs="+",
        default=None,
        help="If not None, must be in that format : path/to/cluster_index.pth [idx_0 idx_1...]",
    )
    parser.add_argument(
        "--arouse",
        type=float,
        nargs="+",
        default=None,
        help="Apply a N(0, std) random over all weight at each step. If not None, must be in that format : first_epoch last_epoch [epoch_step] [std]",
    )
    parser.add_argument(
        "--colorspace",
        default=dft.COLORSPACE,
        type=str,
        help="Output frame colorspace. It can affect compatibility with loss and topo.",
    ).completer = find_colorspace
    parser.add_argument(
        "-p",
        "--pruning",
        nargs="+",
        default=None,
        help=(
            "Use pruning method. Pruning name plus List of (epoch, value) to define when to prune. "
            "Negative value will remove the pruning at the specified epoch."
        ),
    )
    parser.add_argument(
        "--lottery-ticket",
        action="store_true",
        help="Apply the lottery ticket hypothesis alongside pruning.",
    )
    parser.add_argument(
        "--gradient-clipping",
        type=float,
        default=dft.GRAD_CLIPPING,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--accumulate-batches",
        type=int,
        default=dft.BATCH_ACC,
        help="Accumulate gradients of batches to simulate higher batch sizes.",
    )

    def encode(args):
        setup()
        from kompil.encode import args_encode

        assert args.colorspace is None or args.colorspace in find_colorspace()

        args_encode(args)

    parser.set_defaults(func=encode)


def fill_play(subparser):
    parser = subparser.add_parser("play", help="Play an encoded video")
    parser.add_argument("files", type=str, nargs="+", help="Video file path (.pth, .avi or .mp4)")
    parser.add_argument("--cpu", action="store_true", help="Run the model in cpu.")
    parser.add_argument("--fp32", action="store_true", help="Run models using 32 bits floats.")
    parser.add_argument(
        "-r", "--resolution", type=str, default=None, help="Force the target resolution."
    ).completer = find_resolutions_names
    parser.add_argument("--framerate", type=float, default=None, help="Target frame per second.")
    parser.add_argument("--pipe", action="store_true", help="Pipe to stdout.")

    for i in range(10):
        parser.add_argument(
            f"--decoder{i}", type=str, default="auto", help=f"Define decoder {i}"
        ).completer = find_player_decoders
        parser.add_argument(
            f"--filter{i}", nargs="+", type=str, default=[], help=f"Define filter {i}"
        )

    def play(args):
        setup()

        from kompil.play import play

        play(**vars(args))

    parser.set_defaults(func=play)


def fill_eval(subparser):
    parser = subparser.add_parser("eval", help="Evaluate an encoded video")
    parser.add_argument("model_path", type=str, help="File path for the model meta file (.pth)")
    parser.add_argument(
        "video_name", type=str, help="Name or path to the video to compare the model to."
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Where to run model.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Size of the batches during computation.",
    )
    parser.add_argument(
        "-c",
        "--cluster",
        type=str,
        nargs="+",
        default=None,
        help="If not None, must be in that format : path/to/cluster_index.pth [idx_0 idx_1...]",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Compute in float32 (Single precision).",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Write the result of the evaluation in the target file.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=None,
        type=str,
        help="Force the resolution of the evaluation.",
    ).completer = find_resolutions_names

    def eval(args):
        setup()

        from kompil.eval import evaluate

        evaluate(
            args.model_path,
            args.video_name,
            args.device,
            args.batch_size,
            args.fp32,
            args.cluster,
            args.output_file,
            args.resolution,
        )

    parser.set_defaults(func=eval)


def fill_audio(subparser):
    parser = subparser.add_parser("audio", help="Audio management.")

    subparser2 = parser.add_subparsers(dest="what")

    extract_parser = subparser2.add_parser("extract", help="Extract audio from video.")
    extract_parser.add_argument(
        "video_file_path", type=str, default=None, help="The original video file."
    )
    extract_parser.add_argument("-o", "--output", type=str, default=None, help="Output audio file.")

    def audio(args):
        setup()

        from kompil.audio import extract_audio

        extract_audio(args.video_file_path, args.output)

    parser.set_defaults(func=audio)


def fill_topology(subparser):
    dft = kompil.cli_defaults.TopologyDefaults

    parser = subparser.add_parser("topology", help="Tools to manipulate topology patterns.")
    subparser2 = parser.add_subparsers(dest="command2")

    def common_arguments(_parser):
        _parser.add_argument("topology_builder", type=str).completer = find_topology_names
        _parser.add_argument("--frames", type=int, default=dft.FRAMES)
        _parser.add_argument(
            "-r",
            "--resolution",
            default=dft.RESOLUTION,
            type=str,
            help="Specify the resolution of the target video,",
        ).completer = find_resolutions_names
        _parser.add_argument("--framerate", default=dft.FRAMERATE, type=float)
        _parser.add_argument(
            "-x",
            "--model-extra",
            nargs="+",
            default=None,
            help="Addition info to build the model.",
        )
        _parser.add_argument(
            "--colorspace",
            default=dft.COLORSPACE,
            type=str,
            help="Output frame colorspace.",
        ).completer = find_colorspace

    # Show
    parser_show = subparser2.add_parser("show", help="Show characteristics of a topology.")
    common_arguments(parser_show)

    def show(args):
        setup()

        from kompil.topology import show

        show(
            args.topology_builder,
            args.frames,
            args.colorspace,
            args.resolution,
            args.framerate,
            args.model_extra,
        )

    parser_show.set_defaults(func=show)

    # Init
    parser_init = subparser2.add_parser("init", help="Generate model with initial weights.")
    common_arguments(parser_init)
    parser_init.add_argument("output", type=str)

    def init(args):
        setup()

        from kompil.topology import init

        init(
            args.topology_builder,
            args.frames,
            args.colorspace,
            args.output,
            args.resolution,
            args.framerate,
            args.model_extra,
        )

    parser_init.set_defaults(func=init)


def fill_section_cluster(subparser):
    dft = kompil.cli_defaults.SectionClusterDefaults

    parser = subparser.add_parser("cluster", help="Cluster frame from previously learnt data.")
    subparser2 = parser.add_subparsers(dest="what")

    # Generate data
    parser_gen = subparser2.add_parser("data", help="Generate feature data for further clustering.")
    parser_gen.add_argument("video_path", type=str, help="The input video to extract feature from.")
    parser_gen.add_argument("data_path", type=str, help="The output feature data path.")
    parser_gen.add_argument(
        "-m",
        "--method",
        type=str,
        default=dft.GEN_METHOD,
        choices=["sift", "sift-pca", "autoflow"],
        help="The feature data generation method.",
    )

    def gen(args):
        setup()

        from kompil.section_cluster import gen_data

        gen_data(args.video_path, args.data_path, args.method)

    parser_gen.set_defaults(func=gen)

    # Find
    parser_find = subparser2.add_parser("find", help="Find the best cluster combinaison.")
    parser_find.add_argument("file_path", type=str, help="Feature data file path.")
    parser_find.add_argument("nb_cluster", type=int, help="How many cluster to generate.")
    parser_find.add_argument(
        "-o", "--output", type=str, default=None, help="Output file with clusters data."
    )
    parser_find.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=dft.THRESHOLD,
        help="Center shift threshold, default is 1e-4.",
    )

    def find(args):
        setup()

        from kompil.section_cluster import find

        find(args.file_path, args.nb_cluster, args.output, args.threshold)

    parser_find.set_defaults(func=find)

    # Report
    parser_report = subparser2.add_parser("report", help="Write a report on the target cluster")
    parser_report.add_argument("cluster", type=str, help="Cluster file path.")
    parser_report.add_argument("video", type=str)
    parser_report.add_argument("-o", "--output", type=str, default=None)
    parser_report.add_argument(
        "-d", "--data", type=str, default=None, help="Optional data file path."
    )

    def report(args):
        setup()

        from kompil.section_cluster import cluster_report

        cluster_report(args.cluster, args.video, args.output, args.data)

    parser_report.set_defaults(func=report)


def fill_section(subparser):
    parser = subparser.add_parser("section", help="Every tools for the preparation of siwe.")
    subparser2 = parser.add_subparsers(dest="what")

    # Split
    dft = kompil.cli_defaults.SectionSplitDefaults

    parser_split = subparser2.add_parser("split", help="Video frame splitting tool.")
    parser_split.add_argument("video_file", type=str, help="Path to the video file.")
    parser_split.add_argument("nb_section", type=int, help="Nb section to generate.")
    parser_split.add_argument("output_file", type=str, help="Path to the output file.")
    parser_split.add_argument(
        "-m",
        "--method",
        type=str,
        default=dft.SEC_METHOD,
        choices=["cluster", "hard", "dynamic"],
        help="The section generation method.",
    )

    def split(args):
        setup()

        from kompil.section_split import split

        split(args.video_file, args.nb_section, args.output_file, args.method)

    parser_split.set_defaults(func=split)

    # Walloc
    parser_walloc = subparser2.add_parser("walloc", help="Weight allocation over sections.")
    parser_walloc.add_argument("section_file", type=str, help="Path to the section idx file.")
    parser_walloc.add_argument(
        "nb_mp",
        type=float,
        help="Nb of million parameters to allocate for all sections. Example : 14.5",
    )
    parser_walloc.add_argument("output_file", type=str, help="Path to the output file.")
    parser_walloc.add_argument(
        "-f",
        "--force",
        type=float,
        default=None,
        nargs="+",
        help="Force specific mp assignation to section. Must be in pair sect_idx nb_mp_idx. Example 3 0.5 4 12.4.",
    )

    def walloc(args):
        setup()

        from kompil.section_walloc import walloc

        walloc(args.section_file, args.nb_mp, args.output_file, args.force)

    parser_walloc.set_defaults(func=walloc)

    # Cluster
    fill_section_cluster(subparser2)


def fill_quantize(subparser):
    dft = kompil.cli_defaults.QuantizeDefaults

    parser = subparser.add_parser("quantize", help="Quantize the provided model.")
    parser.add_argument("src", type=str, help="Path to the model.")
    parser.add_argument("dst", type=str, help="Path to the output quantized model.")
    parser.add_argument(
        "-m", "--method", type=str, default=dft.METHOD, help="Quantization method used."
    ).completer = find_quantizers

    def quantize(args):
        setup()

        from kompil.quantize import quantize

        return quantize(args.src, args.dst, args.method)

    parser.set_defaults(func=quantize)


def fill_correct(subparser):
    dft = kompil.cli_defaults.CorrectorDefaults

    parser = subparser.add_parser("correct", help="Correct the provided quantized model.")
    parser.add_argument("o_src", type=str, help="Path to the original model.")
    parser.add_argument("qt_src", type=str, help="Path to the quantized model.")
    parser.add_argument("c_dst", type=str, help="Path to the output corrected model.")
    parser.add_argument(
        "-m", "--method", type=str, default=dft.METHOD, help="Correction method used."
    ).completer = find_correctors

    def correct(args):
        setup()

        from kompil.correct import correct

        return correct(args.o_src, args.qt_src, args.c_dst, args.method)

    parser.set_defaults(func=correct)


def fill_packer(subparser):
    dft = kompil.cli_defaults.PackerDefaults

    parser = subparser.add_parser("packer", help="Optimize model size.")
    subparser2 = parser.add_subparsers(dest="what")

    # Pack
    parser_pack = subparser2.add_parser("pack", help="Pack the quantized model.")
    parser_pack.add_argument("src", type=str, help="Path to the model.")
    parser_pack.add_argument("dst", type=str, help="Path to the output packed model.")
    parser_pack.add_argument(
        "-p", "-t", "--packer", type=str, default=dft.METHOD, help="Packer used to pack the file."
    ).completer = find_packers

    def pack(args):
        setup()

        from kompil.packer import packer_pack

        return packer_pack(args.src, args.dst, args.packer)

    parser_pack.set_defaults(func=pack)

    # Unpack
    parser_unpack = subparser2.add_parser("unpack", help="Unpack the packed quantized model.")
    parser_unpack.add_argument("src", type=str, help="Path to the packed model.")
    parser_unpack.add_argument("dst", type=str, help="Path to the output model.")

    def unpack(args):
        setup()

        from kompil.packer import packer_unpack

        return packer_unpack(args.src, args.dst)

    parser_unpack.set_defaults(func=unpack)


def fill_standards(subparser):
    parser = subparser.add_parser("standards", help="Tools to interact with standard codec.")
    subparser2 = parser.add_subparsers(dest="command2")

    # Bench
    parser_bench = subparser2.add_parser("bench", help="Bench a video according to its encoding.")
    parser_bench.add_argument("video_path", type=str)
    parser_bench.add_argument("encoding", choices=["avc", "vp9"], type=str)
    parser_bench.add_argument("quality", type=float, help="Between 0 and 100, it will be adjusted.")
    parser_bench.add_argument(
        "--metrics", nargs="+", type=str, default=["vmaf", "psnr", "ssim"]
    ).completer = find_metrics
    parser_bench.add_argument("-o", "--output", type=str, default=None)
    parser_bench.add_argument(
        "-r", "--resolution", type=str, default=None
    ).completer = find_resolutions_names
    parser_bench.add_argument("-k", "--keep-encoded-video", action="store_true")

    def bench(args):
        setup()

        from kompil.standards import video_bench

        video_bench(
            args.video_path,
            args.encoding,
            args.quality,
            args.metrics,
            args.output,
            args.resolution,
            args.keep_encoded_video,
        )

    parser_bench.set_defaults(func=bench)


def fill_maskmaker(subparser):
    parser = subparser.add_parser("maskmaker", help="Tools manipulate mask for video training.")
    subparser2 = parser.add_subparsers(dest="what")

    build_parser = subparser2.add_parser("build", help="Tools to generate a mask.")

    build_parser.add_argument("name", type=str).completer = find_maskmakers
    build_parser.add_argument("video", type=str)
    build_parser.add_argument(
        "-r", "--resolution", type=str, default=None
    ).completer = find_resolutions_names
    build_parser.add_argument(
        "--colorspace",
        default=None,
        type=str,
        help="Colorspace for the video.",
    ).completer = find_colorspace
    build_parser.add_argument("-o", "--output", type=str, default=None)
    build_parser.add_argument(
        "--params",
        default=None,
        type=str,
        nargs="+",
        help="Parameterization for the selected maskmaker. Goes with key/value pairs.",
    )

    merge_parser = subparser2.add_parser("merge", help="Tools to merge masks into one mask.")

    merge_parser.add_argument(
        "masks",
        default=None,
        type=str,
        nargs="+",
        help="Generated mask paths to merge.",
    )
    merge_parser.add_argument("-o", "--output", type=str, default=None)
    merge_parser.add_argument("-m", "--method", type=str, default=None, choices=["add", "max"])

    def maskmaker(args):
        setup()

        from kompil.maskmaker import run_maskmaker, run_maskmerger

        if args.what == "build":
            run_maskmaker(
                args.name, args.video, args.resolution, args.colorspace, args.params, args.output
            )
        elif args.what == "merge":
            run_maskmerger(args.masks, args.method, args.output)

    parser.set_defaults(func=maskmaker)


def fill_model_make_quantizable(subparser):
    dft = kompil.cli_defaults.QuantizeDefaults

    parser = subparser.add_parser(
        "make-quantizable", help="Add (De)Quantize layers into the provided model."
    )
    parser.add_argument("src", type=str, help="Path to the model.")
    parser.add_argument("dst", type=str, help="Path to the output quantizable model.")
    parser.add_argument(
        "-b", "--blacklist", type=str, default=None, help="Blacklist of layer to not quantize."
    ).completer = find_quantization_blacklists

    def make_quantizable(args):
        setup()

        from kompil.model_make_quantizable import make_quantizable

        return make_quantizable(args.src, args.dst, args.blacklist)

    parser.set_defaults(func=make_quantizable)


def fill_model_compare_quant(subparser):
    dft = kompil.cli_defaults.ModelCompareDefaults

    parser = subparser.add_parser(
        "compare-quant",
        help="Tools to compare a quantizable model and its quantized version. Must have the same topology.",
    )
    parser.add_argument("model_path", type=str)
    parser.add_argument("qmodel_path", type=str)
    parser.add_argument(
        "-f", "--frame-idx", type=int, default=dft.FRAME, help="The frame index to forward."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=dft.OUTPUT_PATH, help="Output file path."
    )

    def compare(args):
        setup()

        from kompil.model_compare_quant import analyse_model_quant

        analyse_model_quant(args.model_path, args.qmodel_path, args.frame_idx, args.output)

    parser.set_defaults(func=compare)


def fill_model_norm(subparser):
    parser = subparser.add_parser("normalize", help="Tools to normalize the nodes.")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--cpu", action="store_true")

    def normalize(args):
        setup()

        from kompil.model_norm import model_norm

        model_norm(args.input, args.output, args.cpu)

    parser.set_defaults(func=normalize)


def fill_model_analyze(subparser):
    parser = subparser.add_parser("analyze", help="Tools to read through stuff.")
    parser.add_argument("path", type=str)

    def analyze(args):
        setup()

        from kompil.model_analyze import analyse_model

        analyse_model(args.path)

    parser.set_defaults(func=analyze)


def fill_model_perf(subparser):
    parser = subparser.add_parser("perf", help="Profile kompil network performance.")
    parser.add_argument("model_path", type=str)

    def perf(args):
        setup()

        from kompil.model_perf import perf

        perf(args.model_path)

    parser.set_defaults(func=perf)


def fill_model_hook(subparser):
    parser = subparser.add_parser("hook", help="Hook tools.")
    parser.add_argument("model_file", type=str)
    parser.add_argument("--cpu", action="store_true", help="Run the model in cpu.")

    def hook(args):
        setup()

        from kompil.model_hook import hook

        hook(args.model_file, args.cpu)

    parser.set_defaults(func=hook)


def fill_model_resolve(subparser):
    parser = subparser.add_parser(
        "resolve", help="Load a model with external module and save it again."
    )
    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--output", type=str, default=None)

    def resolve(args):
        setup()

        from kompil.model_resolve import resolve

        resolve(args.model, args.output)

    parser.set_defaults(func=resolve)


def fill_model_generate(subparser):
    parser = subparser.add_parser("generate", help="Read a model and generate a video.")

    parser.add_argument(
        "path", type=str, help="File path for the model meta file (.pth) or the frames folder."
    )
    parser.add_argument("output", type=str, help="Path to the output video.")
    parser.add_argument(
        "--codec",
        choices=["h264", "libx265", "mpeg4", "y4m"],
        default="y4m",
        help="Codec to use",
    )
    parser.add_argument("--cpu", action="store_true", help="Run the model in cpu.")

    def generate(args):
        setup()

        from kompil.generate import generate_video, generate_y4m

        if args.codec == "y4m":
            generate_y4m(args.path, args.output, args.cpu)
            return

        generate_video(args.path, args.output, args.codec, args.cpu)

    parser.set_defaults(func=generate)


def fill_model(subparser):
    parser_gen = subparser.add_parser("model", help="Every tools to manipulate models.")

    subparser2 = parser_gen.add_subparsers()
    fill_model_generate(subparser2)
    fill_model_perf(subparser2)
    fill_model_hook(subparser2)
    fill_model_resolve(subparser2)
    fill_model_analyze(subparser2)
    fill_model_compare_quant(subparser2)
    fill_model_make_quantizable(subparser2)
    fill_model_norm(subparser2)


def create_parser():
    """
    Create the parser.
    """
    parser = argparse.ArgumentParser(description="Entry point for every kompil commands.")
    parser.set_defaults(func=lambda _: parser.print_help())
    # command
    subparser = parser.add_subparsers(dest="command")

    fill_encode(subparser)
    fill_play(subparser)
    fill_eval(subparser)
    fill_model(subparser)
    fill_topology(subparser)
    fill_packer(subparser)
    fill_standards(subparser)
    fill_audio(subparser)
    fill_section(subparser)
    fill_maskmaker(subparser)
    fill_quantize(subparser)
    fill_correct(subparser)

    return parser


def main():
    """
    launch the rightful command based on arguments from standard input
    """
    # create the parsing system
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # launch command
    args.func(args)


if __name__ == "__main__":
    main()
