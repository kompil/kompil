"""
Regroup all the CLI defaults arguments.
"""


class EncodingDefaults:
    LEARNING_RATE = "primus"
    TOPO_BUILDER = "aurora_quant2_mp"
    MODEL_EXTRA = ["0.5"]
    SCHEDULER = "saliout"
    PRECISION = 16
    COLORSPACE = "ycbcr420"
    LOSS = "iron"
    OPT = "adamo"
    MAX_EPOCH = 3000
    BATCH_SIZE = 16
    LOADING_MODE = "auto"
    FINE_TUNING = None
    QUALITY_METRIC = "psnr"
    EVAL_METRICS = ["psnr", "ssim", "vmaf"]
    GRAD_CLIPPING = 1.0
    BATCH_ACC = 1


class QuantizeDefaults:
    METHOD = "dozer"


class CorrectorDefaults:
    METHOD = "harkonnen"


class PackerDefaults:
    METHOD = "foo"


class TopologyDefaults:
    FRAMES = 150
    RESOLUTION = "320p"
    FRAMERATE = 30.0
    COLORSPACE = "ycbcr420shift"


class ModelCompareDefaults:
    FRAME = 42
    OUTPUT_PATH = "index.html"


class SectionClusterDefaults:
    THRESHOLD = 1e-5
    GEN_METHOD = "sift"


class SectionSplitDefaults:
    SEC_METHOD = "hard"
