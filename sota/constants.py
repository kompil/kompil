import os

FILENAME_MODEL = "model.pth"
FILENAME_EVAL = "eval.json"
FILENAME_QUANTIZED_MODEL = "quantized_model.pth"
FILENAME_EVAL_QUANT = "eval.quant.json"
FILENAME_PACKER_INFO = "packer.json"
FILENAME_PACKED_MODEL = "packed_model.zip"
SOTA_DATA_PATH = os.path.expanduser("~/.kompil/runs/sota/data")
SOTA_HTML_PATH = os.path.expanduser("~/.kompil/runs/sota/html")
SOTA_STD_PATH = os.path.expanduser("~/.kompil/runs/sota/std")
