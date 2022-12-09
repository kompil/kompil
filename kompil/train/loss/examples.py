import torch

from torch.utils.data import IterableDataset

from .base import register_loss, Loss


@register_loss("example_prepare_data")
class PrepareDataExample(Loss):
    """
    Example for handling data preparation
    """

    def __init__(self):
        super().__init__()
        self.__frame_average = None

    def prepare(self, dataset: IterableDataset):
        self.__frame_average = torch.empty(len(dataset))
        print("Preparing data for loss PrepareDataExample", end="\r")
        for i in range(len(dataset)):
            _, ref = dataset[i]
            print(
                f"Preparing data for loss PrepareDataExample: {100.0 * i/(len(dataset) - 1):0.2f}%",
                end="\r",
            )
            self.__frame_average[i] = ref.mean()
        print()

    def __call__(self, y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
        assert self.__frame_average is not None, "Loss is not prepared"
        self.__frame_average = self.__frame_average.to(y_pred.device)
        frames_idx = context["frames_idx"]
        return torch.nn.functional.mse_loss(y_pred, y_ref) * self.__frame_average[frames_idx].mean()
