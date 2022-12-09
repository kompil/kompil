import torch
from facenet_pytorch import MTCNN
from PIL import Image

from .factory import AutoMaskMakerBase, MASKTYPE, register_maskmaker

from kompil.utils.colorspace import colorspace_420_to_444, convert_to_colorspace
from kompil.utils.video import display_frame


@register_maskmaker("facetrack")
class MaskMakerFacetrack(AutoMaskMakerBase):
    def __init__(self, display: bool = False):
        self.__mask = None
        self.__display = display
        self.__counter = 0

    def init(self, nb_frames: int, frame_shape: torch.Size):
        self.__mask = torch.zeros(nb_frames, *frame_shape)
        self.__mtcnn = MTCNN(
            min_face_size=20,
            keep_all=True,
            device=torch.device("cuda"),
        )

    def push_frame(self, frame: torch.Tensor):
        self.__counter += 1

        mask = self.__mask[self.__counter - 1]

        # Detect faces in frame
        frame_rgb = torch.clamp(convert_to_colorspace(frame, "yuv420", "rgb8"), 0, 1.0)
        frame_rgb_per = (frame_rgb.permute(1, 2, 0) * 255.0).to(torch.uint8)
        boxes, _, _ = self.__mtcnn.detect(frame_rgb_per, landmarks=True)

        if boxes is None:
            return

        # Set mask to 1 in these places
        boxes = torch.as_tensor(boxes)
        for i in range(boxes.shape[0]):
            box = boxes[i].int()
            frame_rgb[0, box[1] : box[3], box[0] : box[2]] = 1.0

            box = (box / 2).int()
            mask[:, box[1] : box[3], box[0] : box[2]] = 1.0

        if not self.__display:
            return

        mask_yuv444 = colorspace_420_to_444(mask.unsqueeze(0)).squeeze(0)

        display_frame(mask_yuv444[0], "mask Y", 1)
        display_frame(mask_yuv444[1], "mask U", 1)
        display_frame(mask_yuv444[2], "mask V", 1)
        display_frame(frame_rgb, "original", 0)

    def compute(self) -> torch.HalfTensor:
        return self.__mask.to(MASKTYPE)
