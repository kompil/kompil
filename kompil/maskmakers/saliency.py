import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from kompil.maskmakers.factory import AutoMaskMakerBase, MASKTYPE, register_maskmaker
from kompil.utils.colorspace import colorspace_420_to_444, convert_to_colorspace
from kompil.utils.video import display_frame
from kompil.utils.resources import get_pytorch_model
from kornia.filters import gaussian_blur2d


class _SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(_SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(
            self.in_channels, self.in_channels // 2, (1, self.kernel_size), padding=(0, pad)
        )
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp1_convk1 = nn.Conv2d(
            self.in_channels // 2, 1, (self.kernel_size, 1), padding=(pad, 0)
        )
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(
            self.in_channels, self.in_channels // 2, (self.kernel_size, 1), padding=(pad, 0)
        )
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp2_conv1k = nn.Conv2d(
            self.in_channels // 2, 1, (1, self.kernel_size), padding=(0, pad)
        )
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class _ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(_ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats, ca_act_reg


_vgg_conv1_2 = None
_vgg_conv2_2 = None
_vgg_conv3_3 = None
_vgg_conv4_3 = None
_vgg_conv5_3 = None


def _conv_1_2_hook(module, input, output):
    global _vgg_conv1_2
    _vgg_conv1_2 = output
    return None


def _conv_2_2_hook(module, input, output):
    global _vgg_conv2_2
    _vgg_conv2_2 = output
    return None


def _conv_3_3_hook(module, input, output):
    global _vgg_conv3_3
    _vgg_conv3_3 = output
    return None


def _conv_4_3_hook(module, input, output):
    global _vgg_conv4_3
    _vgg_conv4_3 = output
    return None


def _conv_5_3_hook(module, input, output):
    global _vgg_conv5_3
    _vgg_conv5_3 = output
    return None


class _CPFE(nn.Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(_CPFE, self).__init__()

        self.dil_rates = [3, 5, 7]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == "conv5_3":
            self.in_channels = 512
        elif feature_layer == "conv4_3":
            self.in_channels = 512
        elif feature_layer == "conv3_3":
            self.in_channels = 256

        # Define layers
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False
        )
        self.conv_dil_3 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=self.dil_rates[0],
            padding=self.dil_rates[0],
            bias=False,
        )
        self.conv_dil_5 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=self.dil_rates[1],
            padding=self.dil_rates[1],
            bias=False,
        )
        self.conv_dil_7 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=self.dil_rates[2],
            padding=self.dil_rates[2],
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels * 4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat(
            (conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1
        )
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats


class _SODModel(nn.Module):
    def __init__(self):
        super(_SODModel, self).__init__()

        # Load the [partial] VGG-16 model
        self.vgg16 = models.vgg16(pretrained=True).features

        # Extract and register intermediate features of VGG-16
        self.vgg16[3].register_forward_hook(_conv_1_2_hook)
        self.vgg16[8].register_forward_hook(_conv_2_2_hook)
        self.vgg16[15].register_forward_hook(_conv_3_3_hook)
        self.vgg16[22].register_forward_hook(_conv_4_3_hook)
        self.vgg16[29].register_forward_hook(_conv_5_3_hook)

        # Initialize layers for high level (hl) feature (conv3_3, conv4_3, conv5_3) processing
        self.cpfe_conv3_3 = _CPFE(feature_layer="conv3_3")
        self.cpfe_conv4_3 = _CPFE(feature_layer="conv4_3")
        self.cpfe_conv5_3 = _CPFE(feature_layer="conv5_3")

        self.cha_att = _ChannelwiseAttention(in_channels=384)  # in_channels = 3 x (32 x 4)

        self.hl_conv1 = nn.Conv2d(384, 64, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(64)

        # Initialize layers for low level (ll) feature (conv1_2 and conv2_2) processing
        self.ll_conv_1 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.ll_bn_1 = nn.BatchNorm2d(64)
        self.ll_conv_2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(64)
        self.ll_conv_3 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_3 = nn.BatchNorm2d(64)

        self.spa_att = _SpatialAttention(in_channels=64)

        # Initialize layers for fused features (ff) processing
        self.ff_conv_1 = nn.Conv2d(128, 1, (3, 3), padding=1)

    def forward(self, input_):
        global _vgg_conv1_2, _vgg_conv2_2, _vgg_conv3_3, _vgg_conv4_3, _vgg_conv5_3

        # Pass input_ through vgg16 to generate intermediate features
        self.vgg16(input_)

        # Process high level features
        conv3_cpfe_feats = self.cpfe_conv3_3(_vgg_conv3_3)
        conv4_cpfe_feats = self.cpfe_conv4_3(_vgg_conv4_3)
        conv5_cpfe_feats = self.cpfe_conv5_3(_vgg_conv5_3)

        conv4_cpfe_feats = F.interpolate(
            conv4_cpfe_feats, scale_factor=2, mode="bilinear", align_corners=True
        )
        conv5_cpfe_feats = F.interpolate(
            conv5_cpfe_feats, scale_factor=4, mode="bilinear", align_corners=True
        )

        conv_345_feats = torch.cat((conv3_cpfe_feats, conv4_cpfe_feats, conv5_cpfe_feats), dim=1)

        conv_345_ca, ca_act_reg = self.cha_att(conv_345_feats)
        conv_345_feats = torch.mul(conv_345_feats, conv_345_ca)

        conv_345_feats = self.hl_conv1(conv_345_feats)
        conv_345_feats = F.relu(self.hl_bn1(conv_345_feats))
        conv_345_feats = F.interpolate(
            conv_345_feats, scale_factor=4, mode="bilinear", align_corners=True
        )

        # Process low level features
        conv1_feats = self.ll_conv_1(_vgg_conv1_2)
        conv1_feats = F.relu(self.ll_bn_1(conv1_feats))
        conv2_feats = self.ll_conv_2(_vgg_conv2_2)
        conv2_feats = F.relu(self.ll_bn_2(conv2_feats))

        conv2_feats = F.interpolate(
            conv2_feats, scale_factor=2, mode="bilinear", align_corners=True
        )
        conv_12_feats = torch.cat((conv1_feats, conv2_feats), dim=1)
        conv_12_feats = self.ll_conv_3(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_3(conv_12_feats))

        conv_12_sa = self.spa_att(conv_345_feats)
        conv_12_feats = torch.mul(conv_12_feats, conv_12_sa)

        # Fused features
        fused_feats = torch.cat((conv_12_feats, conv_345_feats), dim=1)
        fused_feats = torch.sigmoid(self.ff_conv_1(fused_feats))

        return fused_feats, ca_act_reg


@register_maskmaker("saliency")
class MaskMakerSaliency(AutoMaskMakerBase):
    def __init__(self, blur: bool = False, display: bool = False):
        self.__mask = None
        self.__display = display
        self.__blur = blur
        self.__counter = 0

    def init(self, nb_frames: int, frame_shape: torch.Size):
        self.__mask = torch.zeros(nb_frames, *frame_shape)

        model_dict = torch.load(get_pytorch_model("saliency_1.pth"))
        self.__saliency_model = _SODModel()
        self.__saliency_model.load_state_dict(model_dict["model"])
        self.__saliency_model.cuda()
        self.__saliency_model.eval()

    def push_frame(self, frame: torch.Tensor):
        self.__counter += 1

        frame_rgb = torch.clamp(convert_to_colorspace(frame, "yuv420", "rgb8"), 0, 1.0)

        with torch.no_grad():
            mask_pred, _ = self.__saliency_model(frame_rgb.unsqueeze(0))

            if self.__blur:
                mask_pred = gaussian_blur2d(mask_pred, (15, 15), (7.5, 7.5))

            _, _, h_2, w_2 = mask_pred.shape
            h = int(h_2 / 2)
            w = int(w_2 / 2)

            y = F.pixel_unshuffle(mask_pred, 2)
            u = F.interpolate(mask_pred, (h, w), mode="bilinear")
            v = F.interpolate(mask_pred, (h, w), mode="bilinear")

        self.__mask[self.__counter - 1] = torch.cat([y, u, v], 1)

        if not self.__display:
            return

        mask_yuv444 = colorspace_420_to_444(self.__mask[self.__counter - 1].unsqueeze(0)).squeeze(0)

        display_frame(mask_yuv444[0], "mask", 1)
        display_frame(frame_rgb, "original", 0)

    def compute(self) -> torch.HalfTensor:
        return self.__mask.to(MASKTYPE)
