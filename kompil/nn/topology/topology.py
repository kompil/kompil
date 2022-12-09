import kompil.nn.topology.builder as builder


### Input layers
in_eye = builder.EyeInput.elem
in_graycode = builder.GrayCodeInput.elem
in_incremental_graycode = builder.IncrementalGrayCodeInput.elem
in_binary = builder.BinaryInput.elem
in_hybrid = builder.HybridInput.elem
in_binary_sinus = builder.BinarySinusInput.elem
in_binary_triangle = builder.BinaryTriangleInput.elem
in_binary_tooth = builder.BinaryToothInput.elem
autoflow = builder.AutoFlow.elem
time_slider = builder.TimeSlider.elem
in_regularized_graycode = builder.RegularizedGrayCodeInput.elem

### Weightfull layers
linear = builder.Linear.elem
conv1d = builder.Conv1d.elem
conv2d = builder.Conv2d.elem
conv3d = builder.Conv3d.elem
deconv1d = builder.Deconv1d.elem
deconv2d = builder.Deconv2d.elem
deconv3d = builder.Deconv3d.elem
adjacent1d = builder.Adjacent1d.elem
adjacent2d = builder.Adjacent2d.elem
invol2d = builder.Invol2d.elem

### Activation layers
relu = builder.ReLU.elem
xelu = builder.XeLU.elem
hardpish = builder.HardPish.elem
lrelu = builder.LeakyReLU.elem
mprelu = builder.PReLU.multi
drelu = builder.DiscretizedPReLU.elem
prelu = builder.PReLU.single
prelub = builder.PReLUB.elem
discretize = builder.Discretize.elem
mish = builder.Mish.elem
pish = builder.Pish.elem
spish = builder.SPish.elem
wish = builder.Wish.elem
out_asymp = builder.AsymParamOutput.elem

### Composite layers
resblock = builder.ResBlock.elem
subpixelconv2d = builder.SubpixelConv2d.elem
subblockconv2d = builder.SubblockConv2d.elem

### Standard operations layers
sequence = builder.Sequence.elem
add = builder.Add.elem
mul = builder.Mul.elem
sum = builder.Sum.elem
softmax = builder.Softmax.elem
concat = builder.Concat.elem
hsv_to_rgb = builder.HsvToRgb.elem
pixel_shuffle = builder.PixelShuffle.elem
pixel_unshuffle = builder.PixelUnshuffle.elem
channel_upscale = builder.ChannelUpscale.elem
upsample = builder.UpSample.elem
reshape = builder.Reshape.elem
permute = builder.Permute.elem
batchnorm1d = builder.BatchNorm1d.elem
batchnorm2d = builder.BatchNorm2d.elem
pixelnorm = builder.PixelNorm.elem
groupnorm = builder.GroupNorm.elem
weightnorm = builder.WeightNorm.elem
instancenorm = builder.InstanceNorm.elem
maxpool2d = builder.MaxPool2d.elem
avgpool2d = builder.AvgPool2d.elem
crop2d = builder.Crop2d.elem
colorspace_420_to_444 = builder.Yuv420ToYuv444.elem
yuv_to_rgb = builder.YuvToRgb.elem
yuv420_astride = builder.Yuv420astride.elem
fourier_feature = builder.FourierFeature.elem
normal_noise = builder.Noise.normal
index_select = builder.IndexSelect.elem

### Dev layers
prune = builder.Prune.elem
repeat = builder.Repeat.elem
conv_module = builder.ConvModule.elem
load = builder.Load.elem
save = builder.Save.elem
switch = builder.Switch.elem
switch_indexed = builder.SwitchIndexed.elem
quantize_stub = builder.QuantizeStub.elem
fixed_quantize_stub = builder.FixedQuantizeStub.elem
dequantize_stub = builder.DeQuantizeStub.elem
context_save = builder.ContextSave.elem
devlayer = builder.DevLayer.elem

### Tools
def duplicate(sequence: list, duplication: int) -> dict:
    assert int(duplication)

    duplications = []

    if not isinstance(sequence, list):
        sequence = [sequence]

    for _ in range(duplication):
        for mod in sequence:
            duplications.append(mod)

    return duplications
