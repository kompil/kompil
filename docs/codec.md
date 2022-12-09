# Codec description

The codec is special in the sense that there is one model specifically trained to restitute one
segment of video. We have to do a training process for every chunk. However, compared to standard
deep learning practices, we don't require a general purpose dataset to train the model: only the
segment.

The models are quite simple. They take the frame ID to be restituted as an input and outputs the
actual image.

## Process

**Encoding**:
```
Raw video
>> training <<
float16 model
>> quantization <<
quantized model
>> packing <<
binary file
```
**Decoding**:
```
binary file
>> unpacking <<
quantized model
>> inference <<
decoded video.
```

# Improvement path

## Metrics

Just like any other codec, being able to have a great metric is always good to validate the
compression. However in our case we found some metrics seems not to work fine, for instance VMAF
does not look at the U and V channels, which in our case may be broken.

Simpler metrics like PSNR seems to work better but we can't rely on them too much.

## Training

There is a lot to do on this front and can imply big performances changes on other part of the
encoding process.

* **Model topology**: Historically, it has been the main performance lever. We call it the model's
topology.
    * **Input**: A mathematical way to transfer the frame ID to a defined code. This has performances implications, notably when we quantize the model.
    * **Layers**: All types of operations that should be differentiable.
    * **Colorspace**: This is how the image is represented at the end. For instance RGB or YUV420.
* **Loss**: Incredidly important for visual fidelity.
* **Optimizer**: How to optimize the learning to get faster and closer to the maximum capability of the
model.
* **Sheduler**: Same as the optimizer but by changing the learning rate. In kompil, we can use the
benchmark to choose the ight learning rate.
* **Pruning**: Can use bigger models but with less entropy so the entropy coding could reduce the
weight.
* Every other common training methods like variable batch size, precision

## Quantization

The quantization have a big impact on performances, the model, notably the input, has to be selected
carefully.

## Packing

Packing is the way the data will be organized in the final binary file. It includes the entropy
coding part as well.

**Note**: Right now the packing does not directly the entropy coding therefore we zip the packed
file to estimate the final size.