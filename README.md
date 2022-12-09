<p align="center" display="block" margin-left="auto" margin-right="auto">
    <img alt="kompil Logo" src="docs/imgs/logo_full.png" width="50%" />
</p>

# Introduction

kompil is a new deep learning based video codec for video. The approach is a very special use case
of deep learning tool: it train video-specific models to restitute the video input.

This repository is meant to explore solutions so the codec may one day reach enough performance for
real applications.

*THIS CODE WASN'T MEANT FOR OPEN-SOURCING*. Any help to make it more friendly is welcome.

## Codec potential

This appoach implies a very long time to encode a video but result in a minimal model size to run
for decoding. This could hopefully one day allow decoders to use the Neural Processing Units
(deployed computer, phones and others) very efficiently instead of hardware dedicated to decoding.

To reach this application, the compression performances have to be improved.

It could also be used in deep learning training infrastructures to store datasets. In this way, the
data would be decoded seemlessly during the forward pass and therefore running by the same
accelerator and memory than the rest of the training.

## A bit of history

We - Valentin Macheret and Basile Collard - started the development in 2020 in the hope to start a
company around the codec technology. We worked on it full time. At the end of 2021 we didn't match
our performance targets and decided to pause the development. at the end of 2022 we decided to open
our code to the community in the hope it will slowly incease its performance and find its
application.

# Get started

* Clone the repository
```
> git clone <this-repo>
```
* Install the dependancies
```
> sudo apt install python pip3
> pip install -r requirements.txt
```
* Install cuda with the pytorch matching version: https://developer.nvidia.com/cuda-downloads
* Install VMAF: https://github.com/Netflix/vmaf/blob/master/libvmaf/README.md
* Compile the pytorch extensions
```
> cd extensions/
> python3 setup.py install
> cd ..
```
* Test encoding
```
> python kompil.py encode mock_320p --autoquit-epoch 10
```
* Play the result video (which is random)
```
> kompil play build/mock_320p.pth
```

## Hierarchy

* **benchmarks**: Some tools to help benchmarking individual functions like the extensions.
* **docs**: Set of description files to regroup all the info on how the codec works, how to improve
it and the articles to read.
* **extensions**: Pytorch extensions in order to speed up the code.
* **kompil**: Python library for the development of the codec.
* **scripts**: Set of scripts to help de development.
* **sota**: Project to regroup the best configuration video-specific to use the codec and draw
curves to help understand how it wen't.
* **unittests**: Tests.
* **kompil.py**: Entry point to the CLI.

## Documentation

* [Description of how the codec works](docs/codec.md)

## Half precision

To enable half precision training and tensor core:
* Install APEX :
```
> git clone https://github.com/NVIDIA/apex
> cd apex
> pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Profiling

To profile the code, one solution might be to externally use cProfile and flamegraph:
```
> pip3 install cProfile flameprof snakeviz
> python3 -m cProfile -o my_profile.prof kompil.py encode my_video.mp4
```
Then to display, few options:
```
> python3 -m flameprof my_profile.prof > my_profile.svg
> python3 -m snakeviz profile.prof
```

# License

This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for
details.

# Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to understand the process for submitting pull requests to us. Please respect our [Code of conduct](CODE_OF_CONDUCT.md).

# Authors

* Basile Collard
* Valentin Macheret