## Deep Learning Sandbox 

Author: Bogdan Dzyubak, PhD

Email: illan7@gmail.com

Date: 02/26/2024

Repository: [torch-control](https://github.com/bdzyubak/torch-control)


## Purpose:

The purpose of this project is to explore a wide variety of neural networks and training/inference/preprocessing 
methods. To that end, I am forking repositories with state-of-the-art architectures, improving 
interfaces, and adding ways to mix-and-match architectures and training/inference methods. My main background is in 
medical image analysis. Consequently, to expand horizons, I will be applying image analysis models to non-MRI/CT/PET 
images, and also exploring Natural Language Processing + GANs.


## Installation
Install the following prerequisites:
1) Anaconda (>=2023) 
2) Version control and git (e.g. GitKraken)
3) [Pytorch](https://pytorch.org/get-started/locally/) with GPU support, if desired
   a) e.g. conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   b) Separate cuDNN, zlib, CUDA drivers installation is not required with this method.
   c) The procedure fails on Linux for me. CUDA drivers need to be pre-installed and it has to be done using the 
      runfile method, e.g. https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
4) Conda install opendasets (to access Kaggle Competition data used in this project)
5) To run a given sub-repo, run the run_setup.py file in it to create a conda environment, and select it as the 
interpreter. The utilities contains shared libraries - its dependencies are built into those of other repos.  
6) Docker 
   a) To install in custom location, use: start /w “” “Docker Desktop Installer.exe” install --installation-dir=G:\Docker
7) In Pycharm settings, define all submodules torch-control/utils, torch-control/nnUnet as Source. Alternatively, use 
   conda develop [each_full_path]

## Repository Organization 

This repository is a top-level controller for running training/inference on a variety of forked AI repos to compare 
performance of architectures/training methods. It contains largely contains entrypoint scripts. The utils submodule 
contains shared functions that can be imported by the controller or any of the AI repos. Each AI sub repo requires its 
own conda environment to run, created by the relevant installer.

## Operating Systems Notes 

1) This project was developed on Windows. It attempts to be OS agnostic - no hardcoded slashes, reliance on 
  python tools instead of system tools - but testing primarily happens on Windows, so Linux patches will probably be 
  needed. 

2) The project is aimed to work with both CPU only and GPU-capable setups. It is tested on a GTX 3060 GPU with 12 Gig 
  of memory. GPU memory allocation is satic. If you run into an out of memory issue, reduce batch size. 

## Bug reporting/feature requests

Feel free to raise an issue or, indeed, contribute to solving one (!) on: https://github.com/bdzyubak/torch-control/issues

## Testing Installation: 

TODO: add integration test with small epoch/batch size to validate setup of each submodule. 

## Testing and Release: 

TODO: add unit-testing and ability to run them at top level to promote to a new release

TODO: add regression testing to ensure that changes to inference code do not change model outputs
