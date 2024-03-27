## Deep Learning Sandbox 

Author: Bogdan Dzyubak, PhD

Email: illan7@gmail.com

Date: 02/26/2024

Repository: [torch-control](https://github.com/bdzyubak/torch-control)


## Purpose:

The purpose of this project is to explore a wide variety of neural networks and training/inference/preprocessing 
methods. To that end, I am forking repositories with state-of-the-art architectures, improving 
interfaces, and adding ways to mix-and-match architectures and training/inference methods. My main background is in 
medical image analysis. Consequently, to expand horizons, I will be applying image analysis models to other computer
vision tasks. I also have a great interest in exploring Natural Language Models, the cutting edge of AI. The repository 
is fairly recent, started on 02/26/2024, and is a work in progress. This repo is in Pytorch. For an older but more 
developed repo see: [tensorflow-sandbox](https://github.com/bdzyubak/tensorflow-sandbox)

## Installation
Install the following prerequisites:
1) Anaconda (>=2023) 
2) Version control and git (e.g. GitKraken)
3) Use run_setup_all.py to install all or some (with command line arguments) of the environments required to run 
projects (see [Repository Organization](#repository-organization)) 
4) In Pycharm settings, define all submodules torch-control/utils, torch-control/nnUnet as Source. For running in the 
command line, these conda paths are developed by run_setup_all.py, but Pycharm overrides the system settings 
5) To conserve space when downloading NLP models, enable symlinks for cached models 
([Windows 10](https://consumer.huawei.com/en/support/content/en-us15594140/#:~:text=Click%20the%20Windows%20icon%20and,displayed%20dialog%20box%2C%20select%20Yes), 
[Windows11](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)) 

[//]: # (4&#41; Docker is currently unused, but for futrue reference)
[//]: # (   a&#41; To install in custom location, use: start /w “” “Docker Desktop Installer.exe” install --installation-dir=G:\Docker)


## Repository Organization

This repository is a top-level controller for running training/inference on a variety of forked AI repos to compare 
performance of architectures/training methods. The /projects folder contains entrypoints for experiments on the 
following topics: 
1) Computer vision: 
   1) Use the cv environment created by run_setup_all.py
   2) [Project Kanban](https://github.com/users/bdzyubak/projects/2/views/1) 
   3) <span style="color:teal">Sample nnUnet training curve</span>\
The loss is negative because dice loss is defined as (-dice) not (1-dice)
![plot](/projects/ComputerVision/kaggle_blood_vessel_segmentation/sample_training_curve_for_README.png)

2) Natural Language Processing:  
   1) Use the nlp environment created by run_setup_all.py
   2) [Project Kanban](https://github.com/users/bdzyubak/projects/4/views/1) 
3) Machine Learning 
   1) Use the ml environment created by run_setup_all.py
   2) [Project Kanban](https://github.com/users/bdzyubak/projects/5/views/1)

Utils is a submodule repository which contains base level library code for interacting with models, the OS, plotting 
etc, which can be imported by other repositories such as tensorflow-sandbox, or forked by users separately from 
torch-control. 


## Available experiments 

1) CV: Blood Vessel Segmentation
   
   a) Use the following script to [download and organize data](projects/ComputerVision/kaggle_blood_vessel_segmentation/organize_nnunet.py)

   b) Train [nnUnet](https://github.com/MIC-DKFZ/nnUNet) by calling "nnUNet/run_training.py 501 2d" (Dataset ID, architecture 
      template)
2) NLP: [Fun with sentiment analysis](projects/NaturalLanguageProcessing/LLMTutorialsHuggingFace/transformers_sentiment_analysis.py)
3) ML: [Semi-supervised learning](projects/MachineLearning/semi_supervised_breast_cancer_classification/semi_supervised_svm.py)

## Operating Systems Notes 

1) This project was developed on Windows. It attempts to be OS agnostic - no hardcoded slashes, reliance on 
  python tools instead of system tools - but testing primarily happens on Windows, so Linux patches will probably be 
  needed. 

2) The project is aimed to work with both CPU only and GPU-capable setups. It is tested on a GTX 3060 GPU with 12 Gig 
  of memory. GPU memory allocation is satic. If you run into an out of memory issue, reduce batch size. 

## Bug reporting/feature requests

Feel free to raise an issue or, indeed, contribute to solving one (!) on: https://github.com/bdzyubak/torch-control/issues

## Testing Installation: 

1) Computer Vision: 
   1) TODO - add integration test that does not require download of a large dataset. 
2) Natural Language Processing: 
   1) projects/NaturalLanguageProcessing/LLMs_tutorials/distilbert_question_answering.py
3) Machine Learning: 
   1) projects/MachineLearning/semi_supervised_breast_cancer_classification/semi_supervised_svm.py


## Testing and Release Process: 
1) Unit: 
   1) Run pytest on teh following folder tests/unit. 
   2) Test coverage is a WIP
2) The master branch is a stable beta where unit tests should all pass and features are reference compatible after 
 every merge. Due to the single user nature of this repo, currently a Release branch is not planned. 
