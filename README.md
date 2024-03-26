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
![plot](https://private-user-images.githubusercontent.com/37943739/311277642-9f465a27-50f0-40a9-b43f-69914f3bc3cb.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk5MTE2NjYsIm5iZiI6MTcwOTkxMTM2NiwicGF0aCI6Ii8zNzk0MzczOS8zMTEyNzc2NDItOWY0NjVhMjctNTBmMC00MGE5LWI0M2YtNjk5MTRmM2JjM2NiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA4VDE1MjI0NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIzYTY5ODQ0NTc5OWZlOWIyOWZiODEwZTM5NzU4ZWRmMmNiMDhkNmFhOGYxZGNmMjNhNzRiZmFlYzFkZDljOTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.9VUkeKL1qha61bXqnlxgkCMbNkfW--_FMjSlJBQGGJk)

2) Natural Language Processing:  
   1) Use the nlp environment created by run_setup_all.py
   2) [Project Kanban](https://github.com/users/bdzyubak/projects/4/views/1) 
3) Classical Machine Learning 
   1) Use the ml environment created by run_setup_all.py
   2) [Project Kanban](https://github.com/users/bdzyubak/projects/5/views/1)


## Available experiments 

1) Blood Vessel Segmentation
   
   a) Use the following script to [download and organize data](projects/ComputerVision/kaggle_blood_vessel_segmentation/organize_nnunet.py)

   b) Train [nnUnet](https://github.com/MIC-DKFZ/nnUNet) by calling "nnUNet/run_training.py 501 2d" (Dataset ID, architecture 
      template)


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