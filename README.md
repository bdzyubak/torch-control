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
vision tasks. I also have a great interest in exploring Natural Language Models, the cutting edge of AI. 

The repository fairly recent, started on in March 2024. Currently, it can serve as code samples, with more thorough 
experimnetation, model complexity, and MLOps to be added over the subsequent months. 

This repo is in Pytorch. For Tensorflow, see: [tensorflow-sandbox](https://github.com/bdzyubak/tensorflow-sandbox)

## Installation
Install the following prerequisites:
1) Anaconda (>=2023) 
2) Version control and git (e.g. GitKraken)
3) Use run_setup_all.py to install all or some (with command line arguments) of the environments required to run 
projects (see [Repository Organization](#repository-organization)) 
4) In Pycharm settings, define all submodules torch-control/utils, torch-control/nnUnet as Source. For running in the 
command line, these conda paths are developed by run_setup_all.py, but Pycharm overrides the system settings 
5) MLFlow: 
   1) Install git if not installed already and add it to PATH (GitKraken does not appear to add a callable git executable)
   2) Configure the mlflow runs directory by changing the environment variable e.g. MLFLOW_TRACKING_URI=D:\Models\mlruns 
   via the control panel on Windows or .bashrc update on Linux. 
   3) To display logs, navigate to the mlruns folder in the terminal and run: mlflow ui --port 8080
   4) Then access via browser: localhost:8080
6) Docker: 
   1) Either install docker desktop (which includes the dependencies above, or install Engine/CLI/Compose separately 
   to avoid bloat https://docs.docker.com/engine/
   2) To install Docker desktop to a custom location, go to the download directory and run the following in command 
   prompt: start /w “” “Docker Desktop Installer.exe” install --installation-dir=D:\Docker)
   3) Install gcc with apt-get on Linux, or MinGW on Windows 
   https://dev.to/gamegods3/how-to-install-gcc-in-windows-10-the-easier-way-422j. MinGW must be installed in default 
   location or it will be missing files


## Repository Organization

This repository is a top-level controller for running training/inference on a variety of forked AI repos to compare 
performance of architectures/training methods. The /projects folder contains entrypoints for experiments on the 
following topics: 
1) Computer vision: 
   1) Use the cv environment created by run_setup_all.py
   2) [Project Kanban](https://github.com/users/bdzyubak/projects/2/views/1)

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

1) [NLP] Movie Sentiment Analysis
   
   a) Fine-tuned Distilbert on Kaggle Movie Sentiment analysis dataset. ![plot](/projects/NaturalLanguageProcessing/MovieReviewAnalysis/training_metrics_version_7.png)

   b) WIP: Fine-tune other common networks for comparison

   c) WIP: Evaluate freezing all but sentiment analysis-head layers. 

   d) WIP: Combine with IMDB reviews dataset - compare training on one, validation on the other, then randomly split for cross-val.


2) [CV] Blood Vessel Segmentation
   
   a) Use the following script to [download and organize data](/projects/ComputerVision/kaggle_blood_vessel_segmentation/organize_nnunet.py)

   b) Train [nnUnet](https://github.com/MIC-DKFZ/nnUNet) by calling "nnUNet/run_training.py 501 2d" (Dataset ID, architecture 
      template)


3) [ML] time series segmentation: 

   a) Use the following script for hyperparameter optimization and model fitting ![plot](/projects/MachineLearning/energy_use_time_series_forecasting/xgboost_depth-10_rmse-1658.3_lr-0.001.png)

   b) WIP: Add MLOps for model/hyperparameter/data tracking
   
   c) WIP: Further improve model hyperparameters and engineered features.

   d) WIP: Add cross validation with datasets from the other companies available. 

4) MLOps - MLflow, Docker, Cloud: 
   
   a) Build and tests successful for [ML](projects/MachineLearning/energy_use_time_series_forecasting/build_inference_docker_container.py)
   
   b) TODO: Build and container

   c) TODO: Deploy to AWS 

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
