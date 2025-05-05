# Domain Generalization on Spinal Cord Gray Matter Segmentation




## Introduction

This project use the fourier-based  method proposed by [FACT](https://arxiv.org/abs/2105.11120]) to improve the domain generalization capbility of model in Medical MRI Segmentation task.

Dataset: The [SCGM (spinal cord gray matter segmentation)](http://niftyweb.cs.ucl.ac.uk/challenge/index.php) dataset contains Spinal Cord MRI data with gray matter labels from four different hospitals (UCL, Montreal, Zurich,
Vanderbilt) using three different MRI devices (Philips Acheiva, Siemens Trio, Siemens Skyra) but with hopital-specific parameters. Each hopital is thus a domain. 

## 需求 (Requirements)


## Run
`python main.py --data_path PATH_TO_DATASET_FOLDER --model_path PATH_TO_SAVED_MODL`

Note: the model path is needed only when you want to train from a checkpoint.

## Result


