import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, ConcatDataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from medpy.metric.binary import dc, jc, asd, hd
import pandas as pd
import random

import dataset
import utils
import models
import metrics
import train
import test
import argparse

# Setting
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    DATA_PATH = Path(args.data_path)
    MODEL_PATH =  Path(args.model_path)

    TEST_DOMAIN = "site1"
    LR = 0.1
    EPOCHS = 20
    LOAD_MODEL = False
    FOURIER_TRANSFORM = False

    ENCODER = "resnet50"
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MOMENTUM = 0.9

    # Loading data
    train_dataset, val_dataset, test_dataset = dataset.get_dataset(DATA_PATH, TEST_DOMAIN, TRAIN_SPLIT_RATIO)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # Logger
    log = {"train": {"loss": [], "dice_score": []},
           "val": {"dice_score": [], "asd": []}}

    # Train
    log = train.do_train(train_loader, val_loader, DEVICE, LOAD_MODEL, MODEL_PATH, TEST_DOMAIN, EPOCHS, LR, MOMENTUM,
                         FOURIER_TRANSFORM, train_dataset, ENCODER, log)

    # Ploting training log
    loss = log["train"]["loss"]
    train_sc_dice = [i[0] for i in log["train"]["dice_score"]]
    train_gm_dice = [i[1] for i in log["train"]["dice_score"]]
    train_log = utils.plot([loss, train_sc_dice, train_gm_dice], ["Loss", "SC DICE", "GM DICE"], "Train History")
    train_log.savefig("train_log.png")
    train_log.close()

    # Ploting val log
    val_sc_dice = [i[0] for i in log["val"]["dice_score"]]
    val_gm_dice = [i[1] for i in log["val"]["dice_score"]]
    val_log_dice = utils.plot([val_sc_dice, val_gm_dice], ["SC DICE", "GM DICE"], "Val Dice History")
    val_log_dice.savefig("val_log_dice.png")
    val_log_dice.close()

    val_sc_asd = [i[0] for i in log["val"]["asd"]]
    val_gm_asd = [i[1] for i in log["val"]["asd"]]
    val_log_asd = utils.plot([val_sc_asd, val_gm_asd], ["SC ASD", "GM ASD"], "Val ASD History")
    val_log_asd.savefig("val_log_asd.png")
    val_log_dice.close()


if __name__ == '__main__':
    main()