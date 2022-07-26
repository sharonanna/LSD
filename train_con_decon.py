import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
import datetime
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from dataset import RocksMap
from model import conv_deconv #Class where the network is defined
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 15
NUM_WORKERS = 0
SLOPE_FACTOR = 10
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "training_data/train_images/"
TRAIN_MASK_DIR = "training_data/train_masks/"
VAL_IMG_DIR = "training_data/val_images/"
VAL_MASK_DIR = "training_data/val_masks/"
SLOPE_DIR = "training_data/slope_images"


def train_fn(loader, model, optimizer, loss_fn, scaler, slope_factor):
    loop = tqdm(loader)
    writer = SummaryWriter()
    iter = 0

    for batch_idx, (data, targets, slope) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        slope = slope.float().to(device=DEVICE)
        data = data + slope*slope_factor
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        #print(f"targets shape:{targets.shape}")

        # forward
        with torch.cuda.amp.autocast():
            #print(f"data shape{data.shape}")
            predictions = model(data)
            #print(f"predictions shape: {predictions.shape}")
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        writer.add_scalar('Training Loss',loss.item(), iter)
        iter+=1
        # update tqdm loop
        loop.set_postfix(loss=loss.item())


    writer.close()


def main():

    begin_time = datetime.datetime.now()
    print(f"SLOPE FACTOR:{SLOPE_FACTOR}")
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            #ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #ToTensorV2(),
        ],
    )

    model = conv_deconv().float().to(DEVICE)
    #print(summary(model,(1,200,200)))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        SLOPE_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint_exp7_10.pth.tar"), model)


    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch:{epoch}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, slope_factor=SLOPE_FACTOR)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, loss_fn, slope_factor=SLOPE_FACTOR, device=DEVICE)

        # save_predictions_as_imgs( 
        #     val_loader, model,slope_factor=SLOPE_FACTOR, device=DEVICE
        # )

    print(datetime.datetime.now() - begin_time)

if __name__ == "__main__":
    main()

##############################################################################################################################################
      