import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 15
NUM_WORKERS = 0
IMAGE_HEIGHT = 200  # 1280 originally
IMAGE_WIDTH = 200  # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "training_data/train_images/"
TRAIN_MASK_DIR = "training_data/train_masks/"
VAL_IMG_DIR = "training_data/val_images/"
VAL_MASK_DIR = "training_data/val_masks/"
# TRAIN_IMG_DIR = "data/train_data/"
# TRAIN_MASK_DIR = "data/train_mask/"
# VAL_IMG_DIR = "data/val_data/"
# VAL_MASK_DIR = "data/val_mask/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        # targets = targets.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        print(f"targets shape:{targets.shape}")

        # forward
        with torch.cuda.amp.autocast():
            print(f"data shape{data.shape}")
            predictions = model(data)
            print(f"predictions shape: {predictions.shape}")
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
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

    model = UNET(in_channels=1, out_channels=3).float().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )


    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint_2.pth.tar"), model)


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch:{epoch}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="predictions_exp2/", device=DEVICE
        )


if __name__ == "__main__":
    main()