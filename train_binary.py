import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from model_binary import UNET
from utils_binary import (
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
SLOPE_FACTOR = 14
IMAGE_HEIGHT = 200  # 1280 originally
IMAGE_WIDTH = 200  # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "training_data_binary/train_images/"
TRAIN_MASK_DIR = "training_data_binary/train_masks/"
VAL_IMG_DIR = "training_data_binary/val_images/"
VAL_MASK_DIR = "training_data_binary/val_masks/"
SLOPE_DIR = "training_data_binary/slope_images"


def train_fn(loader, model, optimizer, loss_fn, scaler, slope_factor=14):
    loop = tqdm(loader)
    #running_loss = 0.0
    #losses = []

    for batch_idx, (data, targets, slope) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        slope = slope.float().to(device=DEVICE)
        data = data + slope*slope_factor
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        #targets = targets.long().to(device=DEVICE)
        print(f"targets shape:{targets.shape}")

        # forward
        with torch.cuda.amp.autocast():
            print(f"data shape{data.shape}")
            predictions = model(data)
            print(f"predictions shape: {predictions.shape}")
            print(f"Slope Factor:{slope_factor}")
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #running_loss += loss.item() * data.size(0) 

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    # epoch_loss = running_loss / len(loader)
    # losses.append(epoch_loss)

    #return losses
 


def main():

    #losses = val_losses= val_accuracy= []

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

    model = UNET(in_channels=1, out_channels=1).float().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(summary(model,(1,200,200)))

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
        load_checkpoint(torch.load("checkpoint_binary_14.pth.tar"), model)


    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch:{epoch}")
        #losses = train_fn(train_loader, model, optimizer, loss_fn, scaler, slope_factor=SLOPE_FACTOR)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, slope_factor=SLOPE_FACTOR)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        #val_losses, val_accuracy = check_accuracy(val_loader, model, loss_fn, slope_factor=SLOPE_FACTOR, device=DEVICE)
        check_accuracy(val_loader, model, loss_fn, slope_factor=SLOPE_FACTOR, device=DEVICE)


        # print some examples to a folder
        save_predictions_as_imgs( 
            val_loader, model,slope_factor=SLOPE_FACTOR, device=DEVICE
        )

    # plt.plot(losses)
    # plt.plot(val_losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend(['Train','Validation'])
    # plt.title('Train vs Validation Loss')
    # plt.show()

    # plt.plot(val_accuracy)
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(['Validation'])
    # plt.title('Accuracy')
    # plt.show()





if __name__ == "__main__":
    main()