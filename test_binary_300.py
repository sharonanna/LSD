import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model_binary import UNET
import numpy as np
import datetime
from torchsummary import summary
from torch.utils.data import DataLoader
from test_image_loader import RocksMap


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 3
NUM_WORKERS = 2
SLOPE_FACTOR = 5
IMAGE_HEIGHT = 300  
IMAGE_WIDTH = 300  
PIN_MEMORY = False
LOAD_MODEL = True
TEST_IMG_DIR = "test_data_B300/test_images"
TEST_IMG_MASK = "test_data_B300/test_masks"
SLOPE_DIR = "test_data_B300/slope_images"


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, slope_factor, device="cuda"):

    num_correct=0
    num_pixels=0
    model.eval()
    dice_score=0

    begin_time_preds = datetime.datetime.now()

    with torch.no_grad():
        for x,y,s in loader:
            
            x = x.float().to(device)
            y = y.float().to(device)
            s = s.float().to(device)
            x = x + s*slope_factor
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2* (preds*y).sum())/((preds+y).sum()+1e-8)
    
    print(f"PREDS ONLY: {datetime.datetime.now() - begin_time_preds}")
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"DiceScore: {dice_score/len(loader)}")
    print(f"SLOPE FACTOR{slope_factor}")
    model.train()   




def save_predictions_as_imgs(loader, model, slope_factor, device="cuda"):

    model.eval()
    for idx, (x,y,s) in enumerate(loader):
        x = x.float().to(device)
        s = s.float().to(device)
        x = x + s*slope_factor
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            np.save('test_predictions_binary_300/tm_org'+str(idx)+'.npy',x)
            np.save('test_predictions_binary_300/tm_preds'+str(idx)+'.npy',preds)
            np.save('test_predictions_binary_300/tm_y'+str(idx)+'.npy',y)

def get_loaders_test(test_dir, test_maskdir, slope_dir,  batch_size, num_workers=2, pin_memory=True):
    
    test_ds = RocksMap(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        slope_dir=slope_dir,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader



def main():

    begin_time = datetime.datetime.now()
    #print(begin_time)

    model = UNET(in_channels=1, out_channels=1).float().to(DEVICE)
    #summary(model,(1,200,200))


    test_loader = get_loaders_test(TEST_IMG_DIR, TEST_IMG_MASK, SLOPE_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint_exp6_300_5.pth.tar"), model)

    check_accuracy(test_loader, model,slope_factor=SLOPE_FACTOR, device=DEVICE)

    print(f"MAIN LOOP: {datetime.datetime.now() - begin_time}")
        # print some examples to a folder
    save_predictions_as_imgs( 
        test_loader, model,slope_factor=SLOPE_FACTOR, device=DEVICE
    )



if __name__ == "__main__":
    main()