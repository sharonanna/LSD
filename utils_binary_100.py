import torch
import numpy as np
import torchvision
from dataset import RocksMap
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint_exp7_100_5.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, slope_dir, batch_size, train_transform, val_transform, num_workers=0, pin_memory=False):
    
    train_ds = RocksMap(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        slope_dir=slope_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = RocksMap(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        slope_dir=slope_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        #batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, loss_fn, slope_factor, device="cuda"):
    
    # losses = []
    # acc = []
    num_correct=0
    num_pixels=0
    #running_loss = 0
    model.eval()
    dice_score = 0

    with torch.no_grad():
        for x, y, s in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            s = s.float().to(device)
            x = x + s*slope_factor
            outputs = model(x)
            preds = torch.sigmoid(outputs)
            #loss = loss_fn(preds,y)
            #running_loss+=loss.item()
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2* (preds*y).sum())/((preds+y).sum()+1e-8)

    accuracy = num_correct/num_pixels*100
    # test_loss = running_loss/len(loader)
    # losses.append(test_loss)
    # acc.append(accuracy)
    
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
    print(f"DiceScore: {dice_score/len(loader)}")
    model.train()

    #return losses, acc        


# def save_predictions_as_imgs(loader, model, slope_factor, device="cuda"):

#     model.eval()
#     for idx, (x,y,s) in enumerate(loader):
#         x = x.float().to(device)
#         s = s.float().to(device)
#         x = x + s*slope_factor
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             np.save('predictions_exp7_100_5/tm_pred'+str(idx)+'.npy',preds)
#             np.save('predictions_exp7_100_5/tm_y'+str(idx)+'.npy',y)


#     model.train()


            