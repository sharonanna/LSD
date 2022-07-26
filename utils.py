import torch
import numpy as np
import torchvision
from dataset import RocksMap
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint_exp7_combined.pth.tar"):
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

#def check_accuracy(loader, model, slope_factor, device="cuda"):
def check_accuracy(loader, model, loss_fn, slope_factor, device="cuda"):

    num_correct=0
    num_pixels=0
    model.eval()


    with torch.no_grad():
        for x, y, s in loader:
            x = x.float().to(device)
            y = y.long().to(device)
            s = s.float().to(device)
            x = x + s*slope_factor
            outputs = model(x)
            probs = torch.softmax(outputs,dim=1)
            loss = loss_fn(outputs,y)
            preds = torch.argmax(probs, dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)


    accuracy = num_correct/num_pixels*100

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
    model.train()



# def save_predictions_as_imgs(loader, model, slope_factor, device="cuda"):

#     model.eval()
#     for idx, (x,y,s) in enumerate(loader):
#         x = x.float().to(device)
#         s = s.float().to(device)
#         x = x + s*slope_factor
#         with torch.no_grad():
#             probs = torch.softmax(model(x),dim=1)
#             preds = torch.argmax(probs, dim=1)
#             np.save('predictions_exp7_5/tm_org'+str(idx)+'.npy',x)
#             np.save('predictions_exp7_5/tm_pred'+str(idx)+'.npy',preds)
#             np.save('predictions_exp7_5/tm_y'+str(idx)+'.npy',y)


#     model.train()        