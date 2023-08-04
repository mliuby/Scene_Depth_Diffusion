import numpy as np
import matplotlib
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_from_checkpoint(ckpt,
                        model,
                        optimizer,
                        epochs,
                        loss_meter=None):

    checkpoint = torch.load(ckpt)
    ckpt_epoch = epochs - (checkpoint["epoch"]+1)
    if ckpt_epoch <= 0:
        raise ValueError("Epochs provided: {}, epochs completed in ckpt: {}".format(epochs, checkpoint["epoch"]+1))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

    return model, optimizer, ckpt_epoch

def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open( file ).resize((640, 480)), dtype=float) / 255, 0, 1).transpose(2, 0, 1)

        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)
def load_jpg(png):#'/content/drive/MyDrive/examples/depth7.png'
  pmg_image=np.clip(np.asarray(Image.open(png).resize((640, 480)), dtype=float) / 255, 0, 1)#.transpose(2, 0, 1)    
  return torch.tensor(pmg_image).unsqueeze(0) # return tensor(1,H,W)
def load_png(jpg):
  jpg_image=np.clip(np.asarray(Image.open(jpg).resize((640, 480)), dtype=float) / 255, 0, 1).transpose(2, 0, 1)
  return torch.tensor(jpg_image) # return tensor(3,H,W) 

reverse_transform = Compose([
     
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),#input is a tensor on cpu
     ToPILImage(),
])   
def init_or_load_model(epochs,
                        lr,
                        ckpt=None,
                        device=torch.device("cuda:0"),
                        loss_meter=None):

    if ckpt is not None:
        checkpoint = torch.load(ckpt)

    model = Unet()
    if ckpt is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ckpt is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    start_epoch = 0
    if ckpt is not None:
        start_epoch = checkpoint["epoch"]+1
        if start_epoch <= 0:
            raise ValueError("Epochs provided: {}, epochs completed in ckpt: {}".format(
                        epochs, checkpoint["epoch"]+1))

    return model, optimizer, start_epoch


def interpolation(rawDepths,delta_h=20):
    output_images = []
    for image in rawDepths:
        image=image.cuda()
        height, width = image.shape[1:3]
        # Edge filling
        input_image =image.clone()
      
        for row in range(delta_h, height - delta_h):
            nonzero_indices=(image[0, row, :] != 0).nonzero(as_tuple=True)[0]
            first_pixel_col = nonzero_indices.min()if nonzero_indices.numel() > 0 else 0
            last_pixel_col = nonzero_indices.max()if nonzero_indices.numel() > 0 else 0
            if first_pixel_col != last_pixel_col:
                input_image[0, row, :first_pixel_col] = image[0, row, first_pixel_col]
                input_image[0, row, last_pixel_col+1:] = image[0, row, last_pixel_col]
            else:  input_image[0, row, :]= image[0, row, first_pixel_col]
        
        for col in range(width):
            nonzero_indices=(input_image[0, :, col] != 0).nonzero(as_tuple=True)[0]
            first_pixel_row = nonzero_indices.min()if nonzero_indices.numel() > 0 else 0
            last_pixel_row = nonzero_indices.max()if nonzero_indices.numel() > 0 else 0
            if first_pixel_row != last_pixel_row:
                input_image[0, :first_pixel_row, col] = input_image[0, first_pixel_row, col]
                input_image[0, last_pixel_row+1:, col] = input_image[0, last_pixel_row, col]
            else:  input_image[0, :, col]= image[0,first_pixel_row,col]
        
        #Internal padding
        input_1d = input_image.reshape(-1)
        missing_pixels = input_image == 0
        missing_pixels_origin=image==0
        # Find missing and non-missing pixel indices
        missing_indices = torch.where(missing_pixels.flatten())[0]
        non_missing_indices = torch.where(~missing_pixels.flatten())[0]

        # Compute Manhattan distances between missing and non-missing pixels in batches
        batch_size = 1000
        n_missing = len(missing_indices)

        n_batches = int(torch.ceil(torch.tensor(n_missing / batch_size)))
        nearest_indices_1d = torch.empty(n_missing, dtype=int)
        
        for j in range(n_batches):
            start = j * batch_size
            end = min((j + 1) * batch_size, n_missing)
            batch_missing_indices = missing_indices[start:end]
            distances= torch.abs(batch_missing_indices.reshape(-1,1) - non_missing_indices.reshape(1,-1))
            distances=distances//width+distances%width
            nearest_indices_1d[start:end] = non_missing_indices[torch.argmin(distances, axis=1)]

        # Find 2D indices of nearest non-missing pixels

        nearest_indices_2d = (nearest_indices_1d // width, nearest_indices_1d % width)
        missing_indices_2d = (missing_indices // width, missing_indices % width)

        
        #Replace missing pixels with the values of the nearest non-missing pixels
        
        output_image = input_image.clone()
        output_image[0, missing_indices_2d[0], missing_indices_2d[1]] = input_image[0, nearest_indices_2d[0],nearest_indices_2d[1]]
        output_images.append(output_image)
        
    output_images=torch.stack(output_images)
    return output_images

