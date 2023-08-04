import os
import argparse as arg
import time
import torchvision.transforms.functional as TF

import torch

import numpy as np
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt


def test(checkpoint,
        device="cuda",
        data="examples/"):

    if len(checkpoint) and not os.path.isfile(checkpoint):
        raise FileNotFoundError("{} no such file".format(checkpoint))

    device = torch.device("cuda" if device == "cuda" else "cpu")
    print("Using device: {}".format(device))

    # Initializing the model and loading the pretrained model
    ckpt = torch.load(checkpoint)
    model = Unet()

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    diffusion_model=DDPM(model,(1e-4, 0.02), 128, device, torch.nn.L1Loss())
    print("model load from checkpoint complete ...")

    # Get Test Images
    img_list = glob(data+"*.jpg")

    # Set model to eval mode
    model.eval()

    # Begin testing loop
    print("Begin Test Loop ...")

    fig, axs = plt.subplots(3, 2)
    for idx, img_name in enumerate(img_list):


        img = load_images([img_name])
        if idx < 3:
            axs[idx, 0].imshow(img[0].transpose(1, 2, 0))
            axs[idx, 0].axis('off')
        img = torch.Tensor(img).float().to(device)


        with torch.no_grad():
            preds_depth = diffusion_model.sample(img,device)
        output = preds_depth[0]
        output = output.cpu().numpy().transpose((1, 2, 0))


        if idx < 3:
            axs[idx, 1].imshow(output, cmap='gray')
            axs[idx, 1].axis('off')

        cv2.imwrite(img_name.split(".")[0].replace(data,data+'output/')+"_result.png", output*255)
        depth_saved = TF.to_pil_image(output*255).convert('L')
        depth_saved.save(f'depth_{idx}.png')
        print("Processing {} done.".format(img_name))
    plt.show()
    plt.savefig("output.png")
'''
test("/content/drive/MyDrive/checkpoints/checkpointsckpt_0_not_pretrained.pth",
        device="cuda",
        data="/content/drive/MyDrive/examples/")
'''        