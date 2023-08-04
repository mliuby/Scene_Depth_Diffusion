import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os 
from PIL import Image
# RGBs. 
path_converted='./nyu_images'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)
f=h5py.File("nyu_depth_v2_labeled.mat")
images=f["images"]
images=np.array(images)  #images.shape=(1449, 3, 640, 480)
imageset=[] # images_number.shape=(1449, 3, 640, 480)
for i in range(len(images)):
    images_number.append(images[i])
    a=np.array(images_number[i])
    r = Image.fromarray(a[0]).convert('L')
    g = Image.fromarray(a[1]).convert('L')
    b = Image.fromarray(a[2]).convert('L')
    img = Image.merge("RGB", (r, g, b))
    img = img.transpose(Image.ROTATE_270)

#RawDepths.
f=h5py.File("nyu_depth_v2_labeled.mat")
rawDepths=f["rawDepths"]
rawDepths=np.array(rawDepths)
path_converted='./nyu_rawDepths/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)
max = rawDepths.max()
rawDepths = rawDepths/ max * 255
rawDepths = rawDepths.transpose((0,2,1))
for i in range(len(rawDepths)):
    rawDepths_img= Image.fromarray(np.uint8(rawDepths[i]))
    rawDepths_img = rawDepths_img.transpose(Image.FLIP_LEFT_RIGHT)
    iconpath=path_converted +'rawDepth'+str(i)+'.png'
    rawDepths_img.save(iconpath, 'PNG', optimize=True)

#NewDepths.
f=h5py.File("nyu_depth_v2_labeled.mat")
depths=f["depths"]
depths=np.array(depths)
path_converted='./nyu_depths/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)
max = depths.max()
depths = depths / max * 255
depths = depths.transpose((0,2,1))
for i in range(len(depths)):
    depths_img= Image.fromarray(np.uint8(depths[i]))
    depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
    iconpath=path_converted +'depth'+str(i)+'.png'
    depths_img.save(iconpath, 'PNG', optimize=True)