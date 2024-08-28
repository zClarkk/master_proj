import monai
from monai.networks import nets
from monai.losses.dice import DiceLoss
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import math
from torchinfo import summary
from monai.data import DataLoader
import numpy as np
from skimage.draw import disk

### Cuda Safety
if torch.cuda.is_available():
  print('CUDA GPU found: ', torch.cuda.get_device_name(['0']))
  device = 'cuda'
else:
  print('CUDA not found!')
  device = 'cpu'

### Dataset creation class
class xRayDataset(Dataset):
  SPLIT_IDX = 200
  num_landmarks = 166

  def __init__(self, mode: str):
    super().__init__()
    
    ## load data
    data_path = "/data_rechenknecht03_2/students/erguen/JSRT_img0_lms.pth"
    data = torch.load(data_path, map_location='cpu')
    self.data = data
    for key, value in self.data.items():
      if key == "JSRT_lms": self.lms = value#.float()
      if key == "JSRT_img0": self.img = value#.float()
    assert self.img.shape[0] == self.lms.shape[0]

    ## Train and Test selection
    if mode == 'train':
      self.img = self.img[:self.SPLIT_IDX]
      self.lms = self.lms[:self.SPLIT_IDX:]
    elif mode == 'test':
      self.img = self.img[self.SPLIT_IDX:]
      self.lms = self.lms[self.SPLIT_IDX:]
    else: raise ValueError(f'Unkown mode {mode}')

  def __len__(self): return self.img.shape[0]

  ## Get tuple of image and landmarks
  def __getitem__(self, idx):
    got_img = self.img[idx]
    got_lms = self.lms[idx]
    return got_img, got_lms

### Inits
lm_count = 0# needs to be 0 for below else 165 for test
patient_nr = 0
mask = torch.load("/home/erguen/Documents/monai-env/mask.pt")
# masks = torch.zeros(247, 166, 256, 256).to(device)
# vizMask = torch.zeros(256, 256)

### Create sets
train_dataset = xRayDataset('train')
test_dataset = xRayDataset('test')
img, lms = train_dataset[0]

def createMasks(dataset, patient_count=200, do_once=False):
  for patient_nr in (tqdm(range(patient_count))):
    lm_nr = 0
    img, lms = dataset[patient_nr]
    for lm in tqdm(lms):
      mask = torch.zeros(256, 256).to(device)
      x, y = disk((lm[0], lm[1]), 5)
      x = np.clip(x, 0, 255)
      y = np.clip(y, 0, 255)
      mask[y, x] = 1
      masks[patient_nr, lm_nr,... ] = mask
      lm_nr += 1
    if do_once: break
  return masks

def createMasks4visual(dataset, patient_count=200, do_once=False):
  for patient_nr in (tqdm(range(patient_count))):
    lm_nr = 0
    img, lms = dataset[patient_nr]
    for lm in tqdm(lms):
      x, y = disk((lm[0], lm[1]), 5)
      x = np.clip(x, 0, 255)
      y = np.clip(y, 0, 255)
      vizMask[y, x] = 1
      lm_nr += 1
    if do_once: break

def harryPlotter(image, landmarks, s_lm=1, c_lm="r"):
  implot = plt.imshow(image[:, :], cmap='gray')
  plt.scatter(landmarks[:, 0], landmarks[:, 1], s=s_lm, c=c_lm)
  plt.imshow(vizMask, cmap='jet', alpha=0.5)

def plotLoaded(mask, pat_nr, s_lm=1, c_lm="r"):
  viz_img, viz_lms = train_dataset[pat_nr]
  implot = plt.imshow(viz_img[:, :], cmap='gray')
  plt.scatter(viz_lms[:, 0], viz_lms[:, 1], s=s_lm, c=c_lm)
  viz_mask = mask[pat_nr].cpu()
  viz_mask[viz_mask==0]= np.nan
  for i in range(166):
    plt.imshow(viz_mask[i].cpu(), cmap='jet', alpha=0.5)

### Function calls
# createMasks(train_dataset, do_once=False)
# createMasks4visual(train_dataset, do_once=True)
# harryPlotter(img, lms)
# torch.save(createMasks(train_dataset, do_once=False), "mask.pt")
# plotLoaded(mask, pat_nr=150)

### Monai Unet
model = nets.UNet(
  spatial_dims=2,
  in_channels=1,
  out_channels=1,
  channels=(64, 128, 256, 512),
  strides=(2, 2, 2),
  kernel_size=3,
  up_kernel_size=3,
  num_res_units=2,
  act='LeakyReLU'
).to(device)

### Loss and Optimizer
# def TRE_loss(lms, lms_hat): # TRE_loss(landmarks, output, ord=2, dim=-1)
#   torch.linalg.vector_norm(lms - lms_hat, ord=2, dim=-1)
loss_function = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### NNetwork inits
epoch_loss_values = []
max_epochs = 10

### Training
for epoch in range(max_epochs):
  model.train()
  epoch_loss = 0
  step = 0
  print(epoch)
  for (batch_image, batch_lm), batch_mask in zip(train_dataset, mask):
    step += 1
    image, landmarks = batch_image.to(device), batch_mask.to(device)
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(image)
    # im = output.cpu().squeeze()
    # im = im.detach().numpy()
    # implot = plt.imshow(im[:,:], cmap='gray')
    # exit()
    loss = loss_function(output, landmarks)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  epoch_loss /= step
  epoch_loss_values.append(epoch_loss)

### Fragen:
