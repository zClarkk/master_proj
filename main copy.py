import monai
from monai.networks import nets
from monai.losses.dice import GeneralizedDiceLoss
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
from monai.networks.utils import one_hot

### Cuda Safety
if torch.cuda.is_available():
  print('CUDA GPU found: ', torch.cuda.get_device_name(['0']))
  device = 'cuda'
else:
  print('CUDA not found!')
  device = 'cpu'

### Dataset creation class
class xRayDataset(Dataset):
  num_landmarks = 166
  def __init__(self, mode: str, split: int):
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
      self.img = self.img[:split]
      self.lms = self.lms[:split:]
    elif mode == 'test':
      self.img = self.img[split:]
      self.lms = self.lms[split:]
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
split = 200
train_mask = torch.load("/home/erguen/Documents/monai-env/train_mask.pt")
test_mask = torch.load("/home/erguen/Documents/monai-env/test_mask.pt")

### Create sets
train_dataset = xRayDataset('train', split)
test_dataset = xRayDataset('test', split)
img, lms = train_dataset[0]

def createMasks(dataset, do_once=False):
  if dataset == train_dataset:
    masks = torch.zeros(split, 166, 256, 256, dtype=torch.uint8).to(device)
    patient_count = split
  if dataset == test_dataset:
    masks = torch.zeros(247-split, 166, 256, 256, dtype=torch.uint8).to(device)
    patient_count = 247 - split
  for patient_nr in (tqdm(range(patient_count))):
    lm_nr = 0
    img, lms = dataset[patient_nr]
    for lm in tqdm(lms):
      mask = torch.zeros(256, 256, dtype=torch.uint8).to(device)
      x, y = disk((lm[0], lm[1]), 5)
      x = np.clip(x, 0, 255)
      y = np.clip(y, 0, 255)
      mask[y, x] = 1
      masks[patient_nr, lm_nr,... ] = mask
      lm_nr += 1
    if do_once: break
  return masks

def plotLoaded(seg, pat_nr=0, s_lm=1, c_lm="r"):
  viz_mask = seg[0] #.to(float) #.cpu()
  viz_mask[viz_mask==0] = np.nan
  numbah = 0
  for i in range(166):
    if numbah+20 < 166:
      plt.imshow(viz_mask[numbah], cmap='grey', alpha=1)
      plt.show()
      numbah += 20

### Monai Unet
model = nets.UNet(
  spatial_dims=2,
  in_channels=1,
  out_channels=166,
  channels=(16, 32, 64, 128, 256),
  strides=(2, 2, 2, 1),
  num_res_units=2,
  act='LeakyReLU',
  norm='batch'
).to(device)

### Loss and Optimizer
# def TRE_loss(lms, lms_hat): # TRE_loss(landmarks, output, ord=2, dim=-1)
#   torch.linalg.vector_norm(lms - lms_hat, ord=2, dim=-1)
loss_function = GeneralizedDiceLoss(softmax=True, include_background=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
inferer = monai.inferers.SimpleInferer()
dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean")

### NNetwork inits
epoch_loss_values = []
max_epochs = 5
val_interval = 2

def validation():
  model.eval()
  with torch.no_grad():
    val_images = None
    val_labels = None
    val_outputs = None
  step = 0
  for (val_image, val_lm), val_masks in zip(test_dataset, test_mask):
    step += 1
    val_image, val_mask = val_image.to(device).float(), val_masks.to(device)
    val_image = torch.unsqueeze(val_image, 0)
    val_image = torch.unsqueeze(val_image, 0)
    val_outputs = inferer(inputs=val_image, network=model)
    dice_metric(y_pred=val_outputs, y=val_mask)
  metric = dice_metric.aggregate().item()
  dice_metric.reset()
  print(metric)
  return val_outputs

def train():
  for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    ### loop over patients, all landmarks for the patient at once
    for (train_image, train_lm), train_masks in zip(train_dataset, train_mask):
      step += 1
      image, lm_mask = train_image.to(device), train_masks.to(device)
      image = torch.unsqueeze(image, 0)
      image = torch.unsqueeze(image, 0)

      optimizer.zero_grad()
      with torch.cuda.amp.autocast( dtype=torch.float16):
        output = model(image)
      # print(output.shape[1])
      loss = loss_function(output[0], lm_mask)
      loss.backward()
      optimizer.step()
    
      epoch_loss += loss.item()
      epoch_len = len(train_dataset) // 1
      # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    # epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if (epoch + 1) % val_interval ==0:
      model.eval()
      with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
      for (val_image, val_lm), val_masks in zip(test_dataset, test_mask):
        val_image, val_mask = val_image.to(device).float(), val_masks.to(device)
        val_image = torch.unsqueeze(val_image, 0)
        val_image = torch.unsqueeze(val_image, 0)
        val_outputs = inferer(inputs=val_image, network=model)
        dice_metric(y_pred=val_outputs, y=val_mask)
      metric = dice_metric.aggregate().item()
      dice_metric.reset()
      print(metric)
  return output, val_outputs

result, val = train()


### Plots the segmentation for patient 200 in the last epoch
cpu_out = val.cpu()
detached = cpu_out.detach().numpy()
print(detached.shape)
plotLoaded(detached)
