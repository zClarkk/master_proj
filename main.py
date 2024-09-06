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
  vizMask = torch.zeros(256, 256, dtype=torch.uint8).to(device)
  implot = plt.imshow(image[:, :], cmap='gray')
  plt.scatter(landmarks[:, 0], landmarks[:, 1], s=s_lm, c=c_lm)
  plt.imshow(vizMask, cmap='jet', alpha=0.5)

def plotLoaded(mask, pat_nr, s_lm=1, c_lm="r"):
  viz_img, viz_lms = train_dataset[pat_nr]
  # implot = plt.imshow(viz_img[:, :], cmap='gray')
  # plt.scatter(viz_lms[:, 0], viz_lms[:, 1], s=s_lm, c=c_lm)
  # viz_mask = mask[pat_nr]#.to(float) #.cpu()
  viz_mask = mask[0] #.to(float) #.cpu()
  viz_mask[viz_mask==0] = np.nan
  for i in range(166):
    plt.imshow(viz_mask[i], cmap='grey', alpha=0.5)

def functionCaller(selector, file_name, dataset):
  match selector:
    case "createMasks":
      torch.save(createMasks(dataset, do_once=False), file_name)
    case "createMasks4visual": createMasks4visual(train_dataset, do_once=True)
    case "harryPlotter": harryPlotter(img, lms)
    case "plotLoaded": plotLoaded(train_mask, pat_nr=14)

### Function calls
functionCaller("asd", file_name="train_mask.pt", dataset=train_dataset)
functionCaller("asd", file_name="test_mask.pt", dataset=test_dataset)

### Monai Unet
model = nets.UNet(
  spatial_dims=2,
  in_channels=166,
  out_channels=166,
  channels=(16, 32, 64, 128, 256),
  strides=(2, 2, 2, 2),
  # kernel_size=3,
  # up_kernel_size=3,
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
val_interval = 2
county = 0

def validation():
  model.eval()
  with torch.no_grad():
    val_images = None
    val_labels = None
    val_outputs = None
  for (val_image, val_lm), val_mask in zip(test_dataset, test_mask):
    val_images, val_labels = None

for epoch in range(1):
  print("-" * 10)
  print(f"epoch {epoch + 1}/{10}")
  model.train()
  epoch_loss = 0
  step = 0
  for (batch_image, batch_lm), batch_mask in zip(train_dataset, train_mask):
    step += 1
    image, lm_mask = batch_image.to(device), batch_mask.to(device)
    optimizer.zero_grad()
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    image = image.expand(-1, 166, -1, -1)
    lm_mask = torch.unsqueeze(lm_mask, 0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
      output = model(torch.cat((image, lm_mask)))
    loss = loss_function(output[0], batch_mask)
    # loss += loss_function(output[1], batch_mask)
    # loss = loss / 2
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_len = len(train_dataset) // 1
    print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    county += 1
  epoch_loss /= step
  epoch_loss_values.append(epoch_loss)
  print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
  
  # if (epoch + 1) % val_interval ==0:
  #   validation()

# print(output.size())
cpu_out = output.cpu()
detached = cpu_out.detach().numpy()
plotLoaded(detached, 199)
# print(loss)

### Fragen:
