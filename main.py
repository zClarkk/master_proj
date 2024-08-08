import monai
from monai.networks import nets
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import math
from torchinfo import summary
from monai.data import DataLoader

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
circ = torch.zeros([247, 166, 2, 200]) # patients, landmarks, axis, possible points
lm_count = 0 # needs to be 0 for below else 165 for test
patient_nr = 0

### Create sets
train_dataset = xRayDataset('train')
test_dataset = xRayDataset('test')
img, lms = train_dataset[patient_nr]

def circleCoordByFormula(image, landmarks, lm_limit, patient=patient_nr, width=255, height=255, radius=10, epsilon=5, lm=0, multiple=True):
    ''' Checks for each combination of y and x coordinates if the described point lies in the circle
        around the landmark. If yes, the point is added to the circ tensor.
        Optionally does this recursively for each landmark if multiple=True.
    '''
    ### Inits
    global lm_count
    count = 0
    x_axis = 0
    y_axis = 1
    print(lm_count)
    ### Calculation loop using the circle inlier formula (x-a)^2 + (y-b)^2 - r^2 < epsilon^2
    for y in range(height):
        for x in range(width):
            if abs((x-landmarks[lm][x_axis])**2 + (y-landmarks[lm][y_axis])**2 - radius**2) < epsilon**2:
                circ[patient][lm][x_axis][count] = x
                circ[patient][lm][y_axis][count] = y
                count += 1
    ### Recursive call for multiple landmarks
    if multiple == True and lm_count < lm_limit: 
        lm_count += 1
        circleCoordByFormula(image, landmarks, lm_limit, lm=lm+1)

def plotIt(image, landmarks, circles=circ, patient=patient_nr, lm=0, s_lm=5, s_circ=1, c_lm="r", c_circ="b"):
    ''' Plots the X-ray with the landmarks and circles ontop of it.
        Offers the option to plot either a single circle (lm>-1) or all (lm=-1).
        Parameters s_ and c_ are for the dot size and color.
    '''
    plot_flag = True
    plot_count = 0
    ### Case for plotting a single circle
    if lm > -1:
        implot = plt.imshow(image[:, :], cmap='gray')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=s_lm, c=c_lm)
        plt.scatter(circles[patient][lm][0][:], circles[patient][lm][1][:],  s=s_circ, c=c_circ)  
    ### Case for plotting all circles
    elif lm == -1:
        implot = plt.imshow(image[:, :], cmap='gray')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=s_lm, c=c_lm)
        ### Plot all calculated circles
        while plot_flag and plot_flag < 166:
            if circles[patient][plot_count][0][0] == 0 and circles[patient][plot_count][0][0] == 0:
                plot_flag = False
            plt.scatter(circles[patient][plot_count][0][:], circles[patient][plot_count][1][:],  s=s_circ, c=c_circ)
            plot_count += 1

### Function calls         
# circleCoordByFormula(img, lms, 165)
# circ = circ[...,0:155] # drops all non-zero entries by assuming that all circles have 155 pixels
# torch.save(circ, "circle.pt")
loaded_circles = torch.load("/home/erguen/Documents/monai-env/circle.pt")
plotIt(img, lms, circles= loaded_circles, lm=-1)


# ### Monai Unet
# model = nets.UNet(
#   spatial_dims=2,
#   in_channels=1,
#   out_channels=1,
#   channels=(64, 128, 256, 512),
#   strides=(2, 2, 2),
#   kernel_size=3,
#   up_kernel_size=3,
#   num_res_units=2,
#   act='LeakyReLU'
# ).to(device)

# ### Loss and Optimizer
# def TRE_loss(lms, lms_hat): # TRE_loss(landmarks, output, ord=2, dim=-1)
#   torch.linalg.vector_norm(lms - lms_hat, ord=2, dim=-1)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ### Initializations
# epoch_loss_values = []
# max_epochs = 10

# ### Training
# for epoch in range(max_epochs):
#   model.train()
#   epoch_loss = 0
#   step = 0
#   for batch_image, batch_landmarks in train_dataset:
#     step += 1
#     image, landmarks = batch_image.to(device), batch_landmarks.to(device)
#     image = image.unsqueeze(0)
#     image = image.unsqueeze(0)
#     optimizer.zero_grad()
#     output = model(image)
#     im = output.cpu().squeeze()
#     im = im.detach().numpy()
#     implot = plt.imshow(im[:,:], cmap='gray')
#     exit()
#     kill
#     loss = loss_function(output, landmarks)
#     loss.backward()
#     optimizer.step()
#     epoch_loss += loss.item()
#   epoch_loss /= step
#   epoch_loss_values.append(epoch_loss)

### Fragen:
