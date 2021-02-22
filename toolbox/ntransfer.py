import numpy as np
from PIL import Image
from torch import nn
from torchvision import models
from torchvision import transforms as T

def loader(path, imsize):
  preproc = T.Compose([
                     T.Resize((imsize, imsize)),            #Rescale to default size
                     T.ToTensor()])                         #transform image into torch tensor

  img = Image.open(path)
  img_proc = preproc(img).unsqueeze(0)           #torch conv layers requires 4 dimensional shape (bs x ch x h x w)
  return img_proc

def unloader(img):
  t_unload = T.ToPILImage()                                   #Transform back to PIL image

  img = img.squeeze(0)
  img = t_unload(img)
  return img

def model_construct(layer_count):
  vgg16 = models.vgg16(pretrained=True).features.eval()
  layers = list(vgg16.children())
  model = nn.Sequential()
  counter = 0
  for i in range(layer_count):
    layer_name = f'layer_{counter}'
    counter += 1
    model.add_module(layer_name, layers[i])
  return model

def white_noise(imsize):
  img_noise = np.abs(np.random.randn(imsize, imsize, 3))/10
  img_noise = T.ToTensor()(img_noise)
  img_noise.clip_(0, 1)
  img_noise = img_noise.unsqueeze(0)

  return img_noise