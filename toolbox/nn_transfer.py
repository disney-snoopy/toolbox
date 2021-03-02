from toolbox.ntransfer import *
import numpy as np
from PIL import Image
from torch import nn
from torchvision import models
from torchvision import transforms as T


class nnTransfer(nn.Module):
  '''this class constructs a nn based on a given pretrained model
  with specified number of conv layers.
  During forward function, the model saves convolutional feature maps (content representation) automatically.'''
  def __init__(self, model, num_fmaps):
    super(nnTransfer, self).__init__()
    self.layers = list(model.children())
    self.num_fmaps = num_fmaps
    self.model = nn.Sequential()
    self.style_layers = range(num_fmaps)
    self.content_layers = range(num_fmaps)
  def model_construct(self):
    self.conv_list = []
    conv_counter = 0
    layer_counter = 0
    for layer in self.layers:
      layer_name =str(type(layer)).split('.')[-1][:-2]
      self.model.add_module(f'{layer_name} {layer_counter}', layer)
      if isinstance(layer, nn.Conv2d):
        conv_counter += 1
        self.conv_list.append(layer_counter)
      layer_counter += 1
      if conv_counter == self.num_fmaps:
        break
  def extract(self, input, fmap_type):
    # this is not the most computationally efficient code
    # feature maps are calculated only once, so keeping the code for now
    if fmap_type == 'content':
      self.content_img = unloader(input)
    elif fmap_type == 'style':
      self.style_img = unloader(input)

    if fmap_type == 'content':
      self.fmaps = []
    elif fmap_type == 'style':
      self.grams = []

    for conv in self.conv_list:
      model = self.model[:conv+2]
      fmap = model(input)
      if fmap_type == 'content':
        self.fmaps.append(fmap.detach())
      elif fmap_type == 'style':
        gram = gram_mat(fmap)
        self.grams.append(gram.detach())
  def model_split(self):
    self.model_segments = []
    for idx in self.conv_list:
      if self.conv_list[0] == idx:
        self.model_segments.append(self.model[:idx+2])
        b = idx+2
      else:
        self.model_segments.append(self.model[b:idx+2])
        b = idx+2
  def loss_calc(self, input, style_weight, content_weight):
    loss_style_sum = 0
    loss_content_sum = 0
    layer_counter = 0
    for model_seg, fmap, gram_map in zip(self.model_segments, self.fmaps, self.grams):
      if model_seg == self.model_segments[0]:
        output = model_seg(input)
        output_gram = gram_mat(output)
        if layer_counter in self.content_layers:
          loss_content = F.mse_loss(output, fmap)
          loss_content_sum += (loss_content * content_weight)
        if layer_counter in self.style_layers:
          loss_style = F.mse_loss(output_gram, gram_map)
          loss_style_sum += (loss_style * style_weight)
        to_next_layer = output
        layer_counter += 1
      else:
        output = model_seg(to_next_layer)
        output_gram = gram_mat(output)
        if layer_counter in self.content_layers:
          loss_content = F.mse_loss(output, fmap)
          loss_content_sum += (loss_content * content_weight)
        if layer_counter in self.style_layers:
          loss_style = F.mse_loss(output_gram, gram_map)
          loss_style_sum += (loss_style * style_weight)
        to_next_layer = output
        layer_counter += 1

    loss = loss_content_sum + loss_style_sum
    return loss

  def transfer(self, tensor_input, style_weight, content_weight, lr = None, epochs = 100, output_freq = 20):
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.lr = lr
    if lr == None:
      starting_loss = float(self.loss_calc(tensor_input, style_weight, content_weight).detach().cpu())
      self.lr = starting_loss/500
      print('Starting loss: ', starting_loss)
      print('lr: ',self.lr)

    tensor_input = tensor_input.to(device, torch.float).requires_grad_()
    opt = optim.Adam([tensor_input], lr = self.lr)
    self.output_imgs = []
    self.epoch_nums = []

    for epoch in range(epochs):
      loss = self.loss_calc(tensor_input, style_weight, content_weight)
      loss.backward()

      if epoch % output_freq == 0:
        print(f'Epoch: {epoch}, loss: {loss}')
      opt.step()
      opt.zero_grad()
      if epoch % output_freq == 0:
        self.output_imgs.append(tensor_input.clip(0, 1))
        self.epoch_nums.append(epoch)
    self.output_imgs.append(tensor_input.clip(0, 1))
    self.epoch_nums.append(epoch)
    print('Number of outputs: ', len(self.output_imgs))
  def plot_output(self, img_per_row = 3):
    num_outputs = len(self.output_imgs)
    num_rows = np.ceil(num_outputs/img_per_row)
    fig, axs = plt.subplots(1, 2, figsize = (16, 6), sharey=True, sharex=True)
    axs = axs.flatten()
    axs[0].imshow(self.content_img)
    axs[1].imshow(self.style_img)

    fig, axs = plt.subplots(nrows=int(num_rows), ncols=int(img_per_row), figsize = (16, 6 * img_per_row), sharex=True, sharey=True)
    axs = axs.flatten()
    img_counter = 0

    for ax in axs:
      ax.imshow(unloader(self.output_imgs[img_counter]))
      ax.set_title(f'Epoch: {self.epoch_nums[img_counter]+1}')
      img_counter += 1
      ax.margins(0.05)
      if img_counter == num_outputs:
        break

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #scientific notation
    sn_style_weight ='{:e}'.format(self.style_weight)
    axs[0].text(30, -60, f'Style weight: {sn_style_weight}\ncontent weight: {self.content_weight},\nlearning rate: {round(float(self.lr), 3)}', fontsize = 12)