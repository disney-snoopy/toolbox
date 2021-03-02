from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.figure(figsize=(7,4))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(content_img, style_img, input_img, content_layers, style_layers,
                       cnn=None, normalization_mean=cnn_normalization_mean, normalization_std=cnn_normalization_std,
                       num_steps=300,style_weight=1000000, content_weight=1, output_freq = 50):
    output_imgs = []
    epoch_nums = []
    if cnn == None:
        cnn = models.vgg19(pretrained=True).features.to(device).eval()

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,normalization_mean,
                                                                    normalization_std,
                                                                    style_img, content_img, content_layers, style_layers)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:

                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            if run[0] % output_freq == 0:
                output_imgs.append(input_img.detach().data.clamp_(0,1))
                epoch_nums.append(run[0])

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    output_imgs.append(input_img.data.clamp_(0,1))
    epoch_nums.append(run[0])

    return output_imgs, epoch_nums

class lbfgs_Transfer():
    def __init__(self, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        self.content_layers = content_layers
        self.style_layers = style_layers

    def learn(self, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
        self.img_content = unloader(content_img[0])
        self.img_style = unloader(style_img[0])
        self.style_weight = style_weight
        self.content_weight = content_weight

        self.output_imgs, self.epoch_nums = run_style_transfer(content_img, style_img, input_img, self.content_layers, self.style_layers,
                                                                cnn=None, normalization_mean=cnn_normalization_mean, normalization_std=cnn_normalization_std,
                                                                num_steps=300,style_weight=1000000, content_weight=1, output_freq = 50)

    def plot_output(self, img_per_row = 3):
        num_outputs = len(self.output_imgs)
        num_rows = np.ceil(num_outputs/img_per_row)
        fig, axs = plt.subplots(1, 2, figsize = (16, 6), sharey=True, sharex=True)
        axs = axs.flatten()
        axs[0].imshow(self.img_content)
        axs[1].imshow(self.img_style)

        fig, axs = plt.subplots(nrows=int(num_rows), ncols=int(img_per_row), figsize = (16, 6 * img_per_row), sharex=True, sharey=True)
        axs = axs.flatten()
        img_counter = 0

        for ax in axs:
            ax.imshow(unloader(self.output_imgs[img_counter][0]))
            ax.set_title(f'Epoch: {self.epoch_nums[img_counter]+1}')
            img_counter += 1
            ax.margins(0.05)
            if img_counter == num_outputs:
                break

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        #scientific notation
        sn_style_weight ='{:e}'.format(self.style_weight)
        axs[0].text(30, -60, f'Style weight: {sn_style_weight}\ncontent weight: {self.content_weight}', fontsize = 12)








