
""" Find the location of an emotion from a collection of images
This code is based on "Neural Transfer Using Pytorch"
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import json
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
import matplotlib.pyplot as plt

style_layers_default = ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5")
#style_layers_default = ("conv_1",)
content_layers_default = ("conv_4",)

def image_loader(image_name, loader, device=None):
    """Given a image file name, load it into a tensor"""
    if device is None:
        device = torch.get_default_device()
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(layer_input):
    """Compute the gram matrix"""
    a, b, c, d = layer_input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = layer_input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    g = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return g.div(a * b * c * d)


# pylint: disable-next=too-few-public-methods
class Normalization(nn.Module):
    """Module used to normalize inputs during forwarding."""
    def __init__(self, mean, std):
        super().__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, image):
        """normalize ``image``"""
        return (image - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, layer_input):
        self.loss = F.mse_loss(layer_input, self.target)
        return layer_input


# pylint: disable-next=too-few-public-methods
class LossMarker(nn.Module):
    """This class is used to intercept and compute the gram matrix
    after specific layers."""
    def __init__(self, name, gram):
        super().__init__()
        self.name = name
        self.gram = gram.to(torch.get_default_device())
        self.loss = None

    def forward(self, layer_input):
        """Save the gram matrix for this layer"""
        g = gram_matrix(layer_input)
        self.loss = F.mse_loss(g, self.gram)
        return layer_input


def build_model(cnn, normalization, style_grams,
                content_image,
                style_layers=style_layers_default,
                content_layers=content_layers_default):
    """Modify the VGG19 model to fit our needs:
       - replace the in-place ReLU layers with out-of-place
       - add identity layers to each of the layers we want to
         capture for style. The layers capture the activations
         of those layers
       - remove the rest of the model (we don't care about the
         final fully-connect ANN layer(s)
    Returns a model we can run our images through"""
    model = nn.Sequential(normalization).to(torch.get_default_device())

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{i}"
        else:
            raise RuntimeError(
                "Unrecognized layer: {layer.__class__.__name__}"
            )

        model.add_module(name, layer)

        if name in style_layers:
            layer_name = f"style_loss_{i}"
            if layer_name in style_grams:
                loss_marker = LossMarker(layer_name, style_grams[layer_name])
                model.add_module(layer_name, loss_marker)
                del style_grams[layer_name]

        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], LossMarker) or isinstance(model[i], ContentLoss):
            break

    # truncate the model after the last layer we care about
    model = model[: (i + 1)]
    return model.to(torch.get_default_device())


def get_input_optimizer(input_image):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_image])
    return optimizer


def get_style_losses(model):
    ret = []
    for layer in model.children():
        if isinstance(layer, LossMarker):
            ret.append(layer)
    assert len(ret) > 0, "no style loss layers?"
    return ret


def get_content_losses(model):
    ret = []
    for layer in model.children():
        if isinstance(layer, ContentLoss):
            ret.append(layer)
    assert len(ret) > 0, "no content loss layers?"
    return ret


def run_style_transfer(model, normalization_mean, normalization_std,
                       content_image, num_steps=300,
                       style_weight=1000000, content_weight=1):
    input_image = content_image.clone()
    input_image.requires_grad_(True)

    model.eval()
    model.requires_grad_(False)
    
    style_losses = get_style_losses(model)
    content_losses = get_content_losses(model)
    optimizer = get_input_optimizer(input_image)
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_image.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_image)
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

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_image.clamp_(0, 1)

    return input_image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
        plt.pause(0.001)

    
def transfer_emotion(image_name, emotion, style_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    imsize = 512
    loader = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    content_image = image_loader(image_name, loader, device=None).to(device)

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

    grams = torch.load(f"{emotion}.grams")
    model = build_model(cnn, normalization, grams, content_image,
                        style_layers=style_layers)

    image = run_style_transfer(model, cnn_normalization_mean,
                               cnn_normalization_std, content_image)
    return image

def main():
    parser = argparse.ArgumentParser(prog="emotion-xfer")
    parser.add_argument('input_image', help="name of input image file")
    parser.add_argument('emotion', help="emotion to transfer")
    parser.add_argument('output_image', help="output image filename")
    parser.add_argument('--style-layer', action='append', help="use style layer")
    args = parser.parse_args()

    if args.style_layer is None:
        args.style_layer = style_layers_default
    image = transfer_emotion(args.input_image, args.emotion, args.style_layer)

    image = image.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(args.output_image)

    
if __name__ == "__main__":
    main()
