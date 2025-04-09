
""" Find the location of an emotion from a collection of images
This code is based on "Neural Transfer Using Pytorch"
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import json
import os

import torch
from torch import nn

from PIL import Image

from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights


style_layers_default = ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5")


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

    def forward(self, img):
        """normalize ``img``"""
        return (img - self.mean) / self.std


# pylint: disable-next=too-few-public-methods
class LossMarker(nn.Module):
    """This class is used to intercept and compute the gram matrix
    after specific layers."""
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.gram = None

    def forward(self, layer_input):
        """Save the gram matrix for this layer"""
        self.gram = gram_matrix(layer_input)
        return layer_input


def build_model(cnn, normalization, style_layers=style_layers_default):
    """Modify the VGG19 model to fit our needs:
       - replace the in-place ReLU layers with out-of-place
       - add identity layers to each of the layers we want to
         capture for style. The layers capture the activations
         of those layers
       - remove the rest of the model (we don't care about the
         final fully-connect ANN layer(s)
    Returns a model we can run our images through"""
    model = nn.Sequential(normalization)

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
            loss_marker = LossMarker(layer_name)
            model.add_module(layer_name, loss_marker)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], LossMarker):
            break

    # truncate the model after the last layer we care about
    model = model[: (i + 1)]
    return model


# process_image(model, fname):
def process_image(model, fname, loader):
    """run the image stored in 'fname' through the model 'model' and
    return a list of tuples. Each tuple is layer name [0] and the
    corresponding gram matrix for that layer [1]."""
    img = image_loader(fname, loader)
    model(img)
    gram_matrices = []
    for layer in model.children():
        if isinstance(layer, LossMarker):
            gram_matrices.append((layer.name, layer.gram))
    return gram_matrices


def process_emotion(model, emotion, loader, image_list):
    """Go through all of the images relating to "emotion" and
    compute the 'center' of that emotion"""

    result = None
    with open("index.json", encoding='utf-8') as jfile:
        image_list = json.load(jfile)

    base_path = "512x512"
    image_list = [k for k, v in image_list.items() if v["emotion"] == emotion]
    assert len(image_list) > 0, f"empty image list for {emotion}"

    # process all of the images for a particular emotion. for each image
    # we have 5 or so gram matrices. The code below finds the centroid
    # (arithmetic mean) of each matrix.
    #
    # i.e. for each image, we get a gram matrix for each of its style layers.
    # we average the corresponding gram matrix layer for all of the images
    # for a particular emotion. This gives a gram matrix which represents
    # the centroid of that emotion.
    #
    # The idea is to use the resulting matrices instead of matrices computed
    # from a single style matrix in the neural transfer of style algorithm.
    #
    for num_images, image_name in enumerate(image_list):
        grams = process_image(model, os.path.join(base_path, f"{image_name}.jpg"), loader)

        if result is None:
            result = []
            for gram in grams:
                name = gram[0]
                result.append([name, torch.zeros(gram[1].size()).detach()])

        print(f"{emotion} {num_images+1} of {len(image_list)}")
        for i, gram in enumerate(grams):
            assert gram[0] == result[i][0]
            result[i][1] = torch.add(result[i][1], gram[1])

        # every N images, consolidate the sum matrices
        if num_images % 10 == 0:
            for i in range(len(grams)):
                result[i][1] = result[i][1].detach()

    # detach/consolidate the result matrices
    for i, _ in enumerate(grams):
        result[i][1] = result[i][1].div(len(image_list)).detach()

    return result

def emotion(emotion, image_list):
    """Initialize torch, build the model, the process an emotion's worth
    of images"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.FloatTensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.FloatTensor([0.229, 0.224, 0.225])

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

    model = build_model(cnn, normalization)

    # desired size of the output image
    # imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
    imsize = 512

    # scale imported image, transform it into a torch tensor
    loader = transforms.Compose(
        [transforms.Resize(imsize), transforms.ToTensor()]
    )

    res_list = process_emotion(model, emotion, loader, image_list)
    m = {}
    for name, tsr in res_list:
        m[name] = tsr.detach()
    torch.save(m, f"{emotion}.grams")

def main():
    with open("index.json", encoding='utf-8') as jfile:
        image_list = json.load(jfile)
        
    emotions = set()
    for _, d in image_list.items():
        emotions.add(d['emotion'])

    for d in emotions:
        emotion(d, image_list)

if __name__ == "__main__":
    main()
