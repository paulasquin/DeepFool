import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from PIL import ImageChops
from deepfool import deepfool
from deepfool import convert_label
import os

PATH_IMAGE = '161749_2s-panneau_stop.jpg'
# PATH_IMAGE = '161850-2m53-person.jpg'
NUMBER_DEEP_FOOL = 1


def show_image(image, title=""):
    plt.figure()
    if title != "":
        plt.title(title)
    plt.imshow(image)
    plt.show()


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A


def main():
    # Switch to evaluation mode
    im_orig = Image.open(PATH_IMAGE)
    show_image(im_orig, "image origin")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    clip = lambda x: clip_tensor(x, 0, 255)
    # Remove the mean
    trans_im = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)])

    trans_pil = transforms.ToPILImage()

    trans_im_2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                             transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                             transforms.Lambda(clip),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(224)])

    tf_2 = transforms.Compose([transforms.ToPILImage(),
                               transforms.CenterCrop(224)])

    # im = trans_im(im_orig)
    im = trans_im(im_orig)
    show_image(trans_pil(im), "image trans_im")

    current_image = im
    les_str_labels_orig = []
    for iteration in range(1, NUMBER_DEEP_FOOL + 1):
        net = models.resnet34(pretrained=True)
        net.eval()
        print("Fooling " + str(iteration) + "/" + str(NUMBER_DEEP_FOOL))
        r, loop_i, label_orig_temp, label_pert, pert_image = \
            deepfool(current_image, net, num_classes=10, only_cpu=True)
        #  Converting labels in real world words
        str_label_orig_temp = convert_label(label_orig_temp)
        str_label_pert = convert_label(label_pert)
        #  Saving labels for modification history
        les_str_labels_orig.append(str_label_orig_temp)
        #  Retrieving image
        pert_image_raw = pert_image.cpu()[0]
        pert_image_show = tf(pert_image_raw)
        show_image(pert_image_show, str_label_orig_temp + "->" + str_label_pert)
        #  Preparing image for new run
        current_image = trans_im(pert_image_show)
        show_image(trans_pil(current_image), "new image analysis")

        if iteration < NUMBER_DEEP_FOOL:
            # If not last iteration
            pass
        else:
            str_label_orig = les_str_labels_orig[0]

    print("Original label = ", str_label_orig)
    print("Perturbed label = ", str_label_pert)

    name_label_pert = '_'.join(str_label_pert.split(" ")[1:])
    name_label_orig = '_'.join(str_label_orig.split(" ")[1:])
    path_pert_image = PATH_IMAGE.replace(".jpg", "-" + name_label_orig + "-" + name_label_pert + ".jpg")
    path_pert_only_image = path_pert_image.replace(".jpg", "_pert-only.jpg")

    print("Saving pert image to " + path_pert_image)
    # pert_image_raw = pert_image.cpu()[0]
    # pert_image = trans_pil(pert_image_raw)
    pert_image_to_save = pert_image_show
    pert_image_to_save.save(path_pert_image)

    with open(PATH_IMAGE.replace(".jpg", ".txt"), 'w') as file:
        file.write(" -> ".join(les_str_labels_orig + [str_label_pert]))

    print("Saving pert only image to " + path_pert_only_image)
    # orig_image = trans_pil(im_orig)
    pert_only_image = ImageChops.subtract(tf_2(trans_im_2(im_orig)), pert_image_show)
    pert_only_image.save(path_pert_only_image)


if __name__ == "__main__":
    main()
