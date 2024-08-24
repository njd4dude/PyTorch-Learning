import torch
import torchvision.models as models

model = models.vgg16(weights="IMAGENET1K_V1")

model.eval()