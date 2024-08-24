import torch
import torchvision.models as models

# method 1 -------->
# save model with only weights
model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")

# load model
model = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()


# method 2 -------->
# saving entire model
torch.save(model, "model2.pth")

# load model
model = torch.load("model2.pth")
