import torch
from torchvision import datasets, transforms
from torch import nn
from torchvision import models
from PIL import Image


def predict(path):
    classes = 3
    model = models.efficientnet_b1(pretrained=False)
    n_inputs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, classes)
    )
    model_path = "styleRecognization/style.pt"
    model.load_state_dict(torch.load(model_path))
    image_size = 224
    img = Image.open(path)
    tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
    img = tfms(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(img)
    preds = torch.topk(logits, k=3).indices.squeeze(0).tolist()
    res = [0, 0, 0]
    maxprob = 0
    for idx in preds:
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        res[idx] = prob
        maxprob = max(prob, maxprob)
    if maxprob == res[0]:
        index = 0
    elif maxprob == res[1]:
        index = 1
    else:
        index = 2
    print(res, index)
    return res, index


if __name__ == '__main__':
    predict('../images/da_vinci.jpg')
    predict('../images/monet.jpg')
    predict('../images/picasso.jpg')
    predict('../images/van_gogh.jpg')
