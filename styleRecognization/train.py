from fastai.vision.all import *
from fastai.data.all import *
import torch
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import random_split
from torchvision import models
from PIL import Image


def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes


def get_data_loaders(data_dir, batch_size, train=False):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply(transforms=[
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            ], p=0.43),
            transforms.Resize(300),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data) * 0.75)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader, train_data_len

    else:
        transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data) * 0.70)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return (val_loader, test_loader, valid_data_len, test_data_len)


def train():
    dataset_path = "images/"
    dls = ImageDataLoaders.from_folder(dataset_path, valid_pct=0.25, item_tfms=Resize(224))
    classes = get_classes(dataset_path)
    model = models.efficientnet_b1(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_inputs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(classes))
    )
    model = model.to(device)
    criterion = CrossEntropyLossFlat()
    learn = Learner(dls, model, loss_func=criterion, metrics=[accuracy, error_rate])
    learn.fit(20)
    torch.save(model.state_dict(), "style.pt")