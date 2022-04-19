import torch as T
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import Sequential
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=True)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.stages = nn.Sequential(*[
            self._make_stage(3, 64, num_blocks=2, max_pooling=True),
            self._make_stage(64, 128, num_blocks=2, max_pooling=True),
            self._make_stage(128, 256, num_blocks=4, max_pooling=True),
            self._make_stage(256, 512, num_blocks=4, max_pooling=True),
            self._make_stage(512, 512, num_blocks=4, max_pooling=True)
        ])
        self.head = nn.Sequential(*[
            nn.Linear(512 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):
        layers = [Conv(in_channels, out_channels, kernel_size=3, stride=1)]
        for _ in range(1, num_blocks):
            layers.append(Conv(out_channels, out_channels, kernel_size=3, stride=1))
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stages(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


model = VGG19(2)
if torch.cuda.is_available():
    model = model.cuda()
a = T.rand(5, 3, 160, 160).cuda() if torch.cuda.is_available() else T.rand(5, 3, 160, 160)
print(model(a))

path = "./dataset/train"
test_path = "./dataset/val"
classes = ["BigNose", "PointyNose"]
batch_size = 32
epochs = 100
lr = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.CenterCrop(160),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

data_train = datasets.ImageFolder(root=path, transform=transform)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

data_test = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)


def get_variable(x):
    x = torch.autograd.Variable(x)
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def train():
    lossF = nn.MSELoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_pth = 999999999.99
    i_pth = 0
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0.0
        print("Epochs [{}/{}]".format(epoch, epochs))

        for X_train, y_train in train_loader:
            X_train, y_train = get_variable(X_train), get_variable(y_train)
            outputs = model(X_train)
            _, predict = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            predict = predict.double()
            y_train = y_train.double()
            loss = lossF(predict, y_train)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += torch.sum(predict == y_train.data)

        testing_correct = 0.0

        for data in test_loader:
            X_test, y_test = data
            X_test, y_test = get_variable(X_test), get_variable(y_test)
            outputs = model(X_test)
            _, predict = torch.max(outputs.data, 1)
            testing_correct += torch.sum(predict == y_test.data)

        print("Loss: {}    Training Accuracy: {}%    Testing Accuracy:{}%".format(
            running_loss,
            100 * running_correct / len(data_train),
            100 * testing_correct / len(data_test)
        ))

        if running_loss < loss_pth:
            loss_pth = running_loss
            torch.save(model, "./models/classify_%d.pth" % i_pth)
            i_pth = i_pth + 1

    torch.save(model, "vgg19.pth")
    print("训练完成！最小损失为：%f" % loss_pth)


def inference_model(test_img, net):
    for data in test_loader:
        test, _ = data
        img, _ = data
        test = get_variable(test)
        outputs = net(test)
        rate, predict = torch.max(outputs.data, 1)
        for i in range(len(data_test)):
            print("It may be %s." % classes[predict[i]])
            img0 = img[i]
            img0 = img0.numpy().transpose(1, 2, 0)
            plt.imshow(img0)
            plt.title("It may be %s." % classes[predict[i]])
            # plt.title("It may be %s, probability is %f." % (classes[predict[i]], rate[i]))
            plt.show()


def test():
    model_path = "./models/cell_classify.pth"
    net = torch.load(model_path)
    net.eval()
    inference_model(test_path, net)


train()