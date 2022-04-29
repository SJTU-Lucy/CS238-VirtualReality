import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fmobilenet import FaceMobileNet
import math


class feature_loss():
    # 用content image计算原始图像的特征向量
    def __init__(self, input):
        embedding_size = 512
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = FaceMobileNet(embedding_size)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()
        self.input_shape = [1, 128, 128]
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize(self.input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.content_feature = self.featurize(input, self.transform, self.model, self.device)

    # input of [1, 3, height, width]
    # 对于迁移后的图像计算特征，并计算二者差异
    def compute(self, input):
        cur_feature = self.featurize(input, self.transform, self.model, self.device)
        x1 = self.content_feature.cpu().numpy().reshape(-1)
        x2 = cur_feature.cpu().numpy().reshape(-1)
        # cos = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        diff = x1 - x2
        res = np.linalg.norm(diff)
        return res

    # 灰度化，缩放尺寸
    def _preprocess(self, image, transform):
        res = []
        im = image[0].detach().numpy()
        im = (np.transpose(im, (1, 2, 0)) + 1) / 2.0 * 255.0
        im = Image.fromarray(np.uint8(im))
        im = transform(im)
        res.append(im)
        data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
        data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
        return data

    # 计算特征矩阵
    def featurize(self, input, transform, net, device):
        # input of [1, 3, H, W] -> [1, 1, H, W]
        data = self._preprocess(input, transform)
        data = data.to(device)
        net = net.to(device)
        with torch.no_grad():
            ret = net(data)
        return ret
