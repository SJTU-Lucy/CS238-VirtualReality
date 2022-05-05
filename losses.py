import torch as T
import torch.nn.functional as F
from torch_cluster import knn_graph

import utils
from vgg import VGG19


# 继承VGG19，计算content和style的loss
class StyleTransferLosses(VGG19):
    def __init__(self, weight_file, content_img: T.Tensor, style_img: T.Tensor, content_layers, style_layers,
                 scale_by_y=False, content_weights=None, style_weights=None):
        super(StyleTransferLosses, self).__init__(weight_file)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.scale_by_y = scale_by_y

        content_weights = content_weights if content_weights is not None else [1.] * len(self.content_layers)
        style_weights = style_weights if style_weights is not None else [1.] * len(self.style_layers)
        self.content_weights = {}
        self.style_weights = {}

        content_features = content_img
        style_features = style_img
        self.content_features = {}
        self.style_features = {}
        if scale_by_y:
            self.weights = {}
        else:
            self.modified_features = {}
            self.eps = 0.0001
        # 产生特征图像
        i, j = 0, 0
        self.to(content_img.device)
        with T.no_grad():
            for name, layer in self.named_children():
                content_features = layer(content_features)
                style_features = layer(style_features)
                if name in style_layers:
                    self.style_features[name] = utils.gram_matrix(style_features)
                    self.style_weights[name] = style_weights[j]
                    j += 1
                if name in content_layers:
                    self.content_features[name] = content_features
                    if scale_by_y:
                        self.weights[name] = T.minimum(content_features, T.sigmoid(content_features))
                    else:
                        print(content_features.size())
                        print(style_features.size())
                        Gl = style_features / (content_features + self.eps)
                        Gl = T.clamp(Gl, min=0.7, max=5)
                        self.modified_features[name] = content_features * Gl
                    self.content_weights[name] = content_weights[i]
                    i += 1

    def forward(self, input):
        content_loss, style_loss = 0., 0.
        features = input
        for name, layer in self.named_children():
            features = layer(features)
            if name in self.content_layers:
                if self.scale_by_y:
                    loss = features - self.content_features[name]
                    loss *= self.weights[name]
                else:
                    loss = features - self.modified_features[name]
                content_loss += (T.mean(loss ** 2) * self.content_weights[name])
            if name in self.style_layers:
                loss = F.mse_loss(self.style_features[name], utils.gram_matrix(features), reduction='sum')
                style_loss += (loss * self.style_weights[name])

        return content_loss, style_loss


# 反映分布情况，值越大表明分布越不均匀
def total_variation_loss(location: T.Tensor, curve_s: T.Tensor, curve_e: T.Tensor, K=10):
    se_vec = curve_e - curve_s
    x_nn_idcs = knn_graph(location, k=K)[0]
    x_sig_nns = se_vec[x_nn_idcs].view(*((se_vec.shape[0], K) + se_vec.shape[1:]))
    dist_to_centroid = T.mean(T.sum((utils.projection(x_sig_nns) - utils.projection(se_vec)[..., None, :]) ** 2, dim=-1))
    return dist_to_centroid


# 反映弯曲程度，值越大则控制点偏离越大
def curvature_loss(curve_s: T.Tensor, curve_e: T.Tensor, curve_c: T.Tensor):
    v1 = curve_s - curve_c
    v2 = curve_e - curve_c
    dist_se = T.norm(curve_e - curve_s, dim=-1) + 1e-6
    return T.mean(T.norm(v1 + v2, dim=-1) / dist_se)


# 反映总体色彩分布
def tv_loss(x):
    diff_i = T.mean((x[..., :, 1:] - x[..., :, :-1]) ** 2)
    diff_j = T.mean((x[..., 1:, :] - x[..., :-1, :]) ** 2)
    loss = diff_i + diff_j
    return loss