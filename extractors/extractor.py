import cv2
from skimage import feature
import imutils
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision import models
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from PIL import Image, ImageOps


def get_scaler(df):
    df = df['features'].values.tolist()
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


class Features1065:
    def __init__(self, by, device):
        self._by = by
        self._device = device
        self._desc = LabHistogram([4, 4, 4])
        self._lbp = LocalBinaryPatterns(64, 8)
        self._hog = HOG(orientations=4, pixels_per_cell=(16, 16))
        self._vgg = NewVGG('features').eval().to(self._device)
        self._resnet = NewResnet('layer4').eval().to(self._device)

    _preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _get_single_image_features(self, image_file):
        result = []
        image = cv2.imread(image_file)
        shape = image.shape
        if len(shape) == 2:  # Если изображение в оттенках серого
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Преобразовать в RGB
        ch_res = self._desc.describe(image)
        lbp_res = self._lbp.describe(image)
        hog_res = self._hog.describe(image)
        image = Image.open(image_file)
        if image.mode == 'L':
            image = ImageOps.colorize(image, black="cyan", white="white")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        batch = self._preprocess(image).unsqueeze(0).to(self._device)
        vgg_res = self._vgg(batch).detach().cpu().numpy().flatten()
        resnet_res = self._resnet(batch).detach().cpu().numpy().flatten()
        result.extend(ch_res)
        result.extend(lbp_res)
        result.extend(hog_res)
        result.extend(vgg_res)
        result.extend(resnet_res)
        return result

class LabHistogram:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image, mask=None):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        else:
            hist = cv2.normalize(hist, hist).flatten()
        return hist


class LocalBinaryPatterns:
    def __init__(self, num_points=8, radius=1):
        self.numPoints = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


class HOG:
    def __init__(self, orientations=5, pixels_per_cell=(10, 10), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def describe(self, image):
        image_for_hog = cv2.resize(image, (100, 100))
        image_for_hog = cv2.cvtColor(image_for_hog, cv2.COLOR_BGR2GRAY)
        H = feature.hog(image_for_hog,
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        transform_sqrt=True,
                        block_norm="L1"
                        )
        return H


class NewVGG(nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()
        self.pretrained = models.vgg11(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x


class NewResnet(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = models.resnet18(pretrained=True)
        self.children_list = []
        for n, c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)

    def forward(self, x):
        x = self.net(x)
        return x
def _chi2_distance(a, b):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b + 10 ** (-5)) for (a, b) in zip(a, b)])
    return chi