from PIL import Image
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        # TODO: Implement forward_mapping
        im = TF.rotate(im, 90, interpolation=InterpolationMode.BILINEAR)
        return im

    def inverse_view(self, noise, background=None, **kwargs):
        # TODO: Implement inverse_mapping
        noise = TF.rotate(noise, -90, interpolation=InterpolationMode.BILINEAR)
        return noise
