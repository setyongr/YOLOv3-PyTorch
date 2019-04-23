import torch

import cv2
import numpy as np
from torch.autograd import Variable

from net import Net
from util import prep_image


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


model = Net("cfg/yolov3.cfg")
inp = prep_image(cv2.imread("image/dog-cycle-car.png"), int(model.net_info['width']))
pred = model(inp)
print(pred)
