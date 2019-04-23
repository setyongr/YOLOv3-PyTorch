import time

import torch

import cv2
from torch.autograd import Variable

from net import Net
from util import load_classes, prep_image, write_results

cfgfile = "cfg/yolov3.cfg"
weightsfile = "weights/yolov3.weights"
classes = load_classes("classes/coco.names")
confidence = 0.5
num_classes = 80
nms_thesh = 0.4

print("Loading network.....")
model = Net(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")

model.eval()

# Capture image
cap = cv2.VideoCapture(0)
inp_dim = int(model.net_info["height"])


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, 0, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, 0, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

def object_count(output, frame):
    class_count = {}
    for x in output:
        cls = classes[int(x[-1])]

        if class_count.get(cls):
            class_count[cls] += 1
        else:
            class_count[cls] = 1

    b = 0
    for k, v in class_count.items():
        label = f'{k} = {v}'
        b += cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0][1]
        cv2.putText(frame, label, (0, b), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)

frames = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        with torch.no_grad():
            output = model(Variable(img, volatile=True))

        # batch_size, center_x, center_y, h, w, conf
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        if type(output) == int:
            frames += 1
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue


        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        list(map(lambda x: write(x, frame), output))

        object_count(output, frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frames += 1
        print("FPS = {:5.2f}".format(frames / (time.time() - start_time)))
    else:
        print("break")
        break
