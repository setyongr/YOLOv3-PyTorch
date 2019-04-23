from __future__ import division

import torch.nn as nn
import torch.nn.functional as F


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def build_conv(
        index,
        batch_normalize,
        input_filter,
        out_filters,
        kernel_size,
        pad,
        stride,
        bias,
        activation
):
    module = nn.Sequential()
    conv = nn.Conv2d(input_filter, out_filters, kernel_size, stride, pad, bias=bias)
    module.add_module("conv_{}".format(index), conv)
    if batch_normalize:
        bn = nn.BatchNorm2d(out_filters)
        module.add_module("batch_norm_{}".format(index), bn)

    if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace=True)
        module.add_module("leaky_{0}".format(index), activn)

    return module


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x['type'] == 'convolutional':

            activation = x['activation']
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            module = build_conv(
                index=index,
                batch_normalize=batch_normalize,
                input_filter=prev_filters,
                out_filters=filters,
                kernel_size=kernel_size,
                pad=pad,
                stride=stride,
                bias=bias,
                activation=activation
            )

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # Positive anotation
            if start > 0:
                print("Start", start)
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        elif x["type"] == "maxpool":
            size = int(x['size'])
            stride = int(x['stride'])

            if stride != 1:
                max_pool = nn.MaxPool2d(kernel_size=size, stride=stride)
            else:
                max_pool = MaxPoolStride1(size)

            module.add_module("maxpool{}".format(index), max_pool)
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list
