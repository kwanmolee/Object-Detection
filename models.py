from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, move_onto_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, alpha = 1, gamma = 0.5, reduction = "mean"):
        super(FocalLoss, self).__init__()
        # apply focal loss to each element
        loss_fcn.reduction = "none"  
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        print("Use focal loss, gamma:", gamma)

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1+1e-6 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == "none":
            return loss
        return loss.mean() if self.reduction == "mean" else loss.sum()   

def build_modules(config, fl_gamma):
    """
    Build modules based on the pre-defined configuration
    @config: model configuration, type: list[dict]
    @fl_gamma: gamma of focal loss, type: float
    @return: 
            -params, the hyperparameters of model, type: dict
            -module_list, the list of modules, type: nn.ModuleList[nn.Sequential]
    """
    module_list = nn.ModuleList()
    params = config.pop(0)
    output_filters = [int(params["channels"])]
    routes = []
    
    
    for idx, cfg in enumerate(config):
        t = cfg["type"]
        modules = nn.Sequential()

        if t == "yolo":
            # Extract anchors
            anchors = [int(x) for x in cfg["anchors"].split(",")]
            anchors = [(i, j) for i, j in zip(anchors[:-1:2], anchors[1::2])]

            anchor_idxs = list(map(int, cfg["mask"].split(",")))
            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = int(cfg["classes"])
            img_size = int(params["height"])

            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size, fl_gamma)
            modules.add_module("{}_{}".format(t, idx), yolo_layer)

        elif t == "convolutional":
            bn = int(cfg["batch_normalize"])
            filters = int(cfg["filters"])
            kernel_size = int(cfg["size"])
            stride = int(cfg["stride"])
            padding = (kernel_size - 1) // 2
            
            modules.add_module(
                "conv_{}".format(idx),
                nn.Conv2d(
                    in_channels = output_filters[-1],
                    out_channels = filters,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    bias = 1 ^ bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_{}".format(idx), nn.BatchNorm2d(filters, momentum = 0.9, eps = 1e-5))
            if cfg["activation"] == "leaky":
                modules.add_module("leaky_{}".format(idx), nn.LeakyReLU(0.1))

        elif t == "dconvolutional":
            filters = int(cfg["filters"])
            kernel_size = int(cfg["size"])
            padding = (kernel_size - 1) // 2 if int(cfg["pad"]) else 0
            bn = int(cfg["batch_normalize"]) if "batch_normalize" in cfg else 0
            if bn:
                modules.add_module("batch_norm_{}".format(idx), nn.BatchNorm2d(filters, momentum = 0.9))
            if cfg["activation"] == "leaky":
                modules.add_module("leaky_{}".format(idx), nn.LeakyReLU(0.1))

        elif t == "route":
            layers = list(map(int, cfg["layers"].split(",")))
            filters = sum([output_filters[1: ][layer] for layer in layers])
            modules.add_module("{}_{}".format(t, idx), EmptyLayer())

        elif t == "shortcut":
            filters = output_filters[1: ][int(cfg["from"])]
            modules.add_module("{}_{}".format(t, idx), EmptyLayer())

        elif t == "maxpool":
            kernel_size = int(cfg["size"])
            stride = int(cfg["stride"])
            padding = (kernel_size - 1) // 2

            if kernel_size == 2 and stride == 1:
                modules.add_module("_debug_padding_{}".format(idx), nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = padding)
            modules.add_module("{}_{}".format(t, idx), maxpool)

        elif t == "upsample":
            upsample = Upsample(scale_factor = int(cfg["stride"]), mode = "nearest")
            modules.add_module("{}_{}".format(t, idx), upsample)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return params, module_list  

class EmptyLayer(nn.Module):
    """
    Build empty layer as placeholder for 'route' and 'shortcut' layers
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    """
    Upsample the input to either the given attr `scale_factor`
    """
    def __init__(self, scale_factor, mode = "nearest"):
        """
        @scale_factor: multiplier for spatial size, type: int
        @mode: algorithm used for upsampling, 
               type: (str) 'nearest' | 'linear' | 'bilinear' | 'trilinear'
        """
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """
        implement interpolation / upsampling
        @x: input tensor, type: torch.tensor
        """
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = self.mode)
        return x


class YOLOLayer(nn.Module):
    """
    Build YOLOv3 object detection layer
    """
    def __init__(self, anchors, num_classes, img_size = 416, fl_gamma = 0):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.MSE = nn.MSELoss()
        if fl_gamma == 0:
            self.FL = nn.BCELoss()
        else:
            self.FL = FocalLoss(nn.BCELoss(), gamma = fl_gamma) 

        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        
        self.grid_size = 0  # grid size

        self.FloatTensor = torch.FloatTensor
        self.BoolTensor = torch.BoolTensor
        self.LongTensor = torch.LongTensor

    def update_grids(self, grid_size, cuda = True):
        g = self.grid_size = grid_size
        self.stride = self.img_size / g

        #FloatTensor = self.FloatTensor.cuda() if cuda else self.FloatTensor
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        # (grid_x, grid_y) - vertical / horizontal offsets
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets = None, img_size = None):

        # move onto cuda if possible 
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if x.is_cuda else torch.BoolTensor

        self.img_size = img_size
        num_samples, grid_size = x.size(0), x.size(2)

        pred = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        ## (x, y) - center coordinates 
        ## (w, h) - width, height
        x = torch.sigmoid(pred[..., 0])  
        y = torch.sigmoid(pred[..., 1])  
        w = pred[..., 2]  
        h = pred[..., 3]  

        # prediction confidence
        pred_conf = torch.sigmoid(pred[..., 4])  
        # predicted classes
        pred_cls = torch.sigmoid(pred[..., 5:])  

        # update the grids by computing offsets if grid sizes do not match
        if grid_size != self.grid_size:
            self.update_grids(grid_size, cuda = x.is_cuda)

        # scale the anchors by adding offset 
        pred_boxes = FloatTensor(pred[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is not None:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes = pred_boxes,
                pred_cls = pred_cls,
                target = targets,
                anchors = self.scaled_anchors,
                ignore_thres = self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.MSE(x[obj_mask], tx[obj_mask])
            loss_y = self.MSE(y[obj_mask], ty[obj_mask])
            loss_w = self.MSE(w[obj_mask], tw[obj_mask])
            loss_h = self.MSE(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.FL(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.FL(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.FL(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss,
                "x": loss_x,
                "y": loss_y,
                "w": loss_w,
                "h": loss_h,
                "conf": loss_conf,
                "cls": loss_cls,
                "cls_acc": cls_acc,
                "recall50": recall50,
                "recall75": recall75,
                "precision": precision,
                "conf_obj": conf_obj,
                "conf_noobj": conf_noobj,
                "grid_size": grid_size,
            }
            self.metrics = move_onto_cpu(self.metrics)

            return output, total_loss

        return output, 0


class Darknet(nn.Module):
    """
    YOLOv3 object detection model
    """
    def __init__(self, config_path, img_size = 416, fl_gamma = 2):
        super(Darknet, self).__init__()
        self.config = parse_model_config(config_path)
        self.params, self.module_list = build_modules(self.config, fl_gamma)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets = None):
        img_size = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (cfg, module) in enumerate(zip(self.config, self.module_list)):
            if cfg["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif cfg["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in cfg["layers"].split(",")], 1)
            elif cfg["type"] == "shortcut":
                layer_i = int(cfg["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif cfg["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_size)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """
        Parses and loads the weights stored in 'weights_path'
        """

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (cfg, module) in enumerate(zip(self.config, self.module_list)):
            if i == cutoff:
                break
            if cfg["type"] == "convolutional":
                conv_layer = module[0]
                if cfg["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff = -1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (cfg, module) in enumerate(zip(self.config[:cutoff], self.module_list[:cutoff])):
            if cfg["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if cfg["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()