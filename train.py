from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import pickle
import shutil
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn

def save_checkpoint(state, is_best, filename = 'checkpoints/yolov3.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/yolov3_best.pth.tar')


def main():
    tr_losses = []
    mAPs = []

    def train():
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad = False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % args.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.zero_grad()
                optimizer.step()
                    
            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss {}".format(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += "\n---- ETA {}".format(time_left)

            print(log_str)

            model.seen += imgs.size(0)

        return loss.item()

    def validate():
        print("\n---- Evaluating Model ----")
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path = valid_path,
            iou_thres = 0.5,
            conf_thres = 0.5,
            nms_thres = 0.5,
            img_size = args.img_size,
            batch_size = 8,
        )
        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print("---- mAP {}".format(AP.mean()))

        curr_mAP = AP.mean()
        is_best = curr_mAP > max_mAP
        max_mAP = max(max_mAP, curr_mAP)
        return curr_mAP, is_best, max_mAP

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok = True)
    os.makedirs("checkpoints", exist_ok = True)

    is_best = 0
    curr_mAP = 0
    max_mAP = 0


    # Parse data configuration 
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    
    # Get dataloader
    dataset = ListDataset(train_path, augment = True, multiscale = args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size = args.batch_size,
                shuffle = True,
                num_workers = args.workers,
                pin_memory = True,
                collate_fn = dataset.collate_fn,
    )

    # Initialize model and optimizer
    model = Darknet(args.model_def).to(device)
    model.apply(weights_init_normal)
    # model = torch.nn.DataParallel(model, device_ids = range(len(args.gpu))).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # Resume the training 
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading YOLOv3 checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    
    if args.eval:
        validate()
        return

    # If specified we start from checkpoint
    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            model.load_darknet_weights(args.pretrained_weights)


    metrics = ["grid_size", "loss", "x", "y", "w", "h", 
               "conf", "cls", "cls_acc", "recall50", 
               "recall75", "precision", "conf_obj", "conf_noobj"]
    
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        tr_loss = train()
        tr_losses.append(tr_loss)
        if epoch % args.evaluation_interval == 0:
            curr_mAP, is_best, max_mAP = validate()
            mAPs.append(curr_mAP)

        if not epoch or not epoch % args.checkpoint_interval:
            save_checkpoint({
            "epoch": epoch + 1, # start from next epoch
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
            }, is_best)
    
    data = {"train": tr_losses, "val": mAPs}
    with open("output/yolo_results.pkl", "wb") as f: 
        pickle.dump(data, f)
    print("--- Training results saved to 'output/yolo_results.pkl' ----")


if __name__ == "__main__":

    # define namespace as the scope for variable use
    parser = argparse.ArgumentParser(description = 'Implement YOLOv3 Object Detection Task')
    parser.add_argument('--gpu', type = str, default = '0', 
                        help = 'Indices of GPU to use (If multiple, use form "0, 1, 2")')
    parser.add_argument("-e", "--epochs", type = int, default = 100, 
                        help = "Number of epochs, default = 100")
    parser.add_argument('-se', '--start-epoch', type = int, default = 0,  
                        help = 'Starting Epoch, defualt = 0')
    parser.add_argument("-b", "--batch-size", type = int, default = 8, 
                        help = "Mini-batch size, default = 8")
    parser.add_argument("-ga", "--gradient-accumulations", type = int, default = 2, 
                        help = "Number of gradient accumulations before step, default = 2")
    parser.add_argument("--lr", type = float, default = 9e-5, 
                        help = "learning rate, default = 9e-5")
    parser.add_argument("-md", "--model-def", type = str, default = "config/yolov3.cfg", 
                        help = "File path for storing model definition / configuration")
    parser.add_argument("-dd", "--data-config", type = str, default = "config/coco.data", 
                        help = "File path for storing data configuration")
    parser.add_argument("-w", "--pretrained-weights", type = str, default = "",
                        help = "Path for pretrained weights (only used for checkpoint loading mode)")
    parser.add_argument("-j", "--workers", type = int, default = 8, 
                        help = "Number of cpu threads / workers to use during batch generation, default = 8")
    parser.add_argument("-s", "--img-size", type = int, default = 416, 
                        help = "Image dimension, default = 416")
    parser.add_argument("-ci", "--checkpoint-interval", type = int, default = 1, 
                        help = "Interval between saving model weights, default = 1")
    parser.add_argument("-ei", "--evaluation-interval", type = int, default = 1, 
                        help = "Interval between evaluations on validation set, default = 1")
    parser.add_argument("-cm", "--compute-map", type = bool, default = False, 
                        help = "Get mAP if True for every 10 batches, default = False")
    parser.add_argument("-mt", "--multiscale-training", type = bool, default = True, 
                        help = "Whether to implement multi-scale training, default = True")
    parser.add_argument('--resume', type = str, default = '', 
                        help = 'Path to latest checkpoint (default: none)')
    parser.add_argument('--eval', default = False, type = bool, 
                    help = 'Turn on evaluation mode, default = False')
    
    global args
    args = parser.parse_args()
    print(args)

    main()

