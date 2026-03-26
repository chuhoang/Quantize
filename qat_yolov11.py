################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import sys
import os
import yaml
import warnings
import argparse
import json
from pathlib import Path
from copy import deepcopy

# QAT modules
import sys
sys.path.insert(1, '.')
# import quantization.quantize as quantize
import quantization.quantize_11 as quantize

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Ultralytics YOLOv8 imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
# from ultralytics.utils.checks import check_imgsz
# from ultralytics.utils.files import increment_path
# from ultralytics.data.utils import check_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel

from copy import deepcopy
import modelopt.torch.opt as mto

# Disable warnings
warnings.filterwarnings("ignore")

ARGS = None  # Global variable to hold YOLOv8 args

from apex.contrib.sparsity import ASP
def prune_trained_model_custom(model, optimizer, allow_recompute_mask=False, allow_permutation=True,
                               compute_sparse_masks=True):
    """ Adds mask buffers to model (init_model_for_pruning), augments optimize, and computes masks if .
    Source: https://github.com/NVIDIA/apex/blob/52c512803ba0a629b58e1c1d1b190b4172218ecd/apex/contrib/sparsity/asp.py#L299
    Modifications:
      1) Abstracted 'allow_recompute_mask' and 'allow_permutation' arguments
      2) Enabled sparse mask computation as optional
    """
    asp = ASP()
    asp.init_model_for_pruning(
        model,
        mask_calculator="m4n2_1d",
        verbosity=2,
        whitelist=[torch.nn.Linear, torch.nn.Conv2d],
        allow_recompute_mask=allow_recompute_mask,
        allow_permutation=allow_permutation
    )
    asp.init_optimizer_for_pruning(optimizer)
    if compute_sparse_masks:
        asp.compute_sparse_masks()
    return asp

class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)

def load_yolov8_model(weight, device):
    try:        
        from ultralytics.nn.tasks import attempt_load_weights

        model = attempt_load_weights(
            weight, device=device, inplace=True, fuse=True
        )
        if hasattr(model, "kpt_shape"):
            kpt_shape = model.kpt_shape  # pose-only
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, "module") else model.names  # get class names
        fp16 = False
        model.half() if fp16 else model.float()
        ch = model.yaml.get("channels", 3)
        # self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        return model
        
    except Exception as e:
        LOGGER.error(f"❌ Failed to load YOLOv8 model: {e}")
        # Fallback to direct loading
        try:
            ckpt = torch.load(weight, map_location=device)
            model = ckpt['model'] if 'model' in ckpt else ckpt
            
            # Handle compatibility issues
            for m in model.modules():
                if type(m) is nn.Upsample:
                    m.recompute_scale_factor = None
                    
            model.float()
            model.eval()
            model = model.to(device)
            if hasattr(model, 'fuse'):
                model.fuse()
                
            return model
            
        except Exception as e2:
            LOGGER.error(f"❌ Failed to load model with fallback: {e2}")
            raise e2

def create_yolov8_dataloader(data_path, imgsz=640, batch_size=10, augment=False, 
                            hyp=None, rect=False, stride=32, workers=8, prefix="", num_classes=86):
    """Create YOLOv8 compatible dataloader"""

    try:
        # Create dataset
        data_dict = {
                'train': data_path,
                'val': data_path,
                'nc': num_classes,  # COCO classes
                'names': [f'class{i}' for i in range(num_classes)],
                'channels': 3  # RGB channels
            }
        
        
        dataset = YOLODataset(
            img_path=data_path,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache=False,
            single_cls=False,
            stride=stride,
            pad=0.0 if augment else 0.5,
            prefix=prefix,
            task='detect',
            classes=None,
            data=data_dict,
            fraction=1.0
        )
        
        # Create dataloader
        loader = build_dataloader(
            dataset=dataset,
            batch=batch_size,
            workers=workers,
            shuffle=augment,
            rank=-1
        )
        
        return loader
        
    except Exception as e:
        LOGGER.error(f"Failed to create dataloader: {e}")
        raise e

def get_default_yolov8_hyp():
    """Get default YOLOv8 hyperparameters"""
    return {
        # Optimizer hyperparameters
        'lr0': 0.01,  # initial learning rate
        'lrf': 0.01,  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay 5e-4
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # cls loss gain
        'dfl': 1.5,  # dfl loss gain
        'pose': 12.0,  # pose loss gain
        'kobj': 1.0,  # keypoint obj loss gain
        'label_smoothing': 0.0,  # label smoothing (fraction)
        'nbs': 64,  # nominal batch size
        'overlap_mask': True,  # masks should overlap during training
        'mask_ratio': 4,  # mask downsample ratio
        'dropout': 0.0,  # use dropout regularization
        
        # Augmentation hyperparameters
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0.1,  # image translation (+/- fraction)
        'scale': 0.5,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mosaic': 1.0,  # image mosaic (probability)
        'mixup': 0.0,  # image mixup (probability)
        'copy_paste': 0.0,  # segment copy-paste (probability)
        'fuse_socre': True
    }

def get_ultralytics_default_hyp():
    """Get hyperparameters directly from Ultralytics"""
    try:
        from ultralytics.utils import DEFAULT_CFG
        return DEFAULT_CFG
    except ImportError:
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.yaml')
            return model.overrides
        except:
            return get_default_yolov8_hyp()

def create_coco_train_dataloader_yolov8(cocodir, batch_size=10, num_classes=86):
    train_path = f"{cocodir}/train2017.txt"
    hyp = get_ultralytics_default_hyp()

    return create_yolov8_dataloader(
        train_path, imgsz=640, batch_size=batch_size, 
        augment=True, hyp=hyp, rect=False, stride=32,
        prefix=colorstr("train: "), num_classes=num_classes
    )

def create_coco_val_dataloader_yolov8(cocodir, batch_size=10, keep_images=None, num_classes=86):
    val_path = f"{cocodir}/val2017.txt"
    hyp = get_ultralytics_default_hyp()
    loader = create_yolov8_dataloader(
        val_path, imgsz=640, batch_size=batch_size,
        augment=False, hyp=hyp, rect=True, stride=32,
        prefix=colorstr("val: "), num_classes=num_classes
    )

    if keep_images is not None:
        original_len = loader.dataset.__len__
        def limited_len():
            return min(keep_images, original_len())
        loader.dataset.__len__ = limited_len
    
    return loader


def extract_images_from_batch(batch, device):
    if isinstance(batch, dict):
        possible_keys = ['img', 'image', 'images', 'input', 'data']
        for key in possible_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                imgs = batch[key]
                break
        else:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                    imgs = value
                    break
            else:
                raise ValueError(f"Could not find image tensor in batch keys: {list(batch.keys())}")
    elif isinstance(batch, (list, tuple)):
        imgs = batch[0]
    else:
        imgs = batch

    if not isinstance(imgs, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(imgs)}")
    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
        imgs = imgs.float() / 255.0
    
    return imgs


def evaluate_coco_yolov8(model, dataloader, using_cocotools=False, save_dir=".", 
                        conf_thres=0.001, iou_thres=0.65):
    """Evaluate model on COCO dataset using YOLOv8 validation"""
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG

    args = deepcopy(DEFAULT_CFG)
    args.data = "/home/mq/disk2T/tungdt/yolo_erd/scripts/coco_train_weapon.yaml"
    args.mode = "val"
    validator = DetectionValidator(
        dataloader=dataloader,
        save_dir=Path(save_dir),
        args=args
    )
    results = validator(model=model)
    map = results['metrics/mAP50-95(B)']
    
    return map
        

def export_onnx_yolov8(model, file, size=640, dynamic_batch=False, noanchor=False):
    """Export YOLOv8 model to ONNX format"""
    device = next(model.parameters()).device
    model.eval()

    dummy = torch.zeros(1, 3, size, size, device=device)
    for m in model.modules():
        if hasattr(m, 'export'):
            m.export = True
        if hasattr(m, 'format'):
            m.format = 'onnx'

    if noanchor:
        output_names = ["output0", "output1", "output2"]
        dynamic_axes = {
            "images": {0: "batch"}, 
            "output0": {0: "batch"}, 
            "output1": {0: "batch"}, 
            "output2": {0: "batch"}
        } if dynamic_batch else None
    else:
        output_names = ["output0"]
        dynamic_axes = {
            "images": {0: "batch"}, 
            "output0": {0: "batch"}
        } if dynamic_batch else None
    
    quantize.export_onnx(
        model, dummy, file, 
        opset_version=13,
        input_names=["images"], 
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Reset export flags
    for m in model.modules():
        if hasattr(m, 'export'):
            m.export = False


def cmd_quantize_yolov8(weight, cocodir, device, ignore_policy, save_ptq, save_qat, 
                       supervision_stride, iters, eval_origin, eval_ptq):
    # im = torch.zeros(1, 3, 640, 640).to(device)

    quantize.initialize()

    # Create directories
    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)
    
    device = torch.device(device)
    model = load_yolov8_model(weight, device)

    # Verify model type
    if not isinstance(model, DetectionModel):
        LOGGER.warning(f"Model type {type(model)} may not be fully compatible with YOLOv8 QAT")

    num_classes = 86
    train_dataloader = create_coco_train_dataloader_yolov8(cocodir=cocodir, batch_size=4, num_classes=num_classes)
    val_dataloader = create_coco_val_dataloader_yolov8(cocodir, num_classes)
    
    # Apply quantization
    quantize.replace_custom_module_forward(model, device)
    quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)    
    quantize.calibrate_model(model, train_dataloader, device)
    quantize.apply_custom_rules_to_quantizer(model, lambda model, file: export_onnx_yolov8(model, file))

    # Setup summary
    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)    

    # Evaluate original model
    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
            summary.append(["Origin", ap])

            print(f"Original mAP: {ap:.5f}")

    # Evaluate PTQ model
    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
        summary.append(["PTQ", ap])
        print(f"PTQ mAP: {ap:.5f}")

    # Save PTQ model
    if save_ptq:
        print(f"Save PTQ model to {save_ptq}")
        torch.save({"model": model}, f'{save_ptq}')

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    # QAT Fine-tuning
    best_ap = 0
    def per_epoch_callback(model, epoch, lr):
        nonlocal best_ap
        ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
        summary.append([f"QAT{epoch}", ap])
        print(f"Epoch {epoch}, mAP: {ap:.5f}")

        if ap > best_ap:
            print(f"Save QAT model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, f'{save_qat}')
        
        return False  # Continue training

    def preprocess(datas):
        try:
            return extract_images_from_batch(datas, device)
        except Exception as e:
            print(f"Error in preprocess: {e}")
            if isinstance(datas, dict):
                for key in ['img', 'image', 'images']:
                    if key in datas and isinstance(datas[key], torch.Tensor):
                        imgs = datas[key]
                        break
                else:
                    for key, value in datas.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                            imgs = value
                            break
                    else:
                        raise ValueError(f"Could not find image tensor in batch keys: {list(datas.keys())}")
            elif isinstance(datas, (list, tuple)):
                imgs = datas[0]
            else:
                imgs = datas

            if not isinstance(imgs, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(imgs)}")
            imgs = imgs.to(device, non_blocking=True)
            if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
                imgs = imgs.float() / 255.0
            
            return imgs

    def supervision_policy():
        """Create supervision policy for YOLOv8"""
        supervision_list = []
        
        # Get all modules
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ModuleList):
                supervision_list.append((name, id(module)))

        # Select modules based on stride
        keep_modules = []
        for i in range(0, len(supervision_list), supervision_stride):
            keep_modules.append(supervision_list[i][1])
        
        # Add final detection layers
        for name, module in model.named_modules():
            if 'detect' in name.lower() or 'head' in name.lower():
                keep_modules.append(id(module))

        def impl(name, module):
            module_id = id(module)
            should_supervise = module_id in keep_modules
            if should_supervise:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            return should_supervise
        
        return impl

    print("Starting QAT fine-tuning...")
    quantize.finetune(
        model, train_dataloader, per_epoch_callback, 
        early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess, 
        supervision_policy=supervision_policy(),
        nepochs=10,
        prefix=colorstr("YOLO11 ")
    )

def cmd_export_yolov8(weight, save, size, dynamic, noanchor, noqadd):
    """Export YOLOv8 quantized model to ONNX"""
    quantize.initialize()
    
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
        
    try:
        model = torch.load(weight, map_location="cpu")["model"]
    except:
        model = load_yolov8_model(weight, "cpu")
    
    if not noqadd:
        quantize.replace_custom_module_forward_yolov8(model)

    export_onnx_yolov8(model, save, size, dynamic_batch=dynamic, noanchor=noanchor)
    print(f"Save ONNX to {save}")


def cmd_sensitive_analysis_yolov8(weight, device, cocodir, summary_save, num_image):
    """Sensitive layer analysis for YOLOv8"""
    quantize.initialize()
    device = torch.device(device)
    model = load_yolov8_model(weight, device)
    model.float()
    
    train_dataloader = create_coco_train_dataloader_yolov8(cocodir)
    val_dataloader = create_coco_val_dataloader_yolov8(
        cocodir, keep_images=None if num_image is None or num_image < 1 else num_image
    )
    
    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader, device)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_coco_yolov8(model, val_dataloader)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    layer_names = []
    for name, module in model.named_modules():
        if quantize.have_quantizer(module):
            layer_names.append((name, module))

    for name, layer in layer_names:
        print(f"Quantization disable {name}")
        quantize.disable_quantization(layer).apply()
        ap = evaluate_coco_yolov8(model, val_dataloader)
        summary.append([ap, name])
        quantize.enable_quantization(layer).apply()
    
    summary_sorted = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary_sorted[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")

def cmd_test_yolov8(weight, device, cocodir, confidence, nmsthres):
    """Test YOLOv8 model"""
    device = torch.device(device)
    model = load_yolov8_model(weight, device)
    val_dataloader = create_coco_val_dataloader_yolov8(cocodir)
    
    ap = evaluate_coco_yolov8(
        model, val_dataloader, True, 
        conf_thres=confidence, iou_thres=nmsthres
    )
    print(f"Test mAP: {ap:.5f}")

def init_seeds(seed=0):
    """Initialize random seeds"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='qat_yolov8.py')
    subps = parser.add_subparsers(dest="cmd")
    
    # Export command
    exp = subps.add_parser("export", help="Export weight to ONNX file")
    # exp.add_argument("weight", type=str, default="yolov8n.pt", help="export pt file")
    # exp.add_argument("--save", type=str, required=False, help="export onnx file")
    # exp.add_argument("--size", type=int, default=640, help="export input size")
    # exp.add_argument("--dynamic", action="store_true", help="export dynamic batch")
    # exp.add_argument("--noanchor", action="store_true", help="export no anchor nodes")
    # exp.add_argument("--noqadd", action="store_true", help="export do not add QuantAdd")

    # Quantize command
    qat = subps.add_parser("quantize", help="PTQ/QAT finetune for YOLOv8")
    qat.add_argument("weight", type=str, nargs="?", default="yolov8n.pt", help="weight file")
    qat.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    qat.add_argument("--device", type=str, default="cuda:0", help="device")
    qat.add_argument("--ignore-policy", type=str, default=r"model\.22\.dfl\.(.*)", help="regex for layers to ignore")
    qat.add_argument("--ptq", type=str, default="ptq_yolov8.pt", help="PTQ model save file")
    qat.add_argument("--qat", type=str, default=None, help="QAT model save file")
    qat.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    qat.add_argument("--iters", type=int, default=200, help="iterations per epoch")
    qat.add_argument("--eval-origin", action="store_true", help="evaluate original model")
    qat.add_argument("--eval-ptq", action="store_true", help="evaluate PTQ model")

    # Sensitive analysis command
    sensitive = subps.add_parser("sensitive", help="Sensitive layer analysis")
    sensitive.add_argument("weight", type=str, nargs="?", default="yolov8n.pt", help="weight file")
    sensitive.add_argument("--device", type=str, default="cuda:0", help="device")
    sensitive.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    sensitive.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")
    sensitive.add_argument("--num-image", type=int, default=None, help="number of images to evaluate")

    # Test command
    testcmd = subps.add_parser("test", help="Test model evaluation")
    testcmd.add_argument("weight", type=str, default="yolov8n.pt", help="weight file")
    testcmd.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    testcmd.add_argument("--device", type=str, default="cuda:0", help="device")
    testcmd.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    testcmd.add_argument("--nmsthres", type=float, default=0.65, help="NMS threshold")

    args = parser.parse_args()
    init_seeds(57)

    if args.cmd == "export":
        # cmd_export_yolov8(args.weight, args.save, args.size, args.dynamic, args.noanchor, args.noqadd)
        ignore_policy = [
            r"model\.10\..*",          # Ignore all of Layer 10 (Attention Header)
            r".*\.attn\..*",           # Ignore other attention internals if needed
            r".*dfl.*",             # Ignore detection head (optional but recommended)
            r"model\.23*dfl\.(.*)"
        ]
        device = torch.device("cpu")
        model = load_yolov8_model("/home/mq/disk2T/tungdt/yolo_erd/best_1.pt", device)
        train_dataloader = create_coco_train_dataloader_yolov8(cocodir="/home/mq/disk2T/tungdt/data_weapon/", batch_size=4, num_classes=86)

        quantize.initialize()
        model = quantize.replace_custom_module_forward(model, device)
        quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)    
        # quantize.calibrate_model(model, train_dataloader, device)
        # quantize.apply_custom_rules_to_quantizer(model, export_onnx_yolov8)

        state_dict = torch.load("/home/mq/disk2T/tungdt/yolo_erd/qat_yolov11.pt", map_location=device)
        model.load_state_dict(state_dict["model_state_dict"], strict=False)

        export_onnx_yolov8(model, "/home/mq/disk2T/tungdt/yolo_erd/dummy.onnx")


    elif args.cmd == "quantize":
        args.ignore_policy = [
            r"model\.0\..*", 
            r"model\.10\..*",          # Ignore all of Layer 10 (Attention Header)
            r".*\.attn\..*",           # Ignore other attention internals if needed
            r".*dfl.*",             # Ignore detection head (optional but recommended)
            r"model\.23*dfl\.(.*)"
        ]
        print(args)
        cmd_quantize_yolov8(
            args.weight, args.cocodir, args.device, args.ignore_policy, 
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq
        )

    elif args.cmd == "sensitive":
        cmd_sensitive_analysis_yolov8(args.weight, args.device, args.cocodir, args.summary, args.num_image)
    elif args.cmd == "test":
        cmd_test_yolov8(args.weight, args.device, args.cocodir, args.confidence, args.nmsthres)
    else:
        parser.print_help()