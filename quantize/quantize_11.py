import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.optim as optim
from torch.cuda import amp

from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from absl import logging as quant_logging

try:
    from quantization.rules_v2 import find_quantizer_pairs
except ImportError:
    def find_quantizer_pairs(onnx_file):
        return []


class QuantAdd(torch.nn.Module):
    """Quantized Add operation for yolo11"""
    def __init__(self, quantization):
        super().__init__()

        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input0_quantizer._calibrator._torch_hist = True
            self._input1_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y


class QuantConcat(torch.nn.Module):
    """Quantized Concat operation for yolo11"""
    def __init__(self, quantization=True):
        super().__init__()
        self.quantization = quantization
        if quantization:
            self._input_quantizers = torch.nn.ModuleList()
    
    def add_input_quantizer(self):
        if self.quantization:
            quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            quantizer._calibrator._torch_hist = True
            self._input_quantizers.append(quantizer)
            return quantizer
        return None
    
    def forward(self, inputs, dim=1):
        if self.quantization and len(self._input_quantizers) > 0:
            quantized_inputs = []
            for i, inp in enumerate(inputs):
                if i < len(self._input_quantizers):
                    quantized_inputs.append(self._input_quantizers[i](inp))
                else:
                    quantized_inputs.append(inp)
            return torch.cat(quantized_inputs, dim=dim)
        return torch.cat(inputs, dim=dim)


class QuantChunk(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c
        
    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)


class QuantUpsample(torch.nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, quantization=True):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.quantization = quantization
        
        # Add yolo11/v9 compatibility attributes
        self.f = -1  # Default flow index
        self.i = -1  # Default layer index
        self.type = 'QuantUpsample'  # Layer type identifier
        
        if quantization:
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input_quantizer._calibrator._torch_hist = True
    
    def forward(self, x):
        if self.quantization:
            x = self._input_quantizer(x)
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                                             mode=self.mode, align_corners=self.align_corners)


class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, submodule in module.named_modules():
        if isinstance(submodule, quant_nn.TensorQuantizer):
            return True
    return False


def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)


def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy: Union[str, List[str], Callable], path: str) -> bool:
    if ignore_policy is None: 
        return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):
        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: 
            return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


def bottleneck_quant_forward_yolo11(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def c3k2_quant_forward(self, x):
    if hasattr(self, "chunkop"):
        y = list(self.chunkop(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    else:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def replace_bottleneck_forward_yolo11(model):
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if hasattr(bottleneck, 'add') and bottleneck.add:
                if not hasattr(bottleneck, "addop"):
                    print(f"Add QuantAdd to {name}")
                    bottleneck.addop = QuantAdd(True)
                bottleneck.__class__.forward = bottleneck_quant_forward_yolo11


def replace_c3k2_forward_yolo11(model):
    for name, c3k2 in model.named_modules():
        if c3k2.__class__.__name__ == "C3k2":
            if not hasattr(c3k2, "chunkop"):
                print(f"Add QuantChunk to {name}")
                c3k2.chunkop = QuantChunk(c3k2.c)
            c3k2.__class__.forward = c3k2_quant_forward


def replace_custom_module_forward_yolo11(model, device=None):
    replace_bottleneck_forward_yolo11(model)
    replace_c3k2_forward_yolo11(model)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Upsample):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                quant_upsample = QuantUpsample(
                                    size=module.size,
                                    scale_factor=module.scale_factor, 
                                    mode=module.mode,
                                    align_corners=module.align_corners,
                                    quantization=True
                                )
                quant_upsample.i = module.i
                quant_upsample.f = module.f
                
                setattr(parent, child_name, quant_upsample)
                print(f"Replace Upsample with QuantUpsample at {name}")


def replace_to_quantization_module(model: torch.nn.Module, ignore_policy: Union[str, List[str], Callable] = None):
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    print(f"Quantization: {path} has ignored.")
                    continue
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
    recursive_and_replace_module(model)


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)

        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def export_onnx(model, input, file, *args, **kwargs):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)
    quant_nn.TensorQuantizer.use_fb_fake_quant = False


def apply_custom_rules_to_quantizer_yolo11(model: torch.nn.Module, export_onnx: Callable):
    try:
        export_onnx(model, "quantization-custom-rules-temp.onnx")
        pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
        for major, sub in pairs:
            print(f"ONNX Rules: {sub} match to {major}")
            try:
                get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
            except:
                print(f"Warning: Could not apply rule {sub} -> {major}")

        if os.path.exists("quantization-custom-rules-temp.onnx"):
            os.remove("quantization-custom-rules-temp.onnx")

    except Exception as e:
        print(f"Warning: Could not apply ONNX-based rules: {e}")

    for name, module in model.named_modules():
        if module.__class__.__name__ == "C3k2":
            module.chunkop._input0_quantizer = module.cv1.conv._input_quantizer
        
        if module.__class__.__name__ == "C3k":
            major = module.cv3.conv._input_quantizer
            module.cv1.conv._input_quantizer = major
            module.cv2.conv._input_quantizer = major

        if module.__class__.__name__ == "Bottleneck":
            major = module.cv1.conv._input_quantizer
            module.cv2.conv._input_quantizer = major
            module.addop._input0_quantizer = major
            module.addop._input1_quantizer = major

        if isinstance(module, torch.nn.MaxPool2d):
            quant_conv_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')
            quant_maxpool2d = quant_nn.QuantMaxPool2d(
                                        module.kernel_size,
                                        module.stride,
                                        module.padding,
                                        module.dilation,
                                        module.ceil_mode,
                                        quant_desc_input=quant_conv_desc_input)
            set_module(model, name, quant_maxpool2d)


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


def calibrate_model(model: torch.nn.Module, dataloader, device, num_batch=25):
    def compute_amax(model, device, **kwargs):  # Add device parameter
        successful_calibrations = 0
        failed_calibrations = 0
        
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    elif isinstance(module._calibrator, calib.HistogramCalibrator):
                        module.load_calib_amax(strict=False, **kwargs)
                    else:
                        try:
                            module.load_calib_amax(strict=False)
                        except TypeError:
                            module.load_calib_amax(strict=False, **kwargs)

                    if module._amax is not None:
                        module._amax = module._amax.to(device)
                        successful_calibrations += 1
                    else:
                        failed_calibrations += 1
    
        print(f"Calibration results: {successful_calibrations} successful, {failed_calibrations} failed")
    

    def collect_stats(model, data_loader, device, num_batch=200):
        model.eval()
        active_quantizers = []
        
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                    active_quantizers.append(name)
                else:
                    module.disable()

        print(f"Found {len(active_quantizers)} active quantizers: {active_quantizers[:5]}{'...' if len(active_quantizers) > 5 else ''}")

        processed_batches = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                try:
                    imgs = extract_images_from_batch(batch, device)
                    if imgs.dtype == torch.uint8:
                        imgs = imgs.float() / 255.0
                    elif imgs.max() > 1.0:
                        imgs = imgs / 255.0

                    _ = model(imgs)
                    processed_batches += 1
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue

                if i >= num_batch: break

        print(f"Successfully processed {processed_batches}/{min(num_batch, len(data_loader))} batches")

        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, device, method="mse")


def finetune(
    model: torch.nn.Module, train_dataloader, per_epoch_callback: Callable = None, preprocess: Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule: Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy: Callable = None, prefix=""
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()
    model.train()
    for param in model.parameters():
        param.requires_grad = True
        
    origin_model.train()
    for param in origin_model.parameters():
        param.requires_grad = False

    scaler = amp.GradScaler(enabled=fp16)
    optimizer = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-6
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        l.append(item)
                        break
                else:
                    l.append(output[0] if len(output) > 0 else output)
            else:
                l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue
        if supervision_policy:
            if not supervision_policy(mname, ml): continue
        supervision_module_pairs.append([ml, ori])

    for iepoch in range(nepochs):
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        origin_model.train()
        for param in origin_model.parameters():
            param.requires_grad = False
            
        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs = []
        origin_outputs = []
        remove_handle = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc=f"{prefix}QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):
            if ibatch >= early_exit_batchs_per_epoch: break
            if preprocess:
                imgs = preprocess(imgs)
            else:
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(device, non_blocking=True).float() / 255.0
                
            with amp.autocast(enabled=fp16):
                model(imgs)
                with torch.no_grad():
                    origin_model(imgs)
                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    # Ensure both are tensors before computing loss
                    if isinstance(mo, torch.Tensor) and isinstance(fo, torch.Tensor):
                        if mo.shape == fo.shape:
                            quant_loss += quant_lossfn(mo, fo)
                        else:
                            print(f"Warning: Shape mismatch at index {index}: {mo.shape} vs {fo.shape}")
                    else:
                        print(f"Warning: Non-tensor outputs at index {index}: {type(mo)} vs {type(fo)}")
                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"{prefix}QAT Finetuning {iepoch + 1}/{nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        for rm in remove_handle:
            rm.remove()
        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break

replace_bottleneck_forward = replace_bottleneck_forward_yolo11
apply_custom_rules_to_quantizer = apply_custom_rules_to_quantizer_yolo11
replace_custom_module_forward = replace_custom_module_forward_yolo11