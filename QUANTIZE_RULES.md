# QUANTIZE_RULES.md

This document provides a comprehensive reference for all quantization rules applied in the YOLOv11 Quantization-Aware Training (QAT) implementation.

## Table of Contents
1. [Initialization Rules](#initialization-rules)
2. [Module Replacement Rules](#module-replacement-rules)
3. [Custom Quantized Operations](#custom-quantized-operations)
4. [Ignore Policies](#ignore-policies)
5. [Calibration Rules](#calibration-rules)
6. [Custom Module-Specific Rules](#custom-module-specific-rules)
7. [ONNX-Based Quantizer Pairing Rules](#onnx-based-quantizer-pairing-rules)
8. [Fine-Tuning Rules](#fine-tuning-rules)

---

## Initialization Rules

### Global Quantization Descriptor Setup
**Location**: `quantize_11.py:148-154`

```python
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)
```

**Rules**:
- All quantized layers use **histogram calibration** method by default
- Applies to: `QuantConv2d`, `QuantMaxPool2d`, `QuantAvgPool2d`, `QuantLinear`
- Logging level set to `ERROR` to reduce verbosity
- **Must be called before any quantization operations**

---

## Module Replacement Rules

### Standard PyTorch to Quantized Module Transfer
**Location**: `quantize_11.py:256-275`

```python
def replace_to_quantization_module(model, ignore_policy=None):
    # Replaces standard PyTorch modules with quantized versions
    # based on pytorch_quantization._DEFAULT_QUANT_MAP
```

**Rules**:
1. **Conv2d** → **QuantConv2d**
2. **Linear** → **QuantLinear**
3. **MaxPool2d** → **QuantMaxPool2d** (conditional, see custom rules)
4. **AvgPool2d** → **QuantAvgPool2d**

**Ignore Policy Application**:
- Regex patterns in `ignore_policy` prevent specific layers from being quantized
- Matching layers print: `"Quantization: {path} has ignored."`
- Pattern matching supports:
  - String literals (exact match)
  - Regex patterns (via `re.match()`)
  - Callable functions (custom logic)

**Example**:
```python
ignore_policy = [
    r"model\.10\..*",           # All of Layer 10
    r".*\.attn\..*",            # All attention layers
    r".*dfl.*",                 # Distribution Focal Loss layers
]
```

---

## Custom Quantized Operations

### 1. QuantAdd
**Location**: `quantize_11.py:26-42`

**Purpose**: Quantizes element-wise addition operations (residual connections)

**Architecture**:
```
Input x ─→ TensorQuantizer (8-bit, histogram) ─┐
                                                 ├─→ Add ─→ Output
Input y ─→ TensorQuantizer (8-bit, histogram) ─┘
```

**Rules**:
- Two independent input quantizers (`_input0_quantizer`, `_input1_quantizer`)
- 8-bit quantization (`num_bits=8`)
- Histogram calibration method
- `_torch_hist = True` for PyTorch-native histogram computation
- Only active when `quantization=True`

**Usage**: Applied to `Bottleneck` residual connections

---

### 2. QuantConcat
**Location**: `quantize_11.py:45-70`

**Purpose**: Quantizes concatenation operations along specified dimension

**Architecture**:
```
Input 0 ─→ TensorQuantizer 0 ─┐
Input 1 ─→ TensorQuantizer 1 ─┼─→ Concat ─→ Output
Input n ─→ TensorQuantizer n ─┘
```

**Rules**:
- Dynamic number of input quantizers (added via `add_input_quantizer()`)
- Each input has independent 8-bit quantizer with histogram calibration
- Concatenates along specified `dim` (default: `dim=1`)
- If quantizers not initialized, falls back to standard `torch.cat()`
- `_torch_hist = True` for all quantizers

**Usage**: Applied to feature concatenation operations in YOLO architecture

---

### 3. QuantChunk
**Location**: `quantize_11.py:73-80`

**Purpose**: Quantizes tensor splitting/chunking operations

**Architecture**:
```
Input ─→ TensorQuantizer ─→ torch.split() ─→ (chunk0, chunk1)
```

**Rules**:
- Single input quantizer (`_input0_quantizer`)
- Default `QuantDescriptor()` (uses global settings)
- Splits tensor into two chunks of size `c` along specified dimension
- Returns tuple of quantized chunks

**Usage**: Applied to `C3k2` module's `torch.split()` operations

---

### 4. QuantUpsample
**Location**: `quantize_11.py:83-105`

**Purpose**: Quantizes upsampling/interpolation operations

**Architecture**:
```
Input ─→ TensorQuantizer (8-bit, histogram) ─→ interpolate() ─→ Output
```

**Rules**:
- Single input quantizer with 8-bit histogram calibration
- Preserves original `Upsample` parameters:
  - `size`, `scale_factor`, `mode`, `align_corners`
- YOLOv11 compatibility attributes:
  - `f = -1` (flow index)
  - `i = -1` (layer index)
  - `type = 'QuantUpsample'`
- `_torch_hist = True` for histogram computation
- Falls back to `torch.nn.functional.interpolate()` when `quantization=False`

**Usage**: Replaces all `torch.nn.Upsample` layers in the model

---

## Ignore Policies

### Default Ignore Policy
**Location**: `qat_yolov11.py:650-655` and `qat_yolov11.py:627-632`

```python
ignore_policy = [
    r"model\.10\..*",          # Ignore all of Layer 10 (Attention Header)
    r".*\.attn\..*",           # Ignore attention internals
    r".*dfl.*",                # Ignore detection head DFL layers
    r"model\.23*dfl\.(.*)"     # Ignore specific DFL components
]
```

**Rules**:
1. **Layer 10** (Attention Header): Completely excluded from quantization
2. **Attention modules** (`.attn.*`): All attention internal operations stay FP32
3. **DFL layers** (`.*dfl.*`): Distribution Focal Loss layers stay FP32
4. **Detection head DFL** (`model.23*dfl.*`): Specific detection layer DFL components stay FP32

**Rationale**:
- Attention mechanisms are sensitive to quantization precision
- DFL requires precise distribution modeling
- Detection head outputs benefit from FP32 precision

### Policy Matching Logic
**Location**: `quantize_11.py:178-193`

```python
def quantization_ignore_match(ignore_policy, path):
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
```

**Supports**:
1. **String literal**: Exact path match (e.g., `"model.10.conv"`)
2. **Regex pattern**: Pattern matching (e.g., `r"model\.10\..*"`)
3. **List of patterns**: Multiple rules combined
4. **Callable function**: Custom logic `(path) -> bool`

---

## Calibration Rules

### Calibration Process
**Location**: `quantize_11.py:379-450`

**Two-Stage Process**:

#### Stage 1: Collect Statistics
```python
def collect_stats(model, data_loader, device, num_batch=200):
    # 1. Enable calibration mode for all quantizers
    # 2. Disable quantization (collect stats only)
    # 3. Run forward passes on calibration data
    # 4. Re-enable quantization and disable calibration
```

**Rules**:
- Default: **25 batches** for calibration (`num_batch=25`)
- All `TensorQuantizer` modules with valid `_calibrator` enabled
- Quantizers without calibrators are disabled entirely
- Input preprocessing:
  - Convert `uint8` to `float` and normalize: `imgs.float() / 255.0`
  - Normalize if `max > 1.0`: `imgs / 255.0`
- Progress tracked via `tqdm` progress bar
- Batch errors are logged but don't stop calibration

#### Stage 2: Compute AMAX
```python
def compute_amax(model, device, **kwargs):
    # Computes activation maximum (amax) from collected statistics
    # Uses MSE method for histogram calibration
```

**Rules**:
- **Histogram calibrators**: Use `method="mse"` for optimal quantization threshold
- **Max calibrators**: Direct load without method argument
- All `_amax` values transferred to target `device`
- Reports: `"{successful_calibrations} successful, {failed_calibrations} failed"`

### Histogram Calibration Settings
**Rules**:
- **Method**: `calib_method="histogram"`
- **PyTorch histogram mode**: `_calibrator._torch_hist = True`
  - Uses native PyTorch histogram implementation
  - Better compatibility with PyTorch operations
- **AMAX computation**: MSE (Mean Squared Error) method for optimal threshold

---

## Custom Module-Specific Rules

### Forward Pass Replacement Rules
**Location**: `quantize_11.py:196-254`

#### 1. Bottleneck Module
**Location**: `quantize_11.py:196-220`

**Original Forward**:
```python
x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

**Quantized Forward**:
```python
self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
```

**Rules**:
- Only applies when `bottleneck.add == True` (residual connection exists)
- Injects `QuantAdd` operation for residual addition
- Prints: `"Add QuantAdd to {name}"`
- Forward method dynamically replaced: `bottleneck.__class__.forward = bottleneck_quant_forward_yolo11`

#### 2. C3k2 Module
**Location**: `quantize_11.py:202-229`

**Original Forward**:
```python
y = list(self.cv1(x).split((self.c, self.c), 1))
y.extend(m(y[-1]) for m in self.m)
return self.cv2(torch.cat(y, 1))
```

**Quantized Forward**:
```python
y = list(self.chunkop(self.cv1(x), 2, 1))
y.extend(m(y[-1]) for m in self.m)
return self.cv2(torch.cat(y, 1))
```

**Rules**:
- Injects `QuantChunk` operation for tensor splitting
- `QuantChunk` initialized with `c3k2.c` (channel split size)
- Prints: `"Add QuantChunk to {name}"`
- Forward method dynamically replaced: `c3k2.__class__.forward = c3k2_qaunt_forward`

#### 3. Upsample Module
**Location**: `quantize_11.py:236-253`

**Rules**:
- **Complete replacement**: `torch.nn.Upsample` → `QuantUpsample`
- Preserves all original attributes:
  - `size`, `scale_factor`, `mode`, `align_corners`
  - `i` (layer index), `f` (flow index)
- Replaces module in parent module's attribute dictionary
- Prints: `"Replace Upsample with QuantUpsample at {name}"`

---

### Advanced Quantizer Sharing Rules
**Location**: `quantize_11.py:307-348`

These rules **share quantizers** between related operations to ensure consistent quantization scales.

#### Rule 1: C3k2 Input Sharing
```python
module.chunkop._input0_quantizer = module.cv1.conv._input_quantizer
```

**Logic**:
- `QuantChunk` operation shares quantizer with `cv1.conv`
- Ensures consistent quantization before splitting

**Simplified Diagram**:
```
Input ─→ cv1.conv (Quantizer A) ─→ QuantChunk (uses Quantizer A) ─→ Split
```

**Detailed C3k2 Block Architecture**:
```
                                    ┌─────────────────────────────────────────┐
                                    │          C3k2 Block (Quantized)         │
                                    └─────────────────────────────────────────┘

Input (c1 channels)
      │
      ├─────► cv1: Conv(c1, 2*c, 1×1) ───► [Conv+BN+Act] ──► Output: 2*c channels
      │                  │                                         │
      │                  │ (Quantizer A attached to cv1.conv)      │
      │                  │                                         │
      │                  └────► QuantChunk ◄────[shares Quantizer A]
      │                            │
      │                            ├─► Split into 2 chunks (each c channels)
      │                            │
      │                            ├─► Chunk[0] (c channels) ────┐
      │                            │                             │
      │                            └─► Chunk[1] (c channels) ────┼─► Bottleneck[0] ─┐
      │                                                          │                  │
      │                                                          │    cv1 ─► cv2    │
      │                                                          │   (Quantized)    │
      │                                                          │                  │
      │                                                          ├─► Bottleneck[1] ─┤
      │                                                          │                  │
      │                                                          ├─► ... (n total) ─┤
      │                                                          │                  │
      │                                                          └──────────────────┘
      │                                                                    │
      └──────────────────► Concat: [Chunk[0], Bottleneck[0], ..., Bottleneck[n-1]]
                                    │
                                    │ (2+n)*c channels total
                                    │
                                    └─► cv2: Conv((2+n)*c, c2, 1×1) ─► Output (c2 channels)
                                              │
                                              └─► [Conv+BN+Act]

Parameters:
  - c1: Input channels
  - c2: Output channels
  - c = int(c2 * e): Hidden channels (default e=0.5)
  - n: Number of Bottleneck blocks
  - Each Bottleneck processes c channels → c channels
```

#### Rule 2: C3k Quantizer Unification
```python
major = module.cv3.conv._input_quantizer
module.cv1.conv._input_quantizer = major
module.cv2.conv._input_quantizer = major
```

**Logic**:
- `cv3` is the "major" quantizer
- `cv1` and `cv2` share the same quantizer
- Ensures all three paths use identical quantization

**Simplified Diagram**:
```
         ┌─→ cv1 (Quantizer A)
Input ───┼─→ cv2 (Quantizer A)  [All share same quantizer]
         └─→ cv3 (Quantizer A - major)
```

**Detailed C3k Block Architecture**:
```
                          ┌──────────────────────────────────────────┐
                          │         C3k Block (Quantized)            │
                          └──────────────────────────────────────────┘

Input (c1 channels)
      │
      ├────────────────────► cv1: Conv(c1, c_, 1×1) ───► [Conv+BN+Act] ──┐
      │                            │                                     │
      │                [shares Quantizer A - major]                      │
      │                                                                  │
      └────────────────────► cv2: Conv(c1, c_, 1×1) ───► [Conv+BN+Act]   │
                                   │                        │            │
                       [shares Quantizer A - major]         │            │    
                                                            │            │
                                                            │       Bottleneck[0] (c_ → c_)
                                                            │            │
                                                            │       (cv1 ─► cv2 (k×k))
                                                            │            │
                                                            |       Bottleneck[1] (c_ → c_)
                                                            │            │
                                                            |       ... (n total)
                                                            │            │
                                                            |       Bottleneck[n-1]
                                                            |            │
                                                            |            └────┐
                                                            |                 │
                                                      cv2 output (c_) ────────┤
                                                                              │
                                                            Concat (2*c_ channels)
                                                                   │
                                                                   └─► cv3: Conv(2*c_, c2, 1×1)
                                                                          │
                                                              [Quantizer A - major attached here]
                                                                          │
                                                                  Output (c2 channels)

Parameters:
  - c1: Input channels
  - c2: Output channels
  - c_ = int(c2 * e): Hidden channels (default e=0.5)
  - n: Number of Bottleneck blocks
  - k: Kernel size for bottleneck convolutions (default k=3)

Quantization Strategy:
  - cv3.conv holds the MAJOR quantizer (Quantizer A)
  - cv1.conv and cv2.conv SHARE this major quantizer
  - All three paths quantized with identical scale/zero-point
```

#### Rule 3: Bottleneck Quantizer Unification
```python
major = module.cv1.conv._input_quantizer
module.cv2.conv._input_quantizer = major
module.addop._input0_quantizer = major
module.addop._input1_quantizer = major
```

**Logic**:
- `cv1` is the "major" quantizer
- `cv2` shares quantizer with `cv1`
- Both inputs to `QuantAdd` share the same quantizer
- Ensures residual addition has matched quantization scales

**Simplified Diagram**:
```
Input ─→ cv1 (Quantizer A - major) ─→ cv2 (Quantizer A) ─┐
  |                                                      ├─→ QuantAdd (both inputs use A)
  └──────────────────────────────────────────────────────┘
```

**Detailed Bottleneck Block Architecture**:

**Case 1: Standard Bottleneck (WITHOUT shortcut or c1≠c2)**
```
                    ┌───────────────────────────────────────┐
                    │   Bottleneck Block (No Residual)      │
                    └───────────────────────────────────────┘

        Input Tensor (FP32, c1 channels)
                │
                │
        ┌───────▼────────────────────────────────────┐
        │  cv1: Conv(c1, c_, k[0], stride=1)         │
        │                                            │
        │  ┌─────────────────────────────────────┐   │
        │  │ Conv2d (c1 → c_)                    │   │
        │  │   └─ INPUT_QUANTIZER_A ──► Quant    │   │
        │  │   └─ WEIGHT_QUANTIZER_A ──► Quant   │   │
        │  ├─────────────────────────────────────┤   │
        │  │ BatchNorm2d(c_) [FP32]              │   │
        │  ├─────────────────────────────────────┤   │
        │  │ Activation (SiLU) [FP32]            │   │
        │  └─────────────────────────────────────┘   │
        └────────────────┬───────────────────────────┘
                         │ (FP32, c_ channels)
                         │
        ┌────────────────▼───────────────────────────┐
        │  cv2: Conv(c_, c2, k[1], stride=1, g=g)    │
        │                                            │
        │  ┌─────────────────────────────────────┐   │
        │  │ Conv2d (c_ → c2)                    │   │
        │  │   └─ INPUT_QUANTIZER_B ──► Quant    │   │
        │  │   └─ WEIGHT_QUANTIZER_B ──► Quant   │   │
        │  ├─────────────────────────────────────┤   │
        │  │ BatchNorm2d(c2) [FP32]              │   │
        │  ├─────────────────────────────────────┤   │
        │  │ Activation (SiLU) [FP32]            │   │
        │  └─────────────────────────────────────┘   │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
                Output Tensor (FP32, c2 channels)

Notes:
  - No QuantAdd operation
  - No quantizer sharing (each Conv has independent quantizers)
  - Simple sequential flow: Input → cv1 → cv2 → Output
```

**Case 2: Residual Bottleneck (WITH shortcut, when c1==c2 and shortcut=True)**
```
                    ┌──────────────────────────────────────────┐
                    │  Bottleneck Block (With Residual)        │
                    │     *** QUANTIZER SHARING ACTIVE ***     │
                    └──────────────────────────────────────────┘

        Input Tensor (FP32, c1 channels)
                │
                │
                ├─────────────────────────────────────┐
                │                                     │
                │ [SHORTCUT PATH]                     │ [MAIN PATH]
                │  (Identity)                         │
                │                                     │
                │                     ┌───────────────▼────────────────────────┐
                │                     │  cv1: Conv(c1, c_, k[0], stride=1)     │
                │                     │                                        │
                │                     │  ┌──────────────────────────────────┐  │
                │                     │  │ Conv2d (c1 → c_)                 │  │
                │                     │  │   └─ INPUT_QUANTIZER_A (MAJOR)   │  │
                │                     │  │   └─ WEIGHT_QUANTIZER_A          │  │
                │                     │  ├──────────────────────────────────┤  │
                │                     │  │ BatchNorm2d(c_) [FP32]           │  │
                │                     │  ├──────────────────────────────────┤  │
                │                     │  │ Activation (SiLU) [FP32]         │  │
                │                     │  └──────────────────────────────────┘  │ 
                │                     └───────────────┬────────────────────────┘
                │                                     │ (FP32, c_ channels)
                │                                     │
                │                     ┌───────────────▼────────────────────────┐
                │                     │  cv2: Conv(c_, c2, k[1], stride=1)     │
                │                     │                                        │
                │                     │  ┌──────────────────────────────────┐  │
                │                     │  │ Conv2d (c_ → c2)                 │  │
                │                     │  │   └─ shares INPUT_QUANTIZER_A ◄──┼──┼─┐
                │                     │  │   └─ WEIGHT_QUANTIZER_cv2        │  │ │
                │                     │  ├──────────────────────────────────┤  │ │
                │                     │  │ BatchNorm2d(c2) [FP32]           │  │ │
                │                     │  ├──────────────────────────────────┤  │ │
                │                     │  │ Activation (SiLU) [FP32]         │  │ │
                │                     │  └──────────────────────────────────┘  │ │
                │                     └───────────────┬────────────────────────┘ │
                │                                     │ (FP32, c2=c1 channels)   │
                │                                     │                          │
                │                     ┌───────────────▼────────────────────────┐ │
                │                     │                                        │ │
                │                     │  QuantAdd (Element-wise Addition)      │ │
                │                     │                                        │ │
                │                     │  ┌──────────────────────────────────┐  │ │
                │                     │  │ input0 (from shortcut):          │  │ │
                └─────────────────────┼──┤   └─ _input0_quantizer ◄─────────┼──┘ │
                                      │  │       shares QUANTIZER_A         │  │ │
                                      │  ├──────────────────────────────────┤  │ │
                                      │  │ input1 (from cv2):               │  │ │
                                      │  │   └─ _input1_quantizer ◄─────────┼──│─┘
                                      │  │       shares QUANTIZER_A         │  │
                                      │  ├──────────────────────────────────┤  │
                                      │  │ Operation: Quantized INT8 Add    │  │
                                      │  │   INT8(input0) + INT8(input1)    │  │
                                      │  │   (same scale, same zero-point)  │  │
                                      │  └──────────────────────────────────┘  │
                                      └───────────────┬────────────────────────┘ 
                                                      │
                                                      ▼
                                        Output Tensor (FP32, c2 channels)

Quantizer Sharing Visualization:
  ┌──────────────────────────────────────────────────────────┐
  │         QUANTIZER_A (Master/Major)                       │
  │         Located at: cv1.conv._input_quantizer            │
  │                                                          │
  │  Shared with:                                            │
  │    1. cv2.conv._input_quantizer        ◄─── SHARES       │
  │    2. addop._input0_quantizer          ◄─── SHARES       │
  │    3. addop._input1_quantizer          ◄─── SHARES       │
  │                                                          │
  │  Result: All 4 locations use IDENTICAL quantization:     │
  │    - Same scale factor                                   │
  │    - Same zero-point                                     │
  │    - Same calibration statistics                         │
  └──────────────────────────────────────────────────────────┘

Why This Matters:
  ✓ INT8 addition requires both operands to have identical quantization
  ✓ Without sharing, addition would require re-quantization (expensive)
  ✓ Sharing ensures: Q(x) + Q(y) is valid in INT8 domain
  ✓ Mismatched quantizers would cause accuracy loss

Parameters:
  - c1: Input channels (must equal c2 for shortcut to activate)
  - c2: Output channels (must equal c1 for shortcut to activate)
  - c_ = int(c2 * e): Hidden/bottleneck channels (default e=0.5)
  - k: Tuple of kernel sizes [k[0] for cv1, k[1] for cv2], default (3,3)
  - g: Groups for cv2 grouped convolution (default g=1)
  - shortcut: Boolean flag to enable residual (AND c1==c2 must be true)
  - self.add = shortcut AND (c1 == c2)  ← Controls if QuantAdd is used

Quantization Strategy Summary:
  Step 1: cv1.conv gets INPUT_QUANTIZER_A (the MAJOR quantizer)
  Step 2: cv2.conv._input_quantizer ← cv1.conv._input_quantizer  (SHARED)
  Step 3: addop._input0_quantizer   ← cv1.conv._input_quantizer  (SHARED)
  Step 4: addop._input1_quantizer   ← cv1.conv._input_quantizer  (SHARED)

  Result: 1 master quantizer, 3 references → perfect quantization alignment
```

#### Fundamental Conv Block Structure
```
                    ┌────────────────────────────────────┐
                    │      Conv Block (Basic Unit)       │
                    └────────────────────────────────────┘

Input
  │
  └─► Conv2d (kernel, stride, padding, groups, dilation) ──[weights]
         │
         └─► BatchNorm2d (channels) ──[running_mean, running_var, weight, bias]
                │
                └─► Activation (default: SiLU/Swish) ──[learnable in some cases]
                       │
                       └─► Output

Quantization in Conv Block:
  - Conv2d → QuantConv2d
    * _input_quantizer: Quantizes input activations (8-bit)
    * _weight_quantizer: Quantizes convolution weights (8-bit)
  - BatchNorm2d: Stays FP32 during training, fused during inference
  - Activation: Stays FP32, output is quantized by next layer's input quantizer

Standard Usage:
  - cv1, cv2, cv3: Conv layers in various blocks
  - Each Conv layer gets its own input and weight quantizers
  - Quantizer sharing rules override this default behavior
```

#### Hierarchical Block Composition
```
YOLOv11 Backbone/Neck Layer (Example):
  │
  ├─► Conv (stride=2, downsampling)
  │
  ├─► C3k2 Block
  │     │
  │     ├─► cv1: Conv(c1, 2*c, 1×1)
  │     │     └─► [Quantizer A - major]
  │     │
  │     ├─► QuantChunk [shares Quantizer A]
  │     │     └─► Splits into 2 chunks (c each)
  │     │
  │     ├─► Bottleneck[0]
  │     │     ├─► cv1: Conv(c, c_, k, 1)  [Quantizer B]
  │     │     ├─► cv2: Conv(c_, c, k, 1)  [shares Quantizer B]
  │     │     └─► QuantAdd [both inputs share Quantizer B]
  │     │
  │     ├─► Bottleneck[1] ... [similar structure]
  │     │
  │     └─► cv2: Conv((2+n)*c, c2, 1×1)
  │
  ├─► QuantUpsample (if upsampling needed)
  │     └─► [Quantizer C for input]
  │
  └─► C3k Block (if present)
        │
        ├─► cv1: Conv(c1, c_, 1×1)  [shares Quantizer D from cv3]
        ├─► cv2: Conv(c1, c_, 1×1)  [shares Quantizer D from cv3]
        ├─► Bottleneck sequence
        └─► cv3: Conv(2*c_, c2, 1×1)  [Quantizer D - major]

Key Insight:
  - Each block type has its own quantizer sharing pattern
  - C3k2: cv1 → chunkop sharing
  - C3k: cv3 is major, cv1/cv2 share
  - Bottleneck: cv1 is major, cv2/addop share
  - Blocks are nested (C3k2 contains Bottlenecks)
```

#### Rule 4: MaxPool2d Replacement
```python
if isinstance(module, torch.nn.MaxPool2d):
    quant_maxpool2d = quant_nn.QuantMaxPool2d(
        module.kernel_size, module.stride, module.padding,
        module.dilation, module.ceil_mode,
        quant_desc_input=QuantDescriptor(num_bits=8, calib_method='histogram')
    )
    set_module(model, name, quant_maxpool2d)
```

**Logic**:
- Standard `MaxPool2d` replaced with `QuantMaxPool2d`
- Uses 8-bit histogram calibration
- Preserves all pooling parameters

---

### Complete Quantization Flow Example

**Example: C3k2 block with n=2 Bottlenecks (complete data flow)**
```
                         Input Tensor (FP32, c1 channels)
                                      │
                                      │
        ┌─────────────────────────────┴───────────────────────────┐
        │                      C3k2 BLOCK                         │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │ cv1: Conv(c1, 2*c, 1×1)                          │   │
        │  │   Conv2d ──[INPUT_QUANTIZER_A] ──► QuantConv2d   │   │
        │  │          ──[WEIGHT_QUANTIZER_A]──►               │   │
        │  │   BatchNorm2d (FP32)                             │   │
        │  │   SiLU() (FP32)                                  │   │
        │  └────────────────────┬─────────────────────────────┘   │
        │                       │ (FP32, 2*c channels)            │
        │                       │                                 │
        │  ┌────────────────────▼──────────────────────────┐      │
        │  │ QuantChunk                                    │      │
        │  │   INPUT_QUANTIZER_A ──► quantize(input)       │      │
        │  │   torch.split() ──► (chunk0, chunk1)          │      │
        │  └─────────┬──────────────────┬──────────────────┘      │
        │            │                  │                         │
        │  (FP32, c) │                  │ (FP32, c)               │
        │            │                  │                         │
        │  ┌─────────▼──────────────────▼──────────────────┐      │
        │  │ Path 0: chunk0 (identity, no processing)      │      │
        │  └───────────────────────┬───────────────────────┘      │
        │                          │                              │
        │  ┌───────────────────────▼───────────────────────┐      │
        │  │ Path 1: Bottleneck[0] (c → c)                 │      │
        │  │  ┌─────────────────────────────────────┐      │      │
        │  │  │ cv1: Conv(c, c_, k, 1)              │      │      │
        │  │  │   [INPUT_QUANTIZER_B0]              │      │      │
        │  │  └──────────┬──────────────────────────┘      │      │
        │  │  ┌──────────▼──────────────────────────┐      │      │
        │  │  │ cv2: Conv(c_, c, k, 1)              │      │      │
        │  │  │   [shares INPUT_QUANTIZER_B0]       │      │      │
        │  │  └──────────┬──────────────────────────┘      │      │
        │  │             │                                 │      │
        │  │  ┌──────────▼──────────────────────────┐      │      │
        │  │  │ QuantAdd(shortcut, cv2_output)      │      │      │
        │  │  │   [both inputs use QUANTIZER_B0]    │      │      │
        │  │  └──────────┬──────────────────────────┘      │      │
        │  └─────────────┼──────────────────────────────┬──┘      │
        │                │ (FP32, c)                    │         │
        │                │                              │         │
        │  ┌─────────────▼──────────────────────────────▼─┐       │
        │  │ Path 2: Bottleneck[1] (c → c)                │       │
        │  │   [Similar structure with QUANTIZER_B1]      │       │
        │  └─────────────┬──────────────────────────────┬─┘       │
        │                │ (FP32, c)                    │         │
        │                │                              │         │
        │  ┌─────────────▼──────────────────────────────▼─┐       │
        │  │ Concat: [chunk0, bottleneck0_out,            │       │
        │  │          bottleneck1_out]                    │       │
        │  │   torch.cat(..., dim=1)                      │       │
        │  └─────────────┬──────────────────────────────┬─┘       │
        │                │ (FP32, (2+n)*c = 4*c channels)         │
        │                │                                        │
        │  ┌─────────────▼────────────────────────────────┐       │
        │  │ cv2: Conv((2+n)*c, c2, 1×1)                  │       │
        │  │   Conv2d ──[INPUT_QUANTIZER_C]──► QuantConv2d│       │
        │  │          ──[WEIGHT_QUANTIZER_C]──►           │       │
        │  │   BatchNorm2d (FP32)                         │       │
        │  │   SiLU() (FP32)                              │       │
        │  └──────────────────┬───────────────────────────┘       │
        └─────────────────────┼───────────────────────────────────┘
                              │
                       Output Tensor (FP32, c2 channels)

Quantization Points Summary:
  ✓ INPUT_QUANTIZER_A: Quantizes input to cv1 Conv
  ✓ WEIGHT_QUANTIZER_A: Quantizes cv1 Conv weights
  ✓ INPUT_QUANTIZER_A (shared): Used by QuantChunk for splitting
  ✓ INPUT_QUANTIZER_B0: Quantizes bottleneck[0] cv1 input
  ✓ WEIGHT_QUANTIZER_B0: Quantizes bottleneck[0] cv1 weights
  ✓ INPUT_QUANTIZER_B0 (shared): Used by bottleneck[0] cv2
  ✓ WEIGHT_QUANTIZER_B0_cv2: Quantizes bottleneck[0] cv2 weights
  ✓ INPUT_QUANTIZER_B0 (shared): Used by QuantAdd inputs (both)
  ✓ [Similar for Bottleneck[1] with QUANTIZER_B1]
  ✓ INPUT_QUANTIZER_C: Quantizes input to cv2 Conv
  ✓ WEIGHT_QUANTIZER_C: Quantizes cv2 Conv weights

Note: All intermediate activations remain FP32, quantization happens
      at layer boundaries via TensorQuantizer modules
```

### Block Comparison Summary

| Block Type | Structure | Input Channels | Output Channels | Quantizer Sharing Rule | Custom Quantized Ops |
|------------|-----------|----------------|-----------------|------------------------|----------------------|
| **Conv** | Conv2d + BN + Act | c1 | c2 | None (standard quantization) | None |
| **Bottleneck** | cv1 → cv2 → [+shortcut] | c1 | c2 | cv1 is major → cv2, addop share | QuantAdd (if shortcut) |
| **C3k2** | cv1 → split → Bottlenecks → concat → cv2 | c1 | c2 | cv1 → chunkop shares | QuantChunk |
| **C3k** | [cv1 → Bottlenecks] + cv2 → concat → cv3 | c1 | c2 | cv3 is major → cv1, cv2 share | None |
| **QuantUpsample** | Quantizer → Interpolate | c | c | None (independent quantizer) | Replaces nn.Upsample |
| **QuantMaxPool2d** | Quantizer → MaxPool2d | c | c | Via ONNX rules | Replaces nn.MaxPool2d |

**Key Differences**:

1. **C3k2 vs C3k**:
   - C3k2: Uses QuantChunk for splitting, quantizer shared from cv1
   - C3k: No chunk operation, cv3 is the major quantizer shared backwards to cv1/cv2

2. **Bottleneck**:
   - Only block with QuantAdd (for residual connection)
   - All components share the same quantizer from cv1
   - Critical for INT8 addition correctness

3. **Quantizer Flow Direction**:
   - C3k2: Forward sharing (cv1 → chunkop)
   - C3k: Backward sharing (cv3 → cv1, cv2)
   - Bottleneck: Forward sharing (cv1 → cv2, addop)

4. **Nesting**:
   - C3k2 and C3k contain multiple Bottleneck blocks
   - Each Bottleneck has its own independent quantizer set
   - Parent block quantizers (cv1, cv2, cv3) are separate from child Bottleneck quantizers

---

## ONNX-Based Quantizer Pairing Rules

### Overview
**Location**: `rules_v2.py:56-110`

After exporting to ONNX, the graph is analyzed to find operations that should share quantizers (quantizer pairing). This ensures TensorRT INT8 optimization.

### Rule 1: Concat Node Analysis
**Location**: `rules_v2.py:60-88`

```python
if node.op_type == "Concat":
    # Find all QuantizeLinear nodes consuming Concat output
    # Identify "major" quantizer (first Conv after Concat)
    # Match all other Convs to use major's quantizer
```

**Logic**:
1. Find all `QuantizeLinear` nodes that take `Concat` **output**
2. Trace each to its following `Conv` layer
3. First `Conv` becomes the "major" quantizer
4. All other `Conv` layers are paired to the major:
   - **Type A**: Other Conv layers that also consume Concat output (parallel branches)
   - **Type B**: Conv layers on the **input paths** to Concat (the branches being concatenated)
5. Additionally pairs all `Resize` and `MaxPool` operators on Concat **input** paths

**Detailed Pairing Relationships**:
```
                        Input Branches (Type B - paired to major)
                               │
                    ┌──────────┼────────┐
                    │          │        │
                    │          │        │
        Conv(sub) ──┼─► Q ──┐  │  Q ◄── ┼── Conv(sub)
                    │       │  │  │     │
          Resize ───┼─► Q ──┼──┴──┼─ Q ◄│─── MaxPool
                    │       │     │     │
                    │       └─► Concat ◄┘
                    │            │
                    │            └─► QuantizeLinear ─► DequantizeLinear ─► Conv (MAJOR)
                    │            │                                           ↑
                    │            └─► QuantizeLinear ─► DequantizeLinear ─► Conv(sub) [Type A]
                    │                                                        ↑
                    └────────────────────────────────────────────────────────┘
                              All these Conv layers share the MAJOR quantizer

Where:
  - Q = QuantizeLinear node
  - Conv(sub) = Subordinate Conv layers that get paired to major
  - Conv(MAJOR) = The first Conv after Concat (master quantizer)
  - Type A: Conv layers consuming Concat output (parallel downstream paths)
  - Type B: Conv layers producing Concat inputs (parallel upstream paths)
```

**Code Flow (rules_v2.py:60-88)**:
```python
# Step 1: Find Convs AFTER Concat output
qnodes = find_all_with_input_node(model, node.output[0])
major = None
for qnode in qnodes:
    if qnode.op_type != "QuantizeLinear": continue
    conv = find_quantizelinear_conv(model, qnode)
    if conv is None: continue

    if major is None:
        # First Conv becomes MAJOR
        major_name = find_quantize_conv_name(model, conv.input[1])
        major = major_name
    else:
        # Subsequent Convs after Concat (Type A)
        sub_name = find_quantize_conv_name(model, conv.input[1])
        match_pairs.append([major, sub_name])  # Pair to major

# Step 2: Find Convs BEFORE Concat (on input branches)
for subnode in model.graph.node:
    # Find QuantizeLinear nodes on Concat INPUT branches
    if subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
        subconv = find_quantizelinear_conv(model, subnode)
        if subconv is not None:
            sub_name = find_quantize_conv_name(model, subconv.input[1])
            match_pairs.append([major, sub_name])  # Type B pairing

    # Also handle Resize/MaxPool on Concat inputs
    if subnode.op_type in ["Resize", "MaxPool"] and subnode.output[0] in node.input:
        sub_name = ".".join((subnode.name).split("/")[:-1])[1:]
        match_pairs.append([major, sub_name])  # Pair to major
```

### Rule 2: MaxPool Node Analysis
**Location**: `rules_v2.py:90-109`

```python
if node.op_type == "MaxPool":
    # Find QuantizeLinear after MaxPool
    # Find Conv after that (major)
    # Match all Convs sharing MaxPool input to major
```

**Logic**:
1. Find `QuantizeLinear` after `MaxPool` **output**
2. Trace to following `Conv` (becomes **major**)
3. Find all `QuantizeLinear` nodes that share MaxPool's **input** (parallel branches)
4. Pair all their following `Conv` layers to major

**Detailed Pairing Relationships**:
```
                    Shared Input Point
                            │
                            ├─────────────────────────────┐
                            │                             │
                            │ [Main Path]                 │ [Parallel Paths]
                            │                             │
                            └─► MaxPool                   ├─► QuantizeLinear ─► Conv(sub1) ─┐
                                   │                      │                                 │
                                   │                      ├─► QuantizeLinear ─► Conv(sub2) ─┤
                                   │                      │                                 │
                                   └─► QuantizeLinear     └─► QuantizeLinear ─► Conv(subN) ─┤
                                          │                                                 │
                                          └─► DequantizeLinear                              │
                                                 │                                          │
                                                 └─► Conv (MAJOR) ◄─────────────────────────┘
                                                        ↑
                                          All Conv layers share this quantizer

Where:
  - Conv(MAJOR): The Conv layer after MaxPool (master quantizer)
  - Conv(sub1..N): Conv layers on parallel branches from the same input
  - These parallel branches split BEFORE MaxPool but share the same input tensor
```

**Example YOLO Scenario**:
```
Feature Map (e.g., C3k output)
      │
      ├────────────► Conv(256→128) ─► [parallel branch 1, becomes sub1]
      │
      ├────────────► Conv(256→64)  ─► [parallel branch 2, becomes sub2]
      │
      └────────────► MaxPool(3x3)  ─► Conv(256→128) ─► [main path, becomes MAJOR]
                                           ↑
                            All branches share this quantizer
```

**Code Flow (rules_v2.py:90-109)**:
```python
# Step 1: Find Conv AFTER MaxPool (becomes major)
qnode = find_with_input_node(model, node.output[0])
if qnode and qnode.op_type == "QuantizeLinear":
    major_conv = find_quantizelinear_conv(model, qnode)
    major_name = find_quantize_conv_name(model, major_conv.input[1])
    major = major_name

# Step 2: Find all Convs on parallel branches (share MaxPool's input)
same_input_nodes = find_all_with_input_node(model, node.input[0])
for same_input_node in same_input_nodes:
    if same_input_node.op_type == "QuantizeLinear":
        subconv = find_quantizelinear_conv(model, same_input_node)
        if subconv is not None:
            sub_name = find_quantize_conv_name(model, subconv.input[1])
            match_pairs.append([major, sub_name])  # Pair to major
```

**Why This Matters**:
- SPP/SPPF modules in YOLO use MaxPool with multiple parallel paths
- All paths process the same input feature map at different scales
- Sharing quantizers ensures consistent quantization across scales
- Prevents quantization mismatch at feature fusion points

### Application
**Location**: `quantize_11.py:307-322`

```python
export_onnx(model, "quantization-custom-rules-temp.onnx")
pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
for major, sub in pairs:
    print(f"ONNX Rules: {sub} match to {major}")
    get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
```

**Process**:
1. Export model to temporary ONNX file
2. Analyze ONNX graph for pairing rules
3. Apply quantizer sharing in PyTorch model
4. Delete temporary ONNX file
5. Log all applied pairings

---

## Fine-Tuning Rules

### QAT Fine-Tuning Process
**Location**: `quantize_11.py:453-564`

#### Overview
Uses knowledge distillation from FP32 model to train INT8 quantized model.

#### Model Setup Rules
```python
origin_model = deepcopy(model).eval()
disable_quantization(origin_model).apply()
model.train()
origin_model.train()  # Set to train mode but no gradients
```

**Rules**:
- **Origin model**: FP32 copy with quantization disabled, no gradients
- **Quantized model**: INT8 model with gradients enabled
- Both set to `.train()` mode for consistent batch norm behavior

#### Optimizer Configuration
```python
optimizer = optim.Adam(model.parameters(), learningrate)
quant_lossfn = torch.nn.MSELoss()
```

**Default Rules**:
- **Optimizer**: Adam
- **Initial learning rate**: `1e-5`
- **Loss function**: MSE (Mean Squared Error) between quantized and FP32 outputs
- **FP16 training**: Enabled by default (`fp16=True`)

#### Learning Rate Schedule
**Default Schedule** (`lrschedule`):
```python
{
    0: 1e-6,   # Epochs 0-2: warm-up with very low LR
    3: 1e-5,   # Epochs 3-7: normal training LR
    8: 1e-6    # Epochs 8+: fine-tuning with low LR
}
```

**Rules**:
- Learning rate changes at epoch boundaries
- Can be customized via `lrschedule` parameter
- Applied at start of each epoch

#### Supervision Policy
**Purpose**: Select which layers to compute distillation loss

**Default Policy** (`qat_yolov11.py:460-486`):
```python
def supervision_policy():
    supervision_list = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            supervision_list.append((name, id(module)))

    # Select every N-th module (supervision_stride)
    keep_modules = []
    for i in range(0, len(supervision_list), supervision_stride):
        keep_modules.append(supervision_list[i][1])

    # Always add detection/head layers
    for name, module in model.named_modules():
        if 'detect' in name.lower() or 'head' in name.lower():
            keep_modules.append(id(module))
```

**Rules**:
1. **Stride selection**: Select every `supervision_stride`-th module (default: 1 = all modules)
2. **Detection layers**: Always supervised regardless of stride
3. **Head layers**: Always supervised regardless of stride
4. **TensorQuantizers**: Automatically skipped
5. Prints: `"Supervision: {name} will compute loss with origin model during QAT training"`

#### Training Loop Rules
**Location**: `quantize_11.py:500-564`

**Process per epoch**:
1. Register forward hooks on supervised modules
2. Run forward pass on both quantized and FP32 models
3. Compute MSE loss between outputs of supervised layers
4. Backpropagate and update quantized model weights
5. Remove hooks
6. Call `per_epoch_callback` for evaluation

**Rules**:
- **Batches per epoch**: Limited by `early_exit_batchs_per_epoch` (default: 200)
- **Total epochs**: `nepochs` (default: 10)
- **FP16 training**: Uses `torch.cuda.amp.GradScaler` when enabled
- **Gradient accumulation**: Not used (update every batch)
- **Loss computation**: Only for layers passing `supervision_policy`
- **Shape mismatch**: Logged as warning, loss not computed for that layer

#### Epoch Callback Rules
**Location**: `qat_yolov11.py:417-428`

```python
def per_epoch_callback(model, epoch, lr):
    nonlocal best_ap
    ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
    summary.append([f"QAT{epoch}", ap])
    print(f"Epoch {epoch}, mAP: {ap:.5f}")

    if ap > best_ap:
        print(f"Save QAT model to {save_qat} @ {ap:.5f}")
        best_ap = ap
        torch.save({"model": model}, f'{save_qat}')

    return False  # Continue training (True = early stop)
```

**Rules**:
- Called after each epoch completes
- Evaluates model on validation set
- **Best model saving**: Only saves when mAP improves
- Logs all results to `summary.json`
- Returns `False` to continue training (can return `True` for early stopping)

#### Preprocessing Rules
**Location**: `qat_yolov11.py:430-458`

```python
def preprocess(datas):
    imgs = extract_images_from_batch(datas, device)
    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
        imgs = imgs.float() / 255.0
    return imgs
```

**Rules**:
- Extract images from batch (supports dict, list, tuple formats)
- Transfer to device with `non_blocking=True`
- Normalize to [0, 1] range if needed:
  - `uint8` tensors: `imgs.float() / 255.0`
  - Values > 1.0: `imgs / 255.0`
- Handles errors gracefully with fallback logic

---

## Export Rules

### ONNX Export Configuration
**Location**: `quantize_11.py:299-304` and `qat_yolov11.py:314-352`

#### Fake Quantization Mode
```python
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, input, file, *args, **kwargs)
quant_nn.TensorQuantizer.use_fb_fake_quant = False
```

**Rules**:
- **Enable fake quantization**: `use_fb_fake_quant = True` before export
- Uses Facebook-style fake quantization nodes in ONNX
- **Disable after export**: Restore to `False`
- Model set to `.eval()` mode
- Uses `torch.no_grad()` context

#### YOLOv11 Export Settings
**Location**: `qat_yolov11.py:314-352`

```python
# Set export flags on all modules
for m in model.modules():
    if hasattr(m, 'export'):
        m.export = True
    if hasattr(m, 'format'):
        m.format = 'onnx'
```

**Rules**:
- **Export mode**: Set `module.export = True` for all compatible modules
- **Format flag**: Set `module.format = 'onnx'`
- **Input name**: `"images"`
- **Opset version**: 13 (for TensorRT compatibility)
- **Dynamic batch**: Optional via `dynamic_batch` parameter

**Output configurations**:

1. **With anchors** (`noanchor=False`):
   - Output names: `["output0"]`
   - Dynamic axes: `{"images": {0: "batch"}, "output0": {0: "batch"}}`

2. **Without anchors** (`noanchor=True`):
   - Output names: `["output0", "output1", "output2"]`
   - Dynamic axes: `{"images": {0: "batch"}, "output0": {0: "batch"}, "output1": {0: "batch"}, "output2": {0: "batch"}}`

#### Post-Export Cleanup
```python
# Reset export flags
for m in model.modules():
    if hasattr(m, 'export'):
        m.export = False
```

**Rule**: Always reset `export=False` after ONNX export completes

---

## Summary of Critical Rules

### Must-Do Rules (Non-negotiable)
1. ✅ Call `quantize.initialize()` before any quantization operations
2. ✅ Apply ignore policy for attention and DFL layers
3. ✅ Use histogram calibration with `_torch_hist = True`
4. ✅ Share quantizers in C3k2, C3k, and Bottleneck modules
5. ✅ Apply ONNX-based quantizer pairing rules
6. ✅ Replace Bottleneck, C3k2 forwards and Upsample modules
7. ✅ Enable fake quantization mode during ONNX export

### Recommended Rules (Best practices)
1. ⚡ Use 25 batches for calibration (balance between speed and accuracy)
2. ⚡ Apply supervision policy with stride=1 for better QAT convergence
3. ⚡ Use MSE method for histogram amax computation
4. ⚡ Enable FP16 training for faster QAT
5. ⚡ Save best model based on validation mAP
6. ⚡ Use learning rate schedule: 1e-6 → 1e-5 → 1e-6

### Module-Specific Rules Matrix

| Module | Quantization | Custom Op | Quantizer Sharing | Notes |
|--------|--------------|-----------|-------------------|-------|
| Conv2d | ✅ Yes | - | Via ONNX rules | Standard replacement |
| Linear | ✅ Yes | - | - | Standard replacement |
| Bottleneck | ✅ Yes | QuantAdd | cv1 → cv2 → addop | Shares cv1 quantizer |
| C3k2 | ✅ Yes | QuantChunk | cv1 → chunkop | Chunk uses cv1 quantizer |
| C3k | ✅ Yes | - | cv3 → cv1, cv2 | All share cv3 quantizer |
| Upsample | ✅ Yes | QuantUpsample | - | Complete replacement |
| MaxPool2d | ✅ Yes | - | Via ONNX rules | Replaced with QuantMaxPool2d |
| Attention | ❌ No | - | - | Ignored via policy |
| DFL | ❌ No | - | - | Ignored via policy |
| Detection Head | ❌ No | - | - | Ignored via policy |

---

## Command Reference

### Quantize Command
```bash
python qat_yolov11.py quantize <weight_file> \
    --cocodir <path> \
    --device cuda:0 \
    --ptq <ptq_output.pt> \
    --qat <qat_output.pt> \
    --supervision-stride 1 \
    --iters 200 \
    --eval-origin \
    --eval-ptq
```

**Applied rules**: All rules in order:
1. Initialize quantization
2. Replace custom module forwards
3. Replace to quantization modules (with ignore policy)
4. Calibrate model (25 batches, histogram, MSE)
5. Apply custom rules (ONNX + module-specific)
6. Evaluate origin (optional)
7. Evaluate PTQ (optional)
8. QAT fine-tuning (if `--qat` specified)

### Export Command
```bash
python qat_yolov11.py export
```

**Applied rules**:
1. Initialize quantization
2. Replace custom module forwards
3. Replace to quantization modules (with ignore policy)
4. Load calibration state
5. Export to ONNX (fake quant mode, opset 13)

---

## Troubleshooting Common Issues

### Issue 1: Calibration Failure
**Symptoms**: "0 successful, N failed" calibrations

**Check**:
- `_torch_hist = True` is set for all quantizers
- Dataloader returns valid tensors
- Input preprocessing normalizes to [0, 1]

### Issue 2: Poor PTQ Accuracy
**Symptoms**: Large mAP drop after PTQ

**Solutions**:
- Increase calibration batches (`num_batch=50`)
- Ensure ignore policy covers sensitive layers
- Verify ONNX rules are applied correctly
- Run sensitive analysis to identify problematic layers

### Issue 3: QAT Not Improving
**Symptoms**: mAP doesn't improve during fine-tuning

**Solutions**:
- Check supervision policy (too few supervised layers?)
- Verify learning rate schedule
- Ensure origin model has quantization disabled
- Check for shape mismatches in supervised layers

### Issue 4: ONNX Export Failure
**Symptoms**: Export crashes or produces invalid ONNX

**Solutions**:
- Ensure model is in `.eval()` mode
- Verify all custom modules have ONNX-compatible operations
- Check that `use_fb_fake_quant = True` is set
- Ensure input tensor is on correct device

---

## Version Notes

- **Quantization library**: NVIDIA `pytorch_quantization`
- **Model**: YOLOv11 (compatible with YOLOv8 architecture)
- **ONNX opset**: 13 (TensorRT compatible)
- **Quantization**: INT8 (8-bit)
- **Random seed**: 57 (for reproducibility)

---

*Last updated: 2026-01-27*
*Codebase version: YOLOv11 QAT with TensorRT INT8 support*
