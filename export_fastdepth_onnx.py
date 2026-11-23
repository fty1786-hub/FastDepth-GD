import argparse
import os
import sys
import torch
import torchvision
from torch.nn import Conv2d, BatchNorm2d, ReLU

# Add project root to path so we can import models

# torchvision 0.11.x 兼容: Conv2dNormActivation 仅 0.12+ 存在
if not hasattr(torchvision.ops.misc, "Conv2dNormActivation"):
    class _Conv2dNormActivation(torch.nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size=3,
                     stride=1, padding=None, groups=1,
                     norm_layer=None, activation_layer=None,
                     dilation=1, inplace=None, bias=None, padding_mode='zeros'):
            if padding is None:
                padding = (kernel_size - 1) // 2 * dilation
            if norm_layer is None:
                norm_layer = BatchNorm2d
            if activation_layer is None:
                activation_layer = ReLU
            bias = bias if bias is not None else (norm_layer is None)
            modules = [Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode)]
            if norm_layer is not None:
                modules.append(norm_layer(out_channels))
            if activation_layer is not None:
                modules.append(activation_layer(inplace=inplace if inplace is not None else True))
            super().__init__(*modules)
    torchvision.ops.misc.Conv2dNormActivation = _Conv2dNormActivation
    torchvision.ops.misc.ConvNormActivation = _Conv2dNormActivation  # 别名
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import models


def parse_args():
    parser = argparse.ArgumentParser(description="Export FastDepth to ONNX for RK3588 pipeline.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model_best.pth.tar")
    parser.add_argument("--output", default="fastdepth_fp32.onnx", type=str, help="Output ONNX path")
    parser.add_argument("--decoder", default="fpnlite", type=str, help="Decoder name")
    parser.add_argument("--height", default=512, type=int, help="Input height (512 or 768)")
    parser.add_argument("--width", default=768, type=int, help="Input width (768 or 1152)")
    parser.add_argument("--opset", default=13, type=int)
    parser.add_argument("--disable-ca", action="store_true", help="Disable CoordAtt")
    parser.add_argument("--use-guided-upsample", action="store_true",
                        help="Enable RGB-guided depth upsampling (must match training config)")
    return parser.parse_args()


def load_checkpoint_model(args):
    model = models.MobileNet(
        decoder=args.decoder,
        output_size=(args.height, args.width),
        pretrained=False,
        use_ca=not args.disable_ca,
        use_guided_upsample=args.use_guided_upsample,
    )
    # PyTorch 2.6+ 默认 weights_only=True，会导致 argparse.Namespace 拒绝加载
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Compatible with existing training checkpoint format.
    state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model = load_checkpoint_model(args)
    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        input_names=["input_rgb"],
        output_names=["depth"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX exported to: {args.output}")


if __name__ == "__main__":
    main()
