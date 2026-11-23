import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
import torch.nn.functional as F

# --- 基础初始化函数 ---
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# --- 基础卷积组件 ---
def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

# --- Decoder 核心组件 ---
class Unpool(nn.Module):
    def __init__(self, stride=2):
        super(Unpool, self).__init__()
        self.stride = stride
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:,:,0,0] = 1
    def forward(self, x):
        num_channels = x.size(1)
        return F.conv_transpose2d(x, self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1),
                                  stride=self.stride, groups=num_channels)

def upconv(in_channels, out_channels):
    return nn.Sequential(
        Unpool(2),
        nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class upproj(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upproj, self).__init__()
        self.unpool = Unpool(2)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return F.relu(x1 + x2)

# --- 具体的 Decoder 类实现 ---
class DeConv(nn.Module):
    def __init__(self, kernel_size, dw):
        super(DeConv, self).__init__()
        self.convt1 = nn.ConvTranspose2d(1024, 512, kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=kernel_size%2)
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=kernel_size%2)
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=kernel_size%2)
        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=kernel_size%2)
        self.convt5 = nn.ConvTranspose2d(64, 32, kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=kernel_size%2)
        self.convf = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))
        x = F.relu(self.convt5(x))
        return self.convf(x)

class UpConv(nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.upconv1 = upconv(1024, 512)
        self.upconv2 = upconv(512, 256)
        self.upconv3 = upconv(256, 128)
        self.upconv4 = upconv(128, 64)
        self.upconv5 = upconv(64, 32)
        self.convf = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x = self.upconv1(x); x = self.upconv2(x); x = self.upconv3(x)
        x = self.upconv4(x); x = self.upconv5(x)
        return self.convf(x)

class UpProj(nn.Module):
    def __init__(self):
        super(UpProj, self).__init__()
        self.upproj1 = upproj(1024, 512)
        self.upproj2 = upproj(512, 256)
        self.upproj3 = upproj(256, 128)
        self.upproj4 = upproj(128, 64)
        self.upproj5 = upproj(64, 32)
        self.convf = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        x = self.upproj1(x); x = self.upproj2(x); x = self.upproj3(x)
        x = self.upproj4(x); x = self.upproj5(x)
        return self.convf(x)

# --- 核心控制类 ---
class Decoder(nn.Module):
    names = ['deconv{}{}'.format(i,dw) for i in range(3,10,2) for dw in ['', 'dw']]
    names += ["upconv", "upproj", "fpnlite"]


class DSConvBlock(nn.Module):
    """Depthwise-separable conv block for mobile-friendly decoding."""
    def __init__(self, in_channels, out_channels):
        super(DSConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FPNLite(nn.Module):
    """
    Lightweight top-down decoder:
    - takes multi-scale features [c4, c3, c2, c1] from MobileNetV2
    - p1 is at 1/4 resolution; f0 refines at 1/2 resolution before final output
    - outputs main prediction at 1/2 resolution (2x upsample to full) + aux at 1/4
    """
    def __init__(self, in_channels=(1280, 96, 32, 24), feat_channels=96):
        super(FPNLite, self).__init__()
        c4, c3, c2, c1 = in_channels
        self.l4 = nn.Conv2d(c4, feat_channels, 1, 1, 0, bias=False)
        self.l3 = nn.Conv2d(c3, feat_channels, 1, 1, 0, bias=False)
        self.l2 = nn.Conv2d(c2, feat_channels, 1, 1, 0, bias=False)
        self.l1 = nn.Conv2d(c1, feat_channels, 1, 1, 0, bias=False)

        self.f3 = DSConvBlock(feat_channels, feat_channels)
        self.f2 = DSConvBlock(feat_channels, feat_channels)
        self.f1 = DSConvBlock(feat_channels, feat_channels)
        # 1/2 resolution refinement — eliminates 4x interpolation artifacts
        self.f0 = DSConvBlock(feat_channels, feat_channels)

        # head_main outputs at 1/2 resolution (only 2x upsample needed)
        self.head_main = nn.Conv2d(feat_channels, 1, kernel_size=3, stride=1, padding=1)
        self.head_aux = nn.Conv2d(feat_channels, 1, kernel_size=3, stride=1, padding=1)

        # Zero-init the prediction heads so initial output is near-zero, not random
        nn.init.zeros_(self.head_main.weight)
        nn.init.zeros_(self.head_main.bias)
        nn.init.zeros_(self.head_aux.weight)
        nn.init.zeros_(self.head_aux.bias)

        # Apply He init to all layers except the heads (already zeroed above)
        for m in [self.l4, self.l3, self.l2, self.l1,
                  self.f3, self.f2, self.f1, self.f0]:
            m.apply(weights_init)

    def forward(self, c4, c3, c2, c1):
        p4 = self.l4(c4)
        p3 = self.f3(self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.f2(self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.f1(self.l1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False))
        # Upsample to 1/2 resolution and refine — avoids 4x blind interpolation
        p0 = self.f0(F.interpolate(p1, scale_factor=2, mode='bilinear', align_corners=False))
        return self.head_main(p0), self.head_aux(p1)

class GuidedUpsample(nn.Module):
    """
    RGB-guided depth upsampling: uses shallow RGB edge features to sharpen
    depth boundaries at full resolution, replacing blind bilinear interpolation.
    Adds ~2K parameters (feat=16).
    """
    def __init__(self, rgb_channels=3, depth_channels=1, feat=16):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(rgb_channels, feat, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(depth_channels + feat, depth_channels, 1)
        # Zero-init so the module starts as identity (pure bilinear) at epoch 0
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.rgb_enc.apply(weights_init)

    def forward(self, depth_low, rgb_full):
        depth_up = F.interpolate(depth_low, size=rgb_full.shape[-2:],
                                 mode='bilinear', align_corners=False)
        rgb_feat = self.rgb_enc(rgb_full)
        return depth_up + self.fuse(torch.cat([depth_up, rgb_feat], dim=1))


def choose_decoder(decoder):
    if decoder == "fpnlite":
        # FPNLite manages its own init (zero-init heads + He for feature layers)
        return FPNLite()
    elif decoder == "upconv":
        model = UpConv()
    elif decoder == "upproj":
        model = UpProj()
    elif "deconv" in decoder:
        kernel_size = 3
        for k in [3, 5, 7, 9]:
            if f"deconv{k}" in decoder:
                kernel_size = k
                break
        model = DeConv(kernel_size=kernel_size, dw=("dw" in decoder))
    else:
        model = DeConv(kernel_size=3, dw=False)
    model.apply(weights_init)
    return model

# --- 引入CA注意力模块 ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True) # RK3588对ReLU支持最好

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# --- 最终网络结构 ---
class MobileNet(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, pretrained=True, use_ca=True,
                 use_guided_upsample=False):
        super(MobileNet, self).__init__()
        self.output_size = output_size
        self.use_ca = use_ca
        self.decoder_name = decoder
        self.use_guided_upsample = use_guided_upsample
        mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)

        # 提取特征
        self.features = mobilenet.features

        # --- 新增：插入 Coordinate Attention ---
        # MobileNetV2 最后一个 feature 层输出通道数为 1280
        self.ca = CoordAtt(1280, 1280)
        self.decoder = choose_decoder(decoder)

        # Legacy decoders still expect 1024-channel feature input.
        self.conv_adapt = nn.Conv2d(1280, 1024, 1)

        # RGB-guided depth upsampling: uses shallow RGB edges to sharpen depth boundaries
        if self.use_guided_upsample:
            self.guided_upsample = GuidedUpsample(rgb_channels=3, depth_channels=1, feat=16)

    def _extract_multiscale_features(self, x):
        c1 = c2 = c3 = c4 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 3:   # 1/4, 24c
                c1 = x
            elif i == 6: # 1/8, 32c
                c2 = x
            elif i == 13: # 1/16, 96c
                c3 = x
            elif i == 18: # 1/32, 1280c
                c4 = x
        return c1, c2, c3, c4

    def forward(self, x, return_aux=False):
        input_size = (x.size(2), x.size(3))
        # Preserve original RGB for guided upsampling (x gets consumed by feature extraction)
        rgb_orig = x if self.use_guided_upsample else None

        if self.decoder_name == "fpnlite":
            c1, c2, c3, c4 = self._extract_multiscale_features(x)
            if self.use_ca:
                c4 = self.ca(c4)
            main_out, aux_out = self.decoder(c4, c3, c2, c1)
            # main_out is at 1/2 resolution; use RGB-guided upsampling for sharper edges
            if self.use_guided_upsample:
                x = self.guided_upsample(main_out, rgb_orig)
            else:
                x = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=False)
            aux = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
        else:
            # 1. 基础特征提取
            x = self.features(x)

            # 2. 增强空间位置感知（提升远景精度关键）
            if self.use_ca:
                x = self.ca(x)

            # 3. 降维与解码
            x = self.conv_adapt(x)
            x = self.decoder(x)
            # Non-fpnlite decoders can also use guided upsample
            if self.use_guided_upsample:
                x = self.guided_upsample(x, rgb_orig)
            else:
                x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
            aux = None
        # Depth should be positive; Softplus keeps gradients smooth near zero.
        x = F.softplus(x) + 1e-6
        if aux is not None:
            aux = F.softplus(aux) + 1e-6
        if return_aux:
            return x, aux
        return x
