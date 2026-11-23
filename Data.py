import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class UseGeoDataset(data.Dataset):
    def __init__(
        self,
        root_path,
        split='train',
        depth_scale=100.0,
        teacher_scale=100.0,
        teacher_path=None,
        target_size=(512, 768)
    ):
        self.root_path = root_path
        self.split = split
        self.target_size = tuple(target_size)
        self.depth_scale = float(depth_scale)
        self.teacher_scale = float(teacher_scale)

        # 基础路径
        data_root = os.path.join(root_path, split)
        self.rgb_path = os.path.join(data_root, 'rgb')
        self.depth_path = os.path.join(data_root, 'depth')

        # 新增：Teacher 伪标签路径 (目录名需与上一节生成脚本中的一致)
        self.teacher_path = teacher_path if teacher_path is not None else r"data\teacher_depth"

        if not os.path.exists(self.rgb_path):
            raise RuntimeError(f"找不到RGB目录: {self.rgb_path}")

        self.filenames = sorted(os.listdir(self.rgb_path))

    def __getitem__(self, index):
        # 1. 获取RGB和GT深度图的文件名
        img_name = self.filenames[index]
        rgb_file = os.path.join(self.rgb_path, img_name)

        depth_name = img_name.replace('_res.jpg', '_depth_res.tiff')
        depth_file = os.path.join(self.depth_path, depth_name)

        if not os.path.exists(depth_file):
            raise FileNotFoundError(f"找不到对应的深度图文件!\n检查路径: {self.depth_path}")

        # 2. 读取图片并缩放
        rgb = Image.open(rgb_file).convert('RGB')
        depth = Image.open(depth_file)

        rgb = F.resize(rgb, self.target_size, Image.BILINEAR)
        depth = F.resize(depth, self.target_size, Image.NEAREST)

        # 3. 转换为 Tensor (RGB 和 GT)
        rgb_tensor = transforms.ToTensor()(rgb)
        depth_np = np.array(depth).astype(np.float32)

        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale 必须大于 0，当前值: {self.depth_scale}")
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0) / self.depth_scale

        # ------------------- 新增：读取 Teacher 标签 -------------------
        if self.split == 'train':
            # 这里是你原来的训练集读取逻辑
            teacher_name = img_name.split('.')[0] + '.tiff'
            teacher_file = os.path.join(self.teacher_path, teacher_name)

            if not os.path.exists(teacher_file):
                raise FileNotFoundError(
                    f"训练集缺少 teacher 伪标签: {teacher_file}\n"
                    f"请先生成所有训练图片的 teacher 标签，或检查 --teacher_path 参数是否正确。\n"
                    f"零填充会导致蒸馏损失与 GT 损失对抗，产生随机噪点。"
                )
            else:
                teacher_depth = Image.open(teacher_file)
                teacher_depth = F.resize(teacher_depth, self.target_size, Image.NEAREST)
                teacher_np = np.array(teacher_depth).astype(np.float32)
                if self.teacher_scale <= 0:
                    raise ValueError(f"teacher_scale 必须大于 0，当前值: {self.teacher_scale}")
                teacher_tensor = torch.from_numpy(teacher_np).unsqueeze(0) / self.teacher_scale
        else:
            # 【重要】：验证集/测试集走这里！必须定义 teacher_tensor
            # 因为验证不需要 Teacher，我们直接给一个全 0 的张量
            # 【注】验证时 DistillationLoss 中的蒸馏部分会对 0 mask 返回 0，所以安全
            teacher_tensor = torch.zeros_like(depth_tensor)


        return rgb_tensor, depth_tensor, teacher_tensor
    def __len__(self):
        return len(self.filenames)

# --- 保持不变的加载器创建函数 ---
def createDataLoaders(args):
    data_root = os.path.join('.', 'data', args.data)
    train_dataset = UseGeoDataset(
        data_root,
        split='train',
        depth_scale=args.depth_scale,
        teacher_scale=args.teacher_scale,
        teacher_path=args.teacher_path,
        target_size=(args.input_height, args.input_width)
    )
    val_dataset = UseGeoDataset(
        data_root,
        split='val',
        depth_scale=args.depth_scale,
        teacher_scale=args.teacher_scale,
        teacher_path=args.teacher_path,
        target_size=(args.input_height, args.input_width)
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

class CustomDataLoader(object):
    modality_names = ['rgb']
