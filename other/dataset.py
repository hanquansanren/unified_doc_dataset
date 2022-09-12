import os
import pickle

import lmdb
import numpy as np
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# from utils import joint_transforms


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'),
                                 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def _make_dataset(root, prefix=('.jpg', '.png')):
    img_path = os.path.join(root, 'Image')
    gt_path = os.path.join(root, 'Mask')
    img_list = [
        os.path.splitext(f)[0] for f in os.listdir(gt_path)
        if f.endswith(prefix[1])
    ]
    return [(os.path.join(img_path, img_name + prefix[0]),
             os.path.join(gt_path, img_name + prefix[1]))
            for img_name in img_list]


# class TestImageFolder(Dataset):
#     def __init__(self, root, in_size, prefix):
#         self.imgs = _make_dataset(root, prefix=prefix)
#         self.test_img_trainsform = transforms.Compose([
#             # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
#             transforms.Resize((in_size, in_size)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
    
#     def __getitem__(self, index):
#         img_path, gt_path = self.imgs[index]
        
#         img = Image.open(img_path).convert('RGB')
#         img_name = (img_path.split(os.sep)[-1]).split('.')[0]
        
#         img = self.test_img_trainsform(img)
#         return img, img_name
    
#     def __len__(self):
#         return len(self.imgs)


class TrainImageFolder(Dataset):
    def __init__(self, root, in_size, scale=1.5, use_bigt=False):
        self.use_bigt = use_bigt
        self.in_size = in_size
        self.root = root
        
        # self.train_joint_transform = joint_transforms.Compose([
        #     joint_transforms.JointResize(in_size),
        #     joint_transforms.RandomHorizontallyFlip(),
        #     joint_transforms.RandomRotate(10)
        # ])
        self.train_img_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 处理的是Tensor
        ])
        # ToTensor 操作会将 PIL.Image 或形状为 H×W×D，数值范围为 [0, 255] 的 np.ndarray 转换为形状为 D×H×W，
        # 数值范围为 [0.0, 1.0] 的 torch.Tensor。
        self.train_target_transform = transforms.ToTensor()
        
        self.gt_root = '/home/lart/coding/TIFNet/datasets/DUTSTR/DUTSTR_GT.lmdb'
        self.img_root = '/home/lart/coding/TIFNet/datasets/DUTSTR/DUTSTR_IMG.lmdb'
        self.paths_gt, self.sizes_gt = _get_paths_from_lmdb(self.gt_root)
        self.paths_img, self.sizes_img = _get_paths_from_lmdb(self.img_root)
        self.gt_env = lmdb.open(self.gt_root, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.img_env = lmdb.open(self.img_root, readonly=True, lock=False, readahead=False,
                                 meminit=False)
    
    
    def __getitem__(self, index):
        gt_path = self.paths_gt[index]
        img_path = self.paths_img[index]
        
        gt_resolution = [int(s) for s in self.sizes_gt[index].split('_')]
        img_resolution = [int(s) for s in self.sizes_img[index].split('_')]
        img_gt = _read_img_lmdb(self.gt_env, gt_path, gt_resolution)
        img_img = _read_img_lmdb(self.img_env, img_path, img_resolution)
        if img_img.shape[-1] != 3:
            img_img = np.repeat(img_img, repeats=3, axis=-1)
        img_img = img_img[:, :, [2, 1, 0]]  # bgr => rgb
        img_gt = np.squeeze(img_gt, axis=2)
        gt = Image.fromarray(img_gt, mode='L')
        img = Image.fromarray(img_img, mode='RGB')
        
        img, gt = self.train_joint_transform(img, gt)
        gt = self.train_target_transform(gt)
        img = self.train_img_transform(img)
        
        if self.use_bigt:
            gt = gt.ge(0.5).float()  # 二值化
        
        img_name = self.paths_img[index]
        return img, gt, img_name
    
    def __len__(self):
        return len(self.paths_img)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())
