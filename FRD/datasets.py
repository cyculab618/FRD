import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
from pathlib import Path
import kornia
from kornia import augmentation as K
import tifffile
import imageio.v2 as imageio

value_dict = {
    'uint8': [0, 255],
    'uint16': [0, 65535],
}

def numpy_to_tensor(image_numpy):
    img_dtype = str(image_numpy.dtype)
    img_numpy = image_numpy.astype(np.float32)
    img_numpy = img_numpy / value_dict[img_dtype][1]
    image_tensor = torch.from_numpy(img_numpy)
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    else:
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    return image_tensor


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        label_file_path = 'voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train or self.gen_attn :
            image_name = os.path.join(self.coco_root, 'train2014', name + '.jpg')
            # img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            image_name = os.path.join(self.coco_root, 'val2014', name + '.jpg')
            # img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        img = imageio.imread(image_name)
        if img.ndim == 3:
            img = img[:, :, :3]
        img = numpy_to_tensor(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
            img = img.squeeze(0)

        return img, label

    def __len__(self):
        return len(self.img_name_list)

class COCOClsDatasetMS(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, scales, train=True, transform=None, gen_attn=False, unit=1):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        if gen_attn:
            img_name_list_path = os.path.join(img_name_list_path, 'train_part_id.txt')
        else:
            img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales
        self.gen_attn = gen_attn
        self.gt_dir = Path(coco_root) / "gt"

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train or self.gen_attn:
            image_name = os.path.join(self.coco_root, 'train2014', name + '.jpg')
            # img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            image_name = os.path.join(self.coco_root, 'val2014', name + '.jpg')
            # img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        img = imageio.imread(image_name)
        if img.ndim == 3:
            img = img[:, :, :3]
        # rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))
        rounded_size = (int(round(img.shape[0] / self.unit) * self.unit), int(round(img.shape[1] / self.unit) * self.unit))
        image_tensor = numpy_to_tensor(img)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            # s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            s_img_resize = K.Resize(target_size, 'BICUBIC')
            s_img = s_img_resize(image_tensor)
            s_img = s_img.squeeze(0)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i]).squeeze(0)

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12Dataset(Dataset):
    def __init__(self, voc12_root, train=True, transform=None, gen_attn=False, img_ch=3):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        img_name_list_path = os.path.join(voc12_root, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path=os.path.join(voc12_root, 'cls_labels.npy'))
        # self.label_list = load_image_label_list_from_npy(self.img_name_list)
        data_root = Path(voc12_root) / "voc12" if "voc12" not in voc12_root else voc12_root
        self.voc12_root = Path(data_root) / "VOCdevkit" / "VOC2012"
        # self.voc12_root = voc12_root
        self.transform = transform
        self.img_ch = img_ch

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        read_img_name = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        # img = PIL.Image.open(read_img_name).convert("RGB")
        img = imageio.imread(read_img_name)
        img = img[:, :, :3]
        img = numpy_to_tensor(img)
        label = torch.from_numpy(self.label_list[idx])
        if self.img_ch == 4:
            slope_path = os.path.join(self.voc12_root, 'SlopeImages', name + '.png')
            slope = imageio.imread(slope_path)
            slope = numpy_to_tensor(slope)
            img = torch.cat((img, slope), dim=0)
            pass
        if self.transform:
            img = self.transform(img)
            img = img.squeeze(0)

        return img, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales, train=True, transform=None, gen_attn=False, unit=1, is_cluster=False, img_ch=3):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        # img_name_list_path = os.path.join(voc12_root, f'{"train" if train or gen_attn else "train_aug"}_id.txt')
        img_name_list_path = os.path.join(voc12_root, img_name_list_path)
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path=os.path.join(voc12_root, 'cls_labels.npy'))
        # self.label_list = load_image_label_list_from_npy(self.img_name_list)
        data_root = Path(voc12_root) / "voc12" if "voc12" not in voc12_root else voc12_root
        self.voc12_root = Path(data_root) / "VOCdevkit" / "VOC2012"
        # self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales
        self.gt_dir = self.voc12_root / "SegmentationClassAug"
        self.img_ch = img_ch

    def __getitem__(self, idx, return_name=False):
        name = self.img_name_list[idx]
        read_img_name = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        img = PIL.Image.open(read_img_name).convert("RGB")
        img = imageio.imread(read_img_name)
        img = img[:, :, :3]
        label = torch.from_numpy(self.label_list[idx])

        # rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))
        rounded_size = (int(round(img.shape[0] / self.unit) * self.unit), int(round(img.shape[1] / self.unit) * self.unit))
        image_tensor = numpy_to_tensor(img)
        if self.img_ch == 4:
            slope_path = os.path.join(self.voc12_root, 'SlopeImages', name + '.png')
            slope = imageio.imread(slope_path)
            slope = numpy_to_tensor(slope)
            image_tensor = torch.cat((image_tensor, slope), dim=0)

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            # s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            s_img_resize = K.Resize(target_size, 'BICUBIC')
            s_img = s_img_resize(image_tensor)
            s_img = s_img.squeeze(0)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])
                ms_img_list[i] = ms_img_list[i].squeeze(0)

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        if return_name:
            return msf_img_list, label, name
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


def build_dataset(is_train, data_set, args, gen_attn=False):
    # transform = build_transform(is_train, args, gen_attn)
    transform = build_transform_kornia(is_train, args, gen_attn)
    dataset = None
    nb_classes = None

    if data_set == 'VOC12':
        dataset = VOC12Dataset(voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif data_set == 'VOC12cluster':
        dataset = VOC12DatasetMS(img_name_list_path=args.cluster_data_list, voc12_root=args.data_path,
                                 scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform, is_cluster=True)
        nb_classes = 20
    elif data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_list, coco_root=args.data_path, scales=tuple(args.scales), label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif data_set == 'landslide':
        dataset = VOC12Dataset(voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, img_ch=args.image_ch)
        nb_classes = 1
    elif data_set == 'landslideMS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform, img_ch=args.image_ch)
        nb_classes = 1
    elif data_set == 'landslidecluster':
        dataset = VOC12DatasetMS(img_name_list_path=args.cluster_data_list, voc12_root=args.data_path,
                                 scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform, is_cluster=True, img_ch=args.image_ch)
        nb_classes = 1
    elif data_set == 'postdam':
        dataset = VOC12Dataset(voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, img_ch=args.image_ch)
        nb_classes = 5
    elif data_set == 'postdamMS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform, img_ch=args.image_ch)
        nb_classes = 5
    elif data_set == 'postdamcluster':
        dataset = VOC12DatasetMS(img_name_list_path=args.cluster_data_list, voc12_root=args.data_path,
                                 scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform, is_cluster=True, img_ch=args.image_ch)
        nb_classes = 5
    elif data_set == 'cityscapes':
        dataset = VOC12Dataset(voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, img_ch=args.image_ch)
        nb_classes = 19
    elif data_set == 'cityscapesMS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform, img_ch=args.image_ch)
        nb_classes = 19
    elif data_set == 'cityscapescluster':
        dataset = VOC12DatasetMS(img_name_list_path=args.cluster_data_list, voc12_root=args.data_path,
                                 scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform, is_cluster=True)
        nb_classes = 19

    return dataset, nb_classes


def build_transform(is_train, args, gen_attn):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not gen_attn:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def create_train_transform(resize_im, args):
    transform_step_not_resize_im = K.RandomCrop((args.input_size, args.input_size), padding=4)
    
    transform_step_0 = K.RandomResizedCrop(
        size=(args.input_size, args.input_size),
        ratio=(0.75, 1.3333),
        resample=kornia.constants.Resample.BICUBIC
    )
    transform_step_1 = K.RandomHorizontalFlip(p=0.5)
    transform_step_2 = K.ColorJitter(args.color_jitter, p=0.5)
    transform_step_3 = K.Normalize(
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD)
    if args.image_ch == 4:
        transform_step_3 = K.Normalize(mean=[0.485, 0.456, 0.406, 0], 
                                       std=[0.229, 0.224, 0.225, 1])
    transform_step_4 = K.RandomErasing(p=args.reprob)
    if not resize_im:
        return K.AugmentationSequential(transform_step_not_resize_im, transform_step_1, transform_step_2, transform_step_3)
    else:
        return K.AugmentationSequential(transform_step_0, transform_step_1, transform_step_2, transform_step_3)

def create_val_transform(resize_im, gen_attn, args):
    if resize_im and not gen_attn:
        resize_size = int((256 / 224) * args.input_size)
        transform_step_0 = K.Resize((resize_size, resize_size))
        transform_step_1 = K.CenterCrop((args.input_size, args.input_size))
        transform_step_2 = K.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD)
        if args.image_ch == 4:
            transform_step_2 = K.Normalize(mean=[0.485, 0.456, 0.406, 0], 
                                           std=[0.229, 0.224, 0.225, 1])
        return K.AugmentationSequential(transform_step_0, transform_step_1, transform_step_2)
    else:
        transform_step_0 = K.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD)
        if args.image_ch == 4:
            transform_step_0 = K.Normalize(mean=[0.485, 0.456, 0.406, 0], 
                                           std=[0.229, 0.224, 0.225, 1])
        return transform_step_0

def build_transform_kornia(is_train, args, gen_attn):
    resize_im = args.input_size > 32
    if is_train:
        return create_train_transform(resize_im, args)
    else:
        return create_val_transform(resize_im, gen_attn, args)