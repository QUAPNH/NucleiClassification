import csv
import glob
import os
import re
from . import pre_proc, transforms, affine_funcs
from skimage import measure
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        data_dir,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
        phase=None,
        input_h=256,
        input_w=256,
        max_objs=500,
        down_ratio=1,
    ):
        #print(input_shape)
        #print(mask_shape)
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        self.img_dir = os.path.join(data_dir[0], phase, 'Images')
        self.mask_dir = os.path.join(data_dir[0], phase, 'Masks')
        self.img_ids = sorted(os.listdir(self.img_dir))
        self.phase=phase
        self.down_ratio=down_ratio
        self.input_h=input_h
        self.input_w=input_w
        self.max_objs=max_objs
        self.num_classes = 1
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def load_gt_bboxes(self,annopath):
        bboxes = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*.png"))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([x1, y1, x2, y2])
        return np.asarray(bboxes, np.float32)

    def load_gt_masks(self,annopath):
        masks = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*.png"))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                masks.append(np.where(mask > 0, 1., 0.))
        return np.asarray(masks, np.float32)

    def load_gt_masks_bboxes(self,annopath):
        bboxes = []
        masks =FileLoader.load_gt_masks(annopath)
        for mask in masks:
            r, c = np.where(mask > 0.)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([x1, y1, x2, y2])
        return np.asarray(masks, np.float32), np.asarray(bboxes, np.float32)

    def find_maximum_mask(self,mask):
        mask = np.where(mask>0., 1., 0.)
        labels = measure.label(mask, connectivity=1)
        props = measure.regionprops(labels)
        props = sorted(props, key=lambda x: x.area, reverse=True)  # descending order
        if len(props)==0:
            return None
        else:
            return np.asarray(np.where(props[0].label == labels, 1., 0.), np.float32)


    def sample_ROI(self, mask, rbox):
        ROI = affine_funcs.sample_masks(mask, rbox, numpy=True)
        return ROI


    def masks_to_bboxes_rois(self, gt_masks):
        out_bboxes = []
        out_rois = []
        for mask in gt_masks:
            mask = self.find_maximum_mask(mask)
            if mask is None:
                continue
            r, c = np.where(mask==1.)
            h, w = mask.shape
            y1 = np.maximum(0, np.min(r))
            x1 = np.maximum(0, np.min(c))
            y2 = np.minimum(h-1, np.max(r))
            x2 = np.minimum(w-1, np.max(c))
            if y2-y1>2 and x2-x1>2:
                rbox = np.asarray([float(x1+x2)/2, float(y1+y2)/2,
                                   float(x2-x1+1), float(y2-y1+1)], np.float32)  #[cenx, ceny, w, h]
                rbox_padding = rbox.copy()
                rbox_padding[2:] *= 1.1
                roi = self.sample_ROI(mask, rbox_padding)
                if roi.shape[0]>1 and roi.shape[1]>1:
                    out_bboxes.append(rbox)
                    out_rois.append(roi)
        return np.asarray(out_bboxes, np.float32), out_rois

    def load_annoFolder(self, img_id):
        return os.path.join(self.mask_dir, img_id[:-4])


    def load_annotation(self, img_id):
        annoFolder = self.load_annoFolder(img_id)
        masks = self.load_gt_masks(annoFolder)
        return masks

    def image_trans(self, image):
        image = cv2.resize(image, (self.input_w, self.input_h))
        image = image.astype(np.float32) / 255.
        image = image - 0.5
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image)


    def load_image(self, img_id):
        imgFile = os.path.join(self.img_dir, img_id)
        img = cv2.imread(imgFile)
        return img


    def data_preparation(self, image, gt_masks, augment=False):
        return np.asarray(image, np.float32), np.asarray(gt_masks, np.float32)


    def get_OJ(self, image, gt_masks):
        # img_id = self.img_ids[index]     #img_id=name
        # image = self.load_image(img_id)
        # if self.phase == 'test':
        #     return {'image': self.image_trans(image),
        #             'img_id': img_id}
        # else:
        #     gt_masks = self.load_annotation(img_id)   # num_obj x h x w

        image, gt_masks = self.data_preparation(image, gt_masks, augment=False)

        gt_bboxes, gt_rois = self.masks_to_bboxes_rois(gt_masks)


        gt_bboxes2 = gt_bboxes.copy()
        gt_bboxes2 /= self.down_ratio

        data_dict = pre_proc.generate_ground_truth(gt_bboxes2=gt_bboxes2,
                                                   num_classes=self.num_classes,
                                                   image_h=self.input_h//self.down_ratio,
                                                   image_w=self.input_w//self.down_ratio,
                                                   max_objs=self.max_objs)
        # for name in data_dict:
        #     data_dict[name] = torch.from_numpy(data_dict[name])
        #data_dict['image'] = image
        #data_dict['gt_bboxes'] = gt_bboxes
        #data_dict['gt_rois'] = gt_rois

        return data_dict



    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # get det_label
        img_id = self.img_ids[idx]
        image = self.load_image(img_id)
        gt_masks = self.load_annotation(img_id)
        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)
            image = shape_augs.augment_image(image)
            gt_masks_aug = []
            if gt_masks.size != 0:
                gt_masks = gt_masks.transpose(1, 2, 0)
                for i in range(gt_masks.shape[-1]): 
                    gt_masks_iter = shape_augs.augment_image(gt_masks[...,i])
                    gt_masks_aug.append(np.ascontiguousarray(gt_masks_iter))
                gt_masks_aug = np.asarray(gt_masks_aug, np.float32)
                #gt_masks = np.ascontiguousarray(gt_masks)
            else:
                gt_masks_aug = gt_masks            
        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)
        det_label = self.get_OJ(image, gt_masks_aug)
        for name in det_label:
            det_label[name] = torch.from_numpy(det_label[name])
        feed_dict.update(det_label)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
