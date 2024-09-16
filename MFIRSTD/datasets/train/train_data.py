"""
Dataloader with train data.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging

from MFIRSTD.datasets import path_config as dataset_path_config
import MFIRSTD.datasets.transforms as T

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, train_size=300):
        self.num_frames = num_frames
        self.split = 'train'
        self._transforms = make_train_transform(train_size=train_size)

        self.train_seqs_file = dataset_path_config.train_seqs_file
        self.img_path = dataset_path_config.img_path
        self.mask_path = dataset_path_config.mask_path
        self.frames_info = {
            'dataset': {}
        }
        self.img_ids = []
        logger.debug('loading dataset train seqs...')
        with open(self.train_seqs_file, 'r') as f:
            video_names = f.readlines()
            video_names = [name.strip() for name in video_names]
            logger.debug('dataset-train num of videos: {}'.format(len(video_names)))
            for video_name in video_names:
                frames = sorted(glob.glob(os.path.join(self.mask_path, video_name, '*.png')))
                self.frames_info['dataset'][video_name] = [frame_path.split('\\')[-1][:-4] for frame_path in frames]
                self.img_ids.extend([('dataset', video_name, frame_index) for frame_index in range(len(frames))])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]
        frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - math.floor(float(self.num_frames) / 2),
                                                                frame_index + math.ceil(float(self.num_frames) / 2), 1)]
        assert len(frame_indices) == self.num_frames
        frame_ids = []
        img = []
        masks = []
        mask_paths = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)

            img_path = os.path.join(self.img_path, video_name, frame_name + '.png')
            gt_path = os.path.join(self.mask_path, video_name, frame_name + '.png')
            # import ipdb;ipdb.set_trace()
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            # mask_paths.append(gt_path)
        # import ipdb;ipdb.set_trace()
        masks = torch.cat(masks, dim=0)
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths}
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return torch.cat(img, dim=0), target


def make_train_transform(train_size=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResize(scales, max_size=800),
        T.PhotometricDistort(),
        T.Compose([
            T.RandomResize([500, 600, 700]),
            T.RandomSizeCrop(473, 750),
            T.RandomResize([train_size], max_size=int(1.8 * train_size)),
        ]),
        normalize,
    ])