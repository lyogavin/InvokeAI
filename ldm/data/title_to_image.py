import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


import random




class TitleToImageDataset(Dataset):
    def __init__(
        self,
        df_path,
        size=None,
        interpolation='bicubic',
        flip_p=0.5,
        set='train',
        center_crop=False,
    ):

        self.df_path = df_path

        self.df = pd.read_csv(self.df_path)
        if set == 'val':
            self.df = self.df.sample(10)

        self.num_images = len(self.df)
        self._length = self.num_images


        self.center_crop = center_crop


        self.size = size
        self.interpolation = {
            'linear': PIL.Image.LINEAR,
            'bilinear': PIL.Image.BILINEAR,
            'bicubic': PIL.Image.BICUBIC,
            'lanczos': PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.df.iloc[i]['image'])

        if not image.mode == 'RGB':
            image = image.convert('RGB')


        example['caption'] = self.df.iloc[i]['text']

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2,
                (w - crop) // 2 : (w + crop) // 2,
            ]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize(
                (self.size, self.size), resample=self.interpolation
            )

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example['image'] = (image / 127.5 - 1.0).astype(np.float32)
        return example
