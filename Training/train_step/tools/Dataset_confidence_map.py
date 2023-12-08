import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    CLASSES = ['tip_instrument', 'tip_shadow', 'tip_another_instrument']

    def __init__(self, images_dir, masks_dir, image_size, max_id=None, classes=None, augmentation=None,
                 preprocessing=None):
        self.current = 0  # iterator implementation
        self.n_classes = len(classes)
        self.ids = sorted([int(img_pth.split(".")[0]) for img_pth in os.listdir(images_dir)])
        self.max_id = max_id
        self.data_num = len(self.ids)
        self.images_dir = images_dir
        self.image_size = image_size
        self.masks_dir = masks_dir

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __iter__(self):
        return self

    def __next__(self):
        id = self.current
        if id >= self.__len__():
            raise StopIteration
        self.current += 1
        return self.__getitem__(id)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError
        image = cv2.imread(os.path.join(self.images_dir, str(self.ids[i]) + '.png'))
        hr = int(self.image_size[1])
        wr = int(self.image_size[0])
        image = cv2.resize(image, (wr, hr))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = np.zeros([hr, wr, self.n_classes])
        for c_id in range(self.n_classes):
            masks[:, :, c_id] = cv2.imread(os.path.join(self.masks_dir,
                                                        str(self.ids[i]) + '-' + str(c_id) + '.png'), 0) / 255
        # extract certain classes from mask (e.g. cars)
        # mask = cv2.resize(masks, (wr, hr))
        # masks = np.expand_dims(masks, axis=2)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=masks)
            image, masks = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=masks)
            image, masks = sample['image'], sample['mask']

        return image, masks

    def __len__(self):
        if self.max_id is not None:
            return self.max_id
        return len(self.ids)
