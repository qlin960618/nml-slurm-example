import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
        Args:
            images_dir (str): parameters to images folder
            masks_dir (str): parameters to segmentation masks folder
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline
                (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing
                (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = None

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.CLASSES = classes+['unlabelled']
        self.ids = os.listdir(images_dir)
        self.data_num = len(self.ids)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_list = list(range(self.data_num))
        # self.images_fps = [os.parameters.join(images_dir, image_id) for image_id in self.ids]
        # print(self.images_fps)
        # self.masks_fps = [os.parameters.join(masks_dir, image_id) for image_id in self.ids]
        # print(self.masks_fps)

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower())*50 for cls in classes]
        # # self.class_values1 = [0, 150]
        # # self.class_values2 = [100, 150]
        # # self.class_values3 = [50, 150]
        # self.class_values1 = [0]
        # self.class_values2 = [100]
        # self.class_values3 = [50]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_dir+'/'+str(i+9000)+'.png')
        hr = int(512)
        wr = int(512)
        image = cv2.resize(image, (wr, hr))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)
        # mask1 = cv2.imread(self.masks_dir+'/'+str(i)+'-1.png', 0)
        # mask2 = cv2.imread(self.masks_dir+'/' + str(i) + '-2.png', 0)
        # mask3 = cv2.imread(self.masks_dir + '/' + str(i) + '-3.png', 0)
        mask = cv2.imread(self.masks_dir + '/' + str(i+9000) + '.png', 0)
        # print(mask)

        # extract certain classes from mask (e.g. cars)
        # masks1 = [(mask1 == v) for v in self.class_values1]
        # masks2 = [(mask2 == v) for v in self.class_values2]
        # masks3 = [(mask3 == v) for v in self.class_values3]
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask = cv2.resize(mask, (wr, hr))
        # mask = mask[256:768, 256:768]
        # print(mask.shape)
        # print(mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            # print(sample)
            image, mask = sample['image'], sample['mask']
            # print(image.shape)
            # print(image)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # print(image.shape)
            # print(image)

        return image, mask

    def __len__(self):
        return len(self.ids)

