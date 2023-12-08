import os
import albumentations as albu


def get_augmentation_preset():
    train_transform = [
        albu.Blur(blur_limit=21, p=1),
        albu.MotionBlur(blur_limit=21),
        albu.GaussianBlur(blur_limit=21),
        albu.GlassBlur(),
        # albu.GaussNoise(p=0.2),
        # albu.ImageCompression(),
        albu.ISONoise(),
        # albu.MultiplicativeNoise(),
        albu.Downscale(),
        albu.Rotate(),
        # albu.OpticalDistortion(),
        # albu.GridDistortion(),
        # albu.ElasticTransform(),
        # albu.RandomGridShuffle(),
        # albu.HueSaturationValue(),
        albu.RGBShift(),
        # albu.ChannelDropout(),
        # albu.Normalize(),
        albu.RandomGamma(),
        albu.RandomBrightnessContrast(),
        # albu.RandomContrast(),
        albu.Compose(
            [
                albu.Rotate(),
                albu.OneOf(
                    [
                        albu.Blur(blur_limit=21, p=1),
                        albu.MotionBlur(blur_limit=21),
                        albu.GaussianBlur(blur_limit=21),
                        # albu.GlassBlur(),
                        # albu.GaussNoise(p=0.2),
                        # albu.ImageCompression(),
                        albu.ISONoise(),
                        # albu.MultiplicativeNoise(),
                        albu.Downscale(),
                        # albu.OpticalDistortion(),
                        # albu.GridDistortion(),
                        # albu.ElasticTransform(),
                        # albu.RandomGridShuffle(),
                        # albu.HueSaturationValue(),
                        albu.RGBShift(),
                        # albu.ChannelDropout(),
                        # albu.Normalize(),
                        albu.RandomGamma(),
                        # albu.RandomBrightness(),
                        # albu.RandomContrast(),
                        albu.RandomBrightnessContrast(),
                    ]
                )
            ]
        ),
    ]
    return albu.OneOf(train_transform)


class TrainingParameters:

    def __init__(self, **kwargs):
        if 'dir_base_path' not in kwargs:
            raise RuntimeError("(dir_base_path) Base path to the dataset folder is not provided")
        if 'model' not in kwargs:
            self.model = 'JaxaPartialPoseEst_model'
        if 'original_video_quality' not in kwargs:
            self.original_video_quality = '720p'

        for key, value in kwargs.items():
            # print("Override", key, "to", value)
            setattr(self, key, value)

        self.image_quality = '720p_576_ver1'
        self.network_img_size = [576, 576]


        # Path

        # 0_extract_frames
        self.frame_interval = 20

        # 1_rename
        # start_num = 0
        # total_image = 973

        # 2_make_labeled_image
        self.debug = True
        self.binary = False
        self.binary_threshold = 100
        self.mask_sigma = 400  # 800  #1500

        # 3_augmentation
        self.image_num_after_augmentation = 20000
        self.original_num_for_train = 800
        self.aug_num_for_train = self.image_num_after_augmentation - \
                                 self.original_num_for_train - 1000  # final number is for aug_val, test
        self.augmentation_preset = get_augmentation_preset()

    def re_process_path(self):
        dir_base_path = self.dir_base_path
        image_quality = self.image_quality
        model = self.model

        self.video_path = os.path.join(dir_base_path, "traning_source1")
        self.extracted_frames_path = os.path.join(dir_base_path, "workspace/extracted_frames/")

        self.json_and_raw_path = os.path.join(dir_base_path, "workspace", image_quality, model, "json_and_raw")
        self.debug_path = os.path.join(dir_base_path, "workspace", image_quality, model, "debug")
        self.raw_resized_path = os.path.join(dir_base_path, "workspace", image_quality, model, "raw_resized")
        self.mask_resized_path = os.path.join(dir_base_path, "workspace", image_quality, model, "mask_resized")
        self.raw_augmented_path = os.path.join(dir_base_path, "workspace", image_quality, model, "raw_augmented")
        self.mask_augmented_path = os.path.join(dir_base_path, "workspace", image_quality, model, "mask_augmented")

        self.dataset_path = os.path.join(dir_base_path, "dataset", image_quality, model)
        self.dataset_train_path = os.path.join(dir_base_path, "dataset", image_quality, model, "train")
        self.dataset_test_path = os.path.join(dir_base_path, "dataset", image_quality, model, "test")
        self.dataset_val_path = os.path.join(dir_base_path, "dataset", image_quality, model, "val")
        self.dataset_train_mask_path = os.path.join(dir_base_path, "dataset", image_quality, model, "train_mask")
        self.dataset_test_mask_path = os.path.join(dir_base_path, "dataset", image_quality, model, "test_mask")
        self.dataset_val_mask_path = os.path.join(dir_base_path, "dataset", image_quality, model, "val_mask")

        self.result_path = os.path.join(dir_base_path, "result", image_quality, model)
        self.log_path = os.path.join(dir_base_path, "result", image_quality, model, "log")
