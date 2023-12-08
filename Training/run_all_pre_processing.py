import glob
import os
import warnings

# Disable warnings from the Albumentations package
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# from train_step import frame_extration
from train_step import re_locate_file
from train_step import process_image_set
from train_step import make_augmentation
from train_step import split_train_val_test

from config.training_param import TrainingParameters
from config.network_param import NetworkParameter

if __name__ == '__main__':
    nn_param_override = {
        "CLASSES": ['class_label1', 'class_label2']
    }
    train_param_override = {
        "model": 'xxx_model',
        #############################
        # Note 1: This need to point from the /storage directory as your user home directory may not have same access
        #         path from the home directory
        'dir_base_path': 'storage/user_data/[YOUR DATA DIRECTORY]'
        #############################
    }
    train_param = TrainingParameters(**train_param_override)
    nn_param = NetworkParameter(**nn_param_override)
    train_param.re_process_path()

    # frame_extration.run(
    #     frame_interval=13,
    #     video_path=train_param.video_path,
    #     # video_list=['real_micro_teleop.mp4', 'real_micro_traj.mp4'],
    #     # video_list=['adapt_lock_R1_0804_exp1_vid_original.mp4'],
    #     video_list=os.listdir(train_param.video_path),
    #     output_frames_path=train_param.extracted_frames_path,
    #     starting_number=0,
    #     overwriting=True,
    #     rotate_by_90=False,
    #     dry_run=False,
    # )

    re_locate_file.run(
        extracted_frames_path=train_param.extracted_frames_path,
        json_and_raw_path=train_param.json_and_raw_path,
        # up_to_id=97,
        dry_run=False,
    )
    #
    process_image_set.run(
        json_and_raw_path=train_param.json_and_raw_path,
        raw_resized_path=train_param.raw_resized_path,
        mask_resized_path=train_param.mask_resized_path,
        network_img_size=train_param.network_img_size,
        make_binary_img_in=train_param.binary,
        mask_sigma=train_param.mask_sigma,
        binary_threshold=train_param.binary_threshold,
        make_debug_img=False, debug_img_path=train_param.debug_path,
        classes=nn_param.CLASSES
    )
    #
    make_augmentation.run(
        json_and_raw_path=train_param.json_and_raw_path,
        raw_resized_path=train_param.raw_resized_path,
        raw_augmented_path=train_param.raw_augmented_path,
        mask_resized_path=train_param.mask_resized_path,
        mask_augmented_path=train_param.mask_augmented_path,
        image_num_after_augmentation=train_param.image_num_after_augmentation,
        augmentation_preset=train_param.augmentation_preset,
        n_classes=len(nn_param.CLASSES),
    )
    #
    split_train_val_test.run(
        # total_original_img=int(len(glob.glob1(train_param.raw_resized_path, "*.png"))),
        original_num_for_train=train_param.original_num_for_train,
        aug_num_for_train=train_param.aug_num_for_train,
        # image_num_after_augmentation=train_param.image_num_after_augmentation,
        raw_resized_path=train_param.raw_resized_path,
        mask_resized_path=train_param.mask_resized_path,
        raw_augmented_path=train_param.raw_augmented_path,
        mask_augmented_path=train_param.mask_augmented_path,
        dataset_train_path=train_param.dataset_train_path,
        dataset_test_path=train_param.dataset_test_path,
        dataset_val_path=train_param.dataset_val_path,
        dataset_train_mask_path=train_param.dataset_train_mask_path,
        dataset_test_mask_path=train_param.dataset_test_mask_path,
        dataset_val_mask_path=train_param.dataset_val_mask_path,
        val_test_ratio=0.9,
        num_classes=len(nn_param.CLASSES)
    )

