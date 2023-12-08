import glob
import cv2
import os
import numpy as np
import json
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm


def run(
        json_and_raw_path=None,
        make_debug_img=None,
        debug_img_path=None,
        raw_resized_path=None,
        mask_resized_path=None,
        network_img_size=None,
        make_binary_img_in=None,
        mask_sigma=None,
        binary_threshold=None,
        classes=None,
):
    assert classes is not None, "classes must be specified"
    if make_debug_img:
        os.makedirs(debug_img_path, exist_ok=True)

    os.makedirs(raw_resized_path, exist_ok=True)

    os.makedirs(mask_resized_path, exist_ok=True)

    image_list = glob.glob1(json_and_raw_path, "*.png")

    print("Starting parallel image processing: ")
    # pool = Pool(processes=30)
    with Pool(processes=30) as pool:
        print("opened pool")
        with tqdm(total=len(image_list)) as progress_bar:
            path_list = {
                "json_and_raw_path": json_and_raw_path,
                "mask_resized_path": mask_resized_path,
                "debug_img_path": debug_img_path,
                "raw_resized_path": raw_resized_path,
            }
            binary_img_param = {
                "make_binary_img_in": make_binary_img_in,
                "binary_threshold": binary_threshold,
            }
            for result in pool.imap(
                    partial(
                        process_img_task,
                        path_list=path_list,
                        network_img_size=network_img_size, mask_sigma=mask_sigma,
                        binary_img_param=binary_img_param, make_debug_img=make_debug_img,
                        classes=classes,
                    ),
                    image_list):
                progress_bar.update(1)
        pool.close()
        pool.join()

    print("Done image processing")


def process_img_task(img_name, path_list,
                     network_img_size, mask_sigma,
                     binary_img_param, make_debug_img,
                     classes):
    """
    path_list={
        "json_and_raw_path":
        "mask_resized_path":
        "raw_resized_path":
        "debug_img_path":
    }
    binary_img_param={
        "make_binary_img_in":
        "binary_threshold"
    }
    """
    num_str = img_name.replace(".png", "")
    k = int(num_str)
    resized_w, resized_h = tuple(network_img_size)

    #######################
    # define save path
    input_img_path = os.path.join(path_list["json_and_raw_path"], str(k) + '.png')
    output_original_img_path = os.path.join(path_list["raw_resized_path"], str(k) + '-original.png')
    output_resized_img_path = os.path.join(path_list["raw_resized_path"], str(k) + '.png')
    # add mask id
    output_heatmap_path_prefix = os.path.join(path_list["mask_resized_path"], str(k) + '-')
    output_debug_img_path_prefix = os.path.join(path_list["debug_img_path"], str(k) + '-')

    # print("processing:", k)
    if binary_img_param["make_binary_img_in"]:
        image = cv2.imread(input_img_path, 0)
        image_original = cv2.imread(input_img_path)
        image_original_resized = cv2.resize(image_original, (resized_w, resized_h))
        cv2.imwrite(output_original_img_path, image_original_resized)
    else:
        image = cv2.imread(input_img_path)
    image_resized = cv2.resize(image, (resized_w, resized_h))
    debug_image = image_resized.copy()

    if binary_img_param["make_binary_img_in"]:
        ret, image_resized = cv2.threshold(image_resized, binary_img_param["binary_threshold"],
                                           255, cv2.THRESH_BINARY)

    """
    ####### Read and process Json file #######
    """

    input_json_path = os.path.join(path_list["json_and_raw_path"], str(k) + '.json')

    label_w, label_h, pos_data = get_meta_data_from_json(input_json_path, classes)
    h_, w_, _ = image.shape
    # check if json file is read correctly
    if label_h is not None and label_w is not None:
        assert h_ == label_h and w_ == label_w, "Expect dimension match"
    else:
        label_h = h_
        label_w = w_

    #######################
    # resizing image
    # print("resized: " + str(int(h)) + "×" + str(int(h)) + " -> " + str(network_img_size))

    for i, class_id in enumerate(classes):
        num_pos, pos_set = pos_data[class_id]

        cv2.imwrite(output_resized_img_path, image_resized)

        debug_image_copy = debug_image.copy()
        #######################
        # point does not exist
        if num_pos == 0:
            image_heatmap = make_empty_confidence_map(resized_w, resized_h)

        else:
            #######################
            # recalculate the tip position after resizing
            if num_pos > 1:
                raise ValueError("need to implement method of multi point per class")

            x_tip, y_tip = pos_set[0][0], pos_set[0][1]

            x_tip = int(float(x_tip) / label_w * resized_w)
            y_tip = int(float(y_tip) / label_h * resized_h)

            tip = np.array([x_tip, y_tip])

            if make_debug_img:
                cv2.circle(debug_image_copy, (int(x_tip), int(y_tip)), 10, (255, 0, 0), -1)

            image_heatmap = get_gaussian_confidence_map(resized_w, resized_h,
                                                        x_tip, y_tip,
                                                        mask_sigma)

        # save image
        cv2.imwrite(output_heatmap_path_prefix + str(i) + ".png", image_heatmap)
        if make_debug_img:
            cv2.imwrite(output_debug_img_path_prefix + str(i) + '-debug.png', debug_image_copy)


def make_empty_confidence_map(w, h):
    return np.zeros([h, w])


def get_gaussian_confidence_map(w, h, tip_x, tip_y, sigma):
    width = np.arange(0, w, 1)
    height = np.arange(0, h, 1)
    X, Y = np.meshgrid(width, height)
    mu = np.array([tip_x, tip_y])
    S = np.array([[sigma, 0], [0, sigma]])

    x_norm = (np.array([X, Y]) - mu[:, None, None]).transpose(1, 2, 0)
    heatmap = np.exp(- x_norm[:, :, None, :] @ np.linalg.inv(S)[None, None, :, :] @ x_norm[:, :, :, None] / 2.0) * 255

    return heatmap[:, :, 0, 0]


def get_meta_data_from_json(json_path, classes):
    if not os.path.isfile(json_path):
        return None, None, {cls: (0, []) for cls in classes}
    with open(json_path, "r", encoding="utf-8") as f:
        dj = json.load(f)
    w = dj['imageWidth']
    h = dj['imageHeight']

    out_data = {}
    for class_name in classes:
        out_data[class_name] = (0, [])
        for shape_ in dj['shapes']:
            if shape_['label'] == class_name:
                points = shape_['points']
                out_data[class_name] = (len(points), points)
                break

    return w, h, out_data
