import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np


def run(
        frame_interval=1,
        video_list=None,
        video_path=None,
        output_frames_path=None,
        starting_number=None,
        overwriting=False,
        rotate_by_90=False,
        dry_run=False,
):
    print("video list:", video_list)
    video_full_path_list = [os.path.join(video_path, video_f) for video_f in video_list]

    total_num_image = 0
    num_image_list = []
    video_frame_to_extract = {}
    # Open Video
    print('Opening video...')
    for video_f in video_full_path_list:
        cap = cv2.VideoCapture(video_f)
        if cap.isOpened():
            video_frame_to_extract[video_f] = list(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_interval))
            total_num_image = total_num_image + len(video_frame_to_extract[video_f])
            num_image_list.append(len(video_frame_to_extract[video_f]))
        else:
            print("Video is not opened")

        cap.release()

    print('Total output image number is ' + str(total_num_image) + ' images.')

    if not os.path.isdir(output_frames_path):
        print("output_frames_path does not exist, creating...")
        if not dry_run:
            os.makedirs(output_frames_path, exist_ok=True)

    total_num_image = 0
    frame_counter = 0
    for video_f, frame_id_list in video_frame_to_extract.items():
        cap = cv2.VideoCapture(video_f)
        if cap.isOpened():
            print("Total Frame number is " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            print("Video is not opened")

        # Skip useless frames
        for i, frame_id in tqdm(enumerate(frame_id_list)):
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
                frame_counter += 1
                # OUTPUT IMAGE ABSOLUTE OR RELATIVE PATH
                output_image_path = os.path.join(output_frames_path,
                                                 str(frame_counter + starting_number - 1) + '.png')
                if not overwriting and os.path.isfile(output_image_path):
                    raise RuntimeError("Old frame will be overritten")
                if rotate_by_90:
                    frame = np.transpose(frame, (1, 0, 2))
                if not dry_run:
                    cv2.imwrite(output_image_path, frame)
                else:
                    cv2.imshow("show_window", frame)
                    cv2.waitKey(1)

        print(video_f + ' extraction is completed!')
        # Release video
        cap.release()
