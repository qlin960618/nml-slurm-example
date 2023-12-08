import cv2

import os
import glob
import labelme


class Param:
    pos = None


class Main:
    def __init__(self, extracted_frame_dir, label_name, test_only=True):
        self.extracted_frame_dir = extracted_frame_dir
        self.label_name = label_name
        self.test_only = test_only
        self.param = Param()

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("x: ", x, "y: ", y)
            self.param.pos = [x, y]
            # cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # cv2.imshow("image", img)

    def run(self):
        frame_img_name = glob.glob1(self.extracted_frame_dir, "*.png")
        frame_json_name = glob.glob1(self.extracted_frame_dir, "*.json")
        frame_imd_ids = [os.path.splitext(os.path.basename(img_name))[0] for img_name in frame_img_name]
        frame_json_ids = [os.path.splitext(os.path.basename(json_name))[0] for json_name in frame_json_name]

        unlabeled_img_ids = list(set(frame_imd_ids) - set(frame_json_ids))
        print("unlabeled_img_ids: ", unlabeled_img_ids)
        print("unlabeled_img_ids number: ", len(unlabeled_img_ids))

        for i, id in enumerate(unlabeled_img_ids):
            img = cv2.imread(os.path.join(self.extracted_frame_dir, id + ".png"))
            cv2.imshow("image", img)
            # detect mouse click position
            self.param.pos = None
            # open file
            if os.path.isfile(os.path.join(self.extracted_frame_dir, id + ".json")):
                labelme_obj = labelme.LabelFile(filename=os.path.join(self.extracted_frame_dir, id + ".json"))
            else:
                labelme_obj = labelme.LabelFile()

            # check if label already exist
            for shape in labelme_obj.shapes:
                if shape['label'] == self.label_name:
                    self.param.pos = [shape["points"][0]]

            while True:
                cv2.setMouseCallback("image", self.mouse_click)
                if self.param.pos is not None:
                    img_label = img.copy()
                    cv2.circle(img_label, (self.param.pos[0], self.param.pos[1]), 10, (0, 0, 255), -1)
                    cv2.imshow("image", img_label)
                key = cv2.waitKey(1)

                if key == ord("n"):
                    break
                elif key == ord("q"):
                    exit()
            # save json file
            labelme_obj.imagePath = id + ".png"
            img_data = labelme_obj.load_image_file(os.path.join(self.extracted_frame_dir, id + ".png"))

            if self.param.pos is not None:
                json_shape = labelme_obj.shapes + [{
                    "label": self.label_name,
                    "points": [[self.param.pos[0], self.param.pos[1]]],
                    "group_id": None,
                    "shape_type": "point",
                    "flags": {},
                    "description": '',
                }]
            else:
                json_shape = labelme_obj.shapes

            if self.test_only:
                print("test_only, saving to ", os.path.join(self.extracted_frame_dir, id + ".json"))
                print(labelme_obj)
            else:
                print("saving to ", os.path.join(self.extracted_frame_dir, id + ".json"))
                labelme_obj.save(
                    filename=os.path.join(self.extracted_frame_dir, id + ".json"),
                    shapes=json_shape,
                    imagePath=id + ".png",
                    imageWidth=img.shape[1],
                    imageHeight=img.shape[0],
                    imageData=img_data,
                )


if __name__ == '__main__':
    _extracted_frame_dir = "F:/Reseearch/JCCAS_advance/workspace/extracted_frames"
    _label_name = "center"
    main_h = Main(_extracted_frame_dir, _label_name, test_only=False)
    main_h.run()
