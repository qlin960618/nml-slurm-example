import os
import shutil
import glob
from tqdm import tqdm


def run(extracted_frames_path, json_and_raw_path, up_to_id=None, dry_run=True):
    ids = glob.glob1(extracted_frames_path, "*.png")
    if up_to_id is None:
        ids_to_move = ids
    else:
        ids_partial = [int(id.replace(".png", "")) for id in ids]
        # pass only id <= up_to_id
        ids_to_move = [id_ for id_ in ids_partial if id_ <= up_to_id]
        if len(ids_to_move) == 0:
            raise RuntimeError("no files to move")
        ids_to_move = [str(id_) + ".png" for id_ in ids_to_move]

    extracted_file_paths = [os.path.join(extracted_frames_path, img_id) for img_id in ids_to_move]

    if not os.path.exists(json_and_raw_path):
        os.makedirs(json_and_raw_path, exist_ok=True)

    # image_number = len(os.listdir(parameters.json_and_raw_path))

    for i, id in enumerate(tqdm(ids_to_move)):
        # also include renumbering
        if not dry_run:
            shutil.copyfile(extracted_file_paths[i], os.path.join(json_and_raw_path, str(i + 1) + ".png"))
            if os.path.isfile(extracted_file_paths[i].replace(".png", ".json")):
                shutil.copyfile(extracted_file_paths[i].replace(".png", ".json"),
                                os.path.join(json_and_raw_path, str(i + 1) + ".json"))
        else:
            print("copying: ", extracted_file_paths[i], " to ", os.path.join(json_and_raw_path, str(i + 1) + ".png"))
            if os.path.isfile(extracted_file_paths[i].replace(".png", ".json")):
                print("copying: ", extracted_file_paths[i].replace(".png", ".json"), " to ",
                      os.path.join(json_and_raw_path, str(i + 1) + ".json"))



