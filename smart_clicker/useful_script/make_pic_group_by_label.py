from smart_clicker import config
import shutil

import json
frame_data_path = config.PROJECT_ROOT_PATH / "frame_data"
cleaned_frame_data_Path = config.PROJECT_ROOT_PATH / "cleaned_frame_data"

train_data_path = config.PROJECT_ROOT_PATH / "train_data"
val_data_path =config.PROJECT_ROOT_PATH / "val_data"
label_data_path = config.PROJECT_ROOT_PATH / "label_data.json"

with open(label_data_path, 'r') as label_data_f:
    label_data = json.load(label_data_f)


for pic_name, label in label_data.items():
    pic_path = cleaned_frame_data_Path / pic_name
    target_dir = frame_data_path / str(label)
    target_dir.mkdir(exist_ok=True)
    shutil.move(pic_path, target_dir/ pic_name)