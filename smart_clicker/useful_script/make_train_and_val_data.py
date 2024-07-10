import shutil
from smart_clicker import config
import json
from loguru import logger
import random

cleaned_frame_data_Path = config.PROJECT_ROOT_PATH / "cleaned_frame_data"
label_data_path = cleaned_frame_data_Path / "label_data.json"

train_data_path = config.PROJECT_ROOT_PATH / "train_data"
val_data_path =config.PROJECT_ROOT_PATH / "val_data"
train_data_path.mkdir(exist_ok=True)
val_data_path.mkdir(exist_ok=True)


true_data_list = []
false_data_list = []

with open(label_data_path, 'r') as label_data_f:
    label_data = json.load(label_data_f)

for pic_name, label  in label_data.items():
    pic_path = cleaned_frame_data_Path / pic_name
    if label: 
        true_data_list.append(pic_path)
    else:
        false_data_list.append(pic_path)

# 移动val
val_data_list = []
for sample_data in random.sample(true_data_list, 5):
    val_data_list.append(sample_data)
for sample_data in random.sample(false_data_list, 5):
    val_data_list.append(sample_data)
for val_data in val_data_list:
    shutil.move(val_data, val_data_path/ val_data.name)
# 移动train
for pic_name in label_data.keys():
    pic_path = cleaned_frame_data_Path / pic_name
    if not pic_path.exists():
        continue
    shutil.move(pic_path, train_data_path/ pic_path.name)

