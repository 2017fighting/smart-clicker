from smart_clicker import config
import shutil

frame_data_path = config.PROJECT_ROOT_PATH / "frame_data"

cleaned_frame_data_Path = config.PROJECT_ROOT_PATH / "cleaned_frame_data"
count = 0
for pic_path in frame_data_path.glob("./*"):
    shutil.move(pic_path, cleaned_frame_data_Path / f"{count:04}.jpg")
    count +=1