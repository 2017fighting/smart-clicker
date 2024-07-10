import json
import cv2
from smart_clicker import config
from loguru import logger


cleaned_frame_data_Path = config.PROJECT_ROOT_PATH / "cleaned_frame_data"
label_data_path = cleaned_frame_data_Path / "label_data.json"

pic_file_path_list = [pic_file for pic_file in cleaned_frame_data_Path.glob('./*')]
pic_file_path_list.sort()
pic_file_len = len(pic_file_path_list)
file_name_to_label = {file_name: 0 for file_name in pic_file_path_list}

def output_json_file():
    with open(label_data_path, 'w') as label_data_f:
        json.dump({str(k.name):v for k,v in file_name_to_label.items()}, label_data_f)

def load_json_file():
    with open(label_data_path, 'r') as label_data_f:
        old_label_data = json.load(label_data_f)
    for file_name in file_name_to_label.keys():
        file_name_to_label[file_name] = old_label_data[file_name.name]

def main():
    load_json_file()

    cur_index = 0
    window_name = "Window"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        if cur_index < 0:
            logger.info(f"{cur_index=} < 0")
            cur_index = 0
        elif cur_index >= pic_file_len:
            logger.info(f"{cur_index=} > len(pic_file_path_list)")
            cur_index = pic_file_len - 1
        pic_path = pic_file_path_list[cur_index]
        cur_label = file_name_to_label[pic_path]
        pic_data = cv2.imread(pic_path)
        cv2.putText(pic_data, f"index: {cur_index}, label:{cur_label}", (30,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255))
        try:
            cv2.imshow(window_name, pic_data)
        except Exception as e:
            logger.exception(e)
        key_get = cv2.waitKey(0) & 0xff
    
        if key_get == ord('q'):
            output_json_file()
            cv2.destroyAllWindows()
            exit(0)
        elif key_get == ord('1'):
            file_name_to_label[pic_path] = 1
            cur_index += 1
            output_json_file()
        elif key_get == ord('0'):
            file_name_to_label[pic_path] = 0
            output_json_file()
        elif key_get == ord('d'):
            cur_index +=1
        elif key_get == ord('a'):
            cur_index -= 1
        
if __name__ == "__main__":
    main()



