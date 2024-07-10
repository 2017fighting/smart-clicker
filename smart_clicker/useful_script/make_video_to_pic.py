import cv2
import config

raw_data_path =  config.PROJECT_ROOT_PATH / "raw_data"
frame_data_path = config.PROJECT_ROOT_PATH / "frame_data"
frame_data_path.mkdir(parents=True, exist_ok=True)

def extract_single_video(video_path):
    vidObj = cv2.VideoCapture(video_path) 
    count = 350
    while True:
        success, image = vidObj.read()
        if not success:
            break
        cv2.imwrite(frame_data_path/f"frame{count:04}.jpg", image)
        count += 1


# extract_single_video(raw_data_path/ "1606122408-1-192 (online-video-cutter.com).mp4")
# extract_single_video(raw_data_path/ "1606122408-1-192 (online-video-cutter.com) (1).mp4")