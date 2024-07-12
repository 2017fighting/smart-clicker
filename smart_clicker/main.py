import cv2 as cv
import pyautogui

from smart_clicker.manual_check import judge_by_img_data
from smart_clicker.video_catpure import grab_screen_as_video


def wait_quit():
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        exit(0)


def main():
    for video_frame in grab_screen_as_video(
        width=1280,
        height=720,
    ):
        # cv.imshow("", video_frame)
        # cv.waitKey(1)
        sc = judge_by_img_data(video_frame)
        if not sc:
            continue
        pyautogui.click(button=sc)


if __name__ == "__main__":
    main()
