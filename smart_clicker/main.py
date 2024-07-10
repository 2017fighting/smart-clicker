from video_catpure import grab_screen_as_video
import cv2 as cv


def wait_quit():
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        exit(0)


def main():
    for video in grab_screen_as_video(
        width=200, height=200
    ):
        cv.imshow("screen", video)
        wait_quit()

if __name__ == "__main__":
    main()