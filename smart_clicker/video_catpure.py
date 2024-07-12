import numpy as np
from mss import mss
import cv2 as cv

def grab_screen_as_video(top=0, left=0, width=-1, height=-1):
    with mss(with_cursor=True) as sct:
        while True:
            sct_img = sct.grab({'top': top, 'left': left, 'width': width, 'height': height})
            yield cv.cvtColor(np.array(sct_img), cv.COLOR_BGRA2BGR)
