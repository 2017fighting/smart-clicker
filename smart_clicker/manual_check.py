import math
import random
import time

import cv2 as cv
import numpy as np
from loguru import logger
from skimage.metrics import structural_similarity

from smart_clicker import config

SHOW_IMG = False
DEBUG_ALL = True
SHOW_DEBUG_IMG = False


MOUSE_LEFT_IMG = cv.imread(config.PROJECT_ROOT_PATH / "assert/mouse_left.jpg")
MOUSE_RIGHT_IMG = cv.imread(config.PROJECT_ROOT_PATH / "assert/mouse_right.jpg")


def make_zero_img(img):
    return np.zeros(img.shape[:2], dtype=np.uint8)


def get_center_point(img):
    return (img.shape[1] // 2, img.shape[0] // 2)


def make_circle_mask(img, r1, r2):
    center_point = get_center_point(img)
    c1 = cv.circle(make_zero_img(img), center_point, r1, (255, 255, 255), -1)
    c2 = cv.circle(make_zero_img(img), center_point, r2, (255, 255, 255), -1)
    mask = cv.subtract(c2, c1)
    return mask


def is_red_point(color):
    b, g, r = color
    if r < 120:
        return False
    if g > r or b > r:
        return False
    return True


def is_hot_area_point(color):
    b, g, r = color
    if g < 140 or b < 140:
        return False
    return True


def circle_iter(nn, start_idx):
    for idx in range(start_idx, len(nn)):
        yield idx
    for idx in range(0, start_idx):
        yield idx


def find_red_point_range(pixel_list):
    pixel_len = len(pixel_list)

    # ensure start_idx is not a red point, make it esay to find the start edge of red point area
    start_idx = 0
    for _ in range(3):
        if not is_red_point(pixel_list[start_idx]):
            break
        start_idx = random.randint(0, pixel_len - 1)
    # after try 3 times, start_idx is still a red point, we consider as all pixel is red point
    if is_red_point(pixel_list[start_idx]):
        return None, None

    red_start_idx, red_end_idx = None, None
    for idx in circle_iter(pixel_list, start_idx):
        if not is_red_point(pixel_list[idx]):
            continue
        if red_start_idx is None:
            red_start_idx = idx
            red_end_idx = idx
        else:
            red_end_idx = idx
    return red_start_idx, red_end_idx


def polar(img):
    h, w = img.shape[:2]
    max_radius = math.hypot(w / 2, h / 2)
    return cv.linearPolar(
        img, get_center_point(img), max_radius, cv.WARP_FILL_OUTLIERS + cv.INTER_LINEAR
    )


def get_pixel_list_by_yaxis(img, yaxis):
    pixel_list = []
    for x in range(img.shape[0]):
        pixel_list.append(img[x, yaxis])
    return pixel_list


def show_img(img):
    if not SHOW_IMG:
        return
    cv.imshow("", img)
    cv.waitKey()


def get_avg_pixel_color(pixel_list, start_idx, pixel_len):
    res = [0, 0, 0]
    cnt = 0
    for idx in circle_iter(pixel_list, start_idx):
        res[0] += pixel_list[idx][0]
        res[1] += pixel_list[idx][1]
        res[2] += pixel_list[idx][2]
        cnt += 1
        if cnt >= pixel_len:
            break
    res = np.array(res)
    return np.array(res / cnt, dtype=np.uint8)


def should_click(polar_judge_area):
    target_yaxis = 95
    judge_pixel_list = get_pixel_list_by_yaxis(polar_judge_area, target_yaxis)
    judege_pixel_len = len(judge_pixel_list)
    red_start_idx, red_end_idx = find_red_point_range(judge_pixel_list)
    if red_start_idx is None or red_end_idx is None:
        if not DEBUG_ALL:
            logger.info(f"{red_start_idx=} {red_end_idx=}")
        return False
    if SHOW_DEBUG_IMG:
        for idx in range(red_start_idx, red_end_idx + 1):
            judge_pixel_list[idx][0] = 0
            judge_pixel_list[idx][1] = 0
            judge_pixel_list[idx][2] = 0
        show_img(polar_judge_area)
    forward_idx = red_start_idx - 2
    if forward_idx < 0:
        forward_idx += judege_pixel_len
    forward_pixel_avg = get_avg_pixel_color(judge_pixel_list, forward_idx, 2)
    backward_pixel_avg = get_avg_pixel_color(judge_pixel_list, red_end_idx + 1, 2)
    if not is_hot_area_point(forward_pixel_avg):
        if not DEBUG_ALL:
            logger.info(f"{forward_pixel_avg=}, {judge_pixel_list[forward_idx]}")
        return False
    if not is_hot_area_point(backward_pixel_avg):
        if not DEBUG_ALL:
            logger.info(f"{backward_pixel_avg=}, {judge_pixel_list[red_end_idx]}")
        return False
    return True


def get_mouse_target(img):
    mouse_img = img[
        435:475,
        940:980,
    ]
    score = structural_similarity(
        mouse_img, MOUSE_RIGHT_IMG, win_size=35, channel_axis=2
    )
    if score > 0.8:
        return "right"
    # logger.info(f"ssim to right:{score=}")
    score = structural_similarity(
        mouse_img, MOUSE_LEFT_IMG, win_size=35, channel_axis=2
    )
    if score > 0.8:
        return "left"
    return None
    # logger.info(f"ssim to left:{score=}")


def judge_by_img_data(image_data):
    image_dsize = (1920, 1080)
    image_data = cv.resize(image_data, image_dsize)
    show_img(image_data)

    judge_area = image_data[373:533, 880:1040]
    show_img(judge_area)

    polar_judge_area = polar(judge_area)
    show_img(polar_judge_area)
    sc = should_click(polar_judge_area)
    if not sc:
        return None
    return get_mouse_target(image_data)


def judge_single_img(img_path):
    image_data = cv.imread(img_path)
    judge_by_img_data(image_data)


def test_all():
    global SHOW_IMG, DEBUG_ALL
    SHOW_IMG = False
    DEBUG_ALL = True
    img_dir = config.PROJECT_ROOT_PATH / "frame_data/0"
    for img_path in img_dir.glob("./*.jpg"):
        sc = judge_single_img(img_path)
        if sc:
            logger.info(f"{img_path.name} {sc=}")
    img_dir = config.PROJECT_ROOT_PATH / "frame_data/1"
    for img_path in img_dir.glob("./*.jpg"):
        sc = judge_single_img(img_path)
        if sc != "left":
            logger.info(f"{img_path.name} {sc=}")
    img_dir = config.PROJECT_ROOT_PATH / "frame_data/2"
    for img_path in img_dir.glob("./*.jpg"):
        sc = judge_single_img(img_path)
        if sc != "right":
            logger.info(f"{img_path.name} {sc=}")


def test_single():
    global SHOW_IMG, SHOW_DEBUG_IMG, DEBUG_ALL
    SHOW_IMG = False
    SHOW_DEBUG_IMG = True
    DEBUG_ALL = False
    img_path = config.PROJECT_ROOT_PATH / "frame_data/0/0236.jpg"
    start_time = time.time()
    print(judge_single_img(img_path))
    cost_time = time.time() - start_time
    logger.info(f"{cost_time=}")


# test_all()
# test_single()
