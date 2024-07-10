import time
import cv2 as cv
import numpy as np
import json
from smart_clicker import config
from loguru import logger


cleaned_frame_data_Path = config.PROJECT_ROOT_PATH / "cleaned_frame_data"
label_data_path = cleaned_frame_data_Path / "label_data.json"

train_data_path = config.PROJECT_ROOT_PATH / "train_data"
val_data_path =config.PROJECT_ROOT_PATH / "val_data"


svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))


trainingData = []
labels = []
pic_path_list = []
with open(train_data_path/"label_data.json", 'r') as label_data_f:
    label_data = json.load(label_data_f)
for pic_name, label in label_data.items():
    pic_path = train_data_path / pic_name
    if not pic_path.exists():
        continue
    pic_data = cv.imread(pic_path)
    pic_data = np.array(pic_data.tolist(), dtype=np.float32)
    pic_path_list.append(pic_path)
    trainingData.append(pic_data)
    labels.append(label)
    # if len(trainingData) == 3:
    #     break
labels = np.array(labels)
trainingData = np.array(trainingData, dtype=np.float32)
trainingData.reshape(-1, trainingData.shape[-1])
trainingData = np.matrix(trainingData, dtype=np.float32)
logger.info('begin train')
logger.info(trainingData.shape)
logger.info(trainingData.dtype)
svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)

logger.info(int(time.time()))
# val
for val_pic_path in val_data_path.glob("./*"):
    manual_set_label = label_data[val_pic_path.name]
    val_pic_data = cv.imread(val_pic_path)
    val_pic_data = np.array(val_pic_data.tolist(), dtype=np.float32)
    predict_label = svm.predict(val_pic_data)[1]
    logger.info(f"{val_pic_path.name=} {manual_set_label=} {predict_label=}")
    logger.info(int(time.time()))