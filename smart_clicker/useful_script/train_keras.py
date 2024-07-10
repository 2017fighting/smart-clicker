import numpy as np
from smart_clicker import config
import cv2 as cv
import json
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
from loguru import logger

cleaned_frame_data_Path = config.PROJECT_ROOT_PATH / "cleaned_frame_data"
label_data_path = cleaned_frame_data_Path / "label_data.json"

train_data_path = config.PROJECT_ROOT_PATH / "train_data"
val_data_path =config.PROJECT_ROOT_PATH / "val_data"

logger.info('begin load data')
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
    if len(labels) > 10:
        break
logger.info('load data done.')
X = np.array(trainingData)
y = np.array(labels)
num_samples = len(labels)
# num_features = 500 # TODO

# 将标签转化为独热编码
y_onehot = np.zeros((num_samples, 2))
for i in range(num_samples):
    y_onehot[i, y[i]] = 1

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 创建神经网络模型
model = Sequential()
model.add(Dense(
    64, 
    # input_dim=num_features, 
    activation='relu'
))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 编译模型
optimizer = adam_v2.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
