import keras
import time
# import matplotlib.pyplot as plt
from smart_clicker import config
from loguru import logger
import pathlib

model_path = config.PROJECT_ROOT_PATH / "models"

# model = keras.saving.load_model(model_path / "save_at_25.keras")
model = keras.saving.load_model(model_path / "save_at_25.keras")
image_size = (180, 180)

def predict_single_img(img_path):
    # img = keras.utils.load_img("smart_clicker/frame_data/1/0004.jpg", target_size=image_size)
    img = keras.utils.load_img(img_path, target_size=image_size)
    # plt.imshow(img)
    
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
    
    predictions = model.predict(img_array)
    # print(predictions)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    logger.info(f"{pathlib.Path(img_path).name}=>{score}")
    # print(score)
    # print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

# start_time = time.time()
# predict_single_img("smart_clicker/frame_data/0/0544.jpg")
# cost_time = time.time() - start_time
# logger.info(f"{cost_time}")
true_data_path = config.PROJECT_ROOT_PATH / "frame_data/0"
for img_path in true_data_path.glob("./*"):
    predict_single_img(img_path)
