from tensorflow import keras
from keras.models import load_model
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from model3 import crop_center_square
IMG_SIZE = 180
BATCH_SIZE = 64
EPOCHS = 15
CLASS_NAMES = None

def get_class_names(MODEL_PATH):
    class_path = MODEL_PATH+"/class_name.txt"
    class_names = open(class_path, "r").read()
    CLASS_NAMES = []
    for name in class_names.split('*'):
        if len(name) > 2:
            CLASS_NAMES.append(name)
    return CLASS_NAMES

def photo_data_maker(image_path):
    data = np.ndarray(shape=(1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    data[0] = img_array

    return data

def get_model(saved_model):
    new_model = tf.keras.models.load_model(saved_model)
    return new_model

def model_output(data, class_names, model=None):
    if model == None:
        print('We need a model!')
    data_count = len(data)
    print(f'Data Count = {data_count}')
    data_output = {}
    for frame_data_ID in range(data_count):
        predictions = model.predict(data)[frame_data_ID]
        soft_prediction = tf.nn.softmax(predictions)
        data_output[frame_data_ID] = {}
        for i in np.argsort(soft_prediction)[::-1]:
            data_output[frame_data_ID][str(
                class_names[i])] = soft_prediction[i]
    return data_output

def json_output(class_names, data_output=None):
    if (data_output == None):
        print('We need data...')
    data = {}
    json_data = {}
    for class_name in class_names:
        data[class_name] = 0
    class_count = int(len(data_output))
    for data_ID in data_output:
        for class_name in data_output[data_ID]:
            data[class_name] += (data_output[data_ID][class_name] / class_count)
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    for key, value in data.items():
        json_data[key] = (f'{value * 100:5.2f}')
    return json_data