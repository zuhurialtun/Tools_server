
from itertools import count
import math
# import numbers
import pathlib
from statistics import mode
# from tkinter import image_names
from keras.models import load_model
# from PIL import Image, ImageOps
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from torch import batch_norm

# For running inference on the TF-Hub module.
# import tensorflow as tf

# import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
# import tempfile
# from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
# import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time


# from model3 import crop_center_square
IMG_SIZE = 180
BATCH_SIZE = 64
EPOCHS = 15
CLASS_NAMES = None

# MAX_SEQ_LENGTH = 20
# NUM_FEATURES = 2048

# train_df = pd.read_csv("ucf101Top5/train.csv")
# test_df = pd.read_csv("ucf101Top5/test.csv")

# label_processor = keras.layers.StringLookup(
#     num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
# )
# class_vocab = label_processor.get_vocabulary()

# def crop_center_square(frame):
#     y, x = frame.shape[0:2]
#     min_dim = min(y, x)
#     start_x = (x // 2) - (min_dim // 2)
#     start_y = (y // 2) - (min_dim // 2)
#     return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# def video_data_maker(video_name,max_frame=60,videos_folder='DataImages',save_frame=False):
#     data = np.ndarray(shape=(max_frame, 224, 224, 3), dtype=np.float32)
#     video_read_path = os.path.join('./'+videos_folder+'/', video_name)
#     # print(f'Video path : \'{video_read_path}\'')
#     cap = cv2.VideoCapture(video_read_path)
#     cap.set(cv2.CAP_PROP_FPS, 20)
#     # fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_slice = frame_count / max_frame
#     # print(f'Frame Slice = {frame_slice}')

#     if save_frame==True:
#         try:
#             os.mkdir(os.path.join(os.path.join('./DataImages/'),video_name.split('.')[0]))
#             # print(video_name.split('.')[0] + ' file created.')
#         except:
#                 # print(f'File {video_name} Already Created')
#                 pass

#     count=0
#     while (cap.isOpened()):
#         frameId = cap.get(1)  # current frame number
#         ret, frame = cap.read()
#         if (ret != True):
#             break
#         if count < max_frame:
#             if (frameId % math.floor(frame_slice) == 0):
#                 filename = "frame%d.jpg" % count
#                 cv2.imwrite(os.path.join('./frameTest.bmp'), frame)
#                 image = Image.open('./frameTest.bmp')
#                 size = (224, 224)
#                 image = ImageOps.fit(image, size)
#                 image_array = np.asarray(image)
#                 normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#                 data[count] = normalized_image_array
#                 count += 1
#                 if save_frame == True: cv2.imwrite(os.path.join(os.path.join(os.path.join('./'+videos_folder+'/'),video_name.split('.')[0]),filename), frame)
#     cap.release()
#     return np.array(data)

# def build_feature_extractor():
#     feature_extractor = keras.applications.InceptionV3(
#         weights="imagenet",
#         include_top=False,
#         pooling="avg",
#         input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     )
#     preprocess_input = keras.applications.inception_v3.preprocess_input

#     inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
#     preprocessed = preprocess_input(inputs)

#     outputs = feature_extractor(preprocessed)
#     return keras.Model(inputs, outputs, name="feature_extractor")


# feature_extractor = build_feature_extractor()

# def load_video(path, max_frames=10, resize=(IMG_SIZE, IMG_SIZE)):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = crop_center_square(frame)
#             frame = cv2.resize(frame, resize)
#             frame = frame[:, :, [2, 1, 0]]
#             frames.append(frame)

#             if len(frames) == max_frames:
#                 break
#     finally:
#         cap.release()
#     return np.array(frames)

# def prepare_all_videos(df, root_dir):
#     num_samples = len(df)
#     video_paths = df["video_name"].values.tolist()
#     labels = df["tag"].values
#     labels_clone = np.ndarray(shape=(num_samples * 10), dtype=('O'))

#     label_number = 0
#     last_number = 0
#     for label in labels:
#         for i in range(last_number,(last_number + 10)):
#             labels_clone[i] = label
#             label_number = i
#         last_number = label_number+1

#     for idx, path in enumerate(video_paths):
#         data_frames = video_data_maker(path,10,root_dir,True)
#         count = (idx - 1) * 10
#         data_count = 0
#         frame_features = np.zeros(
#             shape=(num_samples, IMG_SIZE, IMG_SIZE, 3), dtype="float32"
#         )
#         frame_features_clone = np.zeros(
#             shape=((num_samples * 10), IMG_SIZE, IMG_SIZE, 3), dtype="float32"
#         )
#         temp_frame_features = np.zeros(
#             shape=(1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"
#         )
#         label_number = 0
#         last_number = 0
#         for label in data_frames:
#             data_number = 0
#             for i in range(last_number,(last_number + 10)):
#                 frame_features_clone[i,] = data_frames[data_number].squeeze()
#                 label_number = i
#                 data_number += 1
#             last_number = label_number+1

#     return frame_features_clone, labels_clone

def convert_videoData_to_imageData(input_dir, output_dir, frame_size):

    pass


def model_maker(image_data_path='./ImageData',BATCH_SIZE=64,EPOCHS=15):
    data_dir = pathlib.Path(image_data_path)
    # data_dir = pathlib.Path('./flower_photos')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    global CLASS_NAMES
    CLASS_NAMES = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",
                                       input_shape=(IMG_SIZE,
                                                    IMG_SIZE,
                                                    3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    epochs = EPOCHS
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    model.summary()
    return model


def save_model(model, model_name):
    try:
        os.mkdir('./MODELS/')
        print('MODELS file created.')
    except:
        print(f'MODELS File Already Created')

    model.save('./MODELS/'+model_name+'.h5')
    model.save('./MODELS/'+model_name)

    f = open('./MODELS/'+model_name+'/class_name.txt', "w")
    for name in CLASS_NAMES:
        f.write(str(name)+'*')

def class_name_maker(data_path):
    data_dir = pathlib.Path('./'+data_path)
    batch_size = BATCH_SIZE
    img_height = IMG_SIZE
    img_width = IMG_SIZE

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = test_ds.class_names
    data_dir = pathlib.Path(data_path)
    extensions = ('*/*.jpg', '*/*.jpeg', '*/*.png')
    files_list = []
    for ext in extensions:
        files_list.extend(data_dir.glob(ext))
    image_list = list(files_list)

    return class_names,image_list

def get_class_names(MODEL_PATH):
    class_path = MODEL_PATH+"/class_name.txt"
    class_names = open(class_path, "r").read()
    CLASS_NAMES = []
    for name in class_names.split('*'):
        if len(name) > 2:
            CLASS_NAMES.append(name)
    return CLASS_NAMES

# def photo_data_maker(image_name):
#     data = np.ndarray(shape=(1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
#     IMAGE_NAME = image_name
#     IMAGE_PATH = './TestImages/'+IMAGE_NAME
#     image = Image.open(IMAGE_PATH).save("./TestImages/sample.bmp")
#     img0 = Image.open('./TestImages/sample.bmp')
#     size = (IMG_SIZE, IMG_SIZE)
#     image = ImageOps.fit(img0, size)
#     image_array = np.asarray(image)
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#     data[0] = normalized_image_array
#     return data


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

def model_output(data, class_names, model=None, saved_model=None):
    if model == None:
        if saved_model == None:
            print('We need a model!')
        else:
            new_model = tf.keras.models.load_model(saved_model)
            model = new_model
    # model.summary()
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

def total_output(class_names, data_output=None, details=False):
    if (data_output == None):
        print('We need data...')
    data = {}
    for class_name in class_names:
        data[class_name] = 0
    class_count = int(len(data_output))
    for data_ID in data_output:
        for class_name in data_output[data_ID]:
            if (details == True):
                print(f'{class_name:22}-{data[class_name]:22} += ({data_output[data_ID][class_name]} / {class_count})')
            data[class_name] += (data_output[data_ID][class_name] / class_count)
        if (details == True):
            print('-----')
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    for key, value in data.items():
        print(f'{key:24}: {value * 100:5.2f}')

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


def object_detech(image_tensor):
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    # module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    # module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    detector = hub.load(module_handle).signatures['default']
    # detector = hub.load(module_handle)
    run_detector(detector, image_tensor)


def display_image(image,image_name):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    cv2.imwrite(os.path.join('./TestImages/','obj_'+image_name), image)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                           int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)
    print('---Object List---')
    for i in range(min(len(result["detection_class_entities"]),10)):
        print(f'{(i+1)}. {str(result["detection_class_entities"][i]):22}: {(result["detection_scores"][i] * 100):5.2f}%')
    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])
    image_name = path.split('/')[2]
    display_image(image_with_boxes,image_name)
b'Fashion accessory'