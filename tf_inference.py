import tarfile
import urllib.request
import os
import time 
import sys
from pathlib import Path

import pandas as pd

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

MODEL_PATH = "./data/models/frozen_drone_5"

DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz
# efficientdet_d3_coco17_tpu-32
# Download and extract model
MODEL_DATE = '20200711'
MODEL_NAME = 'frozen_drone_5'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
print(PATH_TO_CKPT)
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    print(MODEL_DOWNLOAD_LINK)
    print(PATH_TO_MODEL_TAR)
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

# Download labels file
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')

print("Initializing TF and a model")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

images_loading_time = 0
# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def convert_tf_detection_to_yolo(image_id, box, score):
    result = {
        'image_id': image_id,
        'xc': round((box[1] + box[3])/2, 4), # center of bbox x coordinate
        'yc': round((box[0] + box[2])/2, 4), # center of bbox y coordinate
        'w': round(box[3] - box[1], 4), # width of bbox
        'h': round(box[2] - box[0], 4), # height of bbox
        'label': 0, # COCO class label
        'score': round(score, 4) # class probability score
    }
    return result

def generate_solution(image_name):
    global images_loading_time
    results = []
    image_id = image_name.name[:-len(image_name.suffix)]
    logger.debug(f"processing image: {str(image_id)}")
    image_loading_start_time = time.perf_counter()
    # image_np = cv2.imread(str(image_name))
    img = tf.io.read_file(str(image_name))
    image_np = tf.io.decode_jpeg(img, channels=3)
    image_loading_time_elapsed = time.perf_counter() - image_loading_start_time
    logger.debug("image loaded, time elapsed: {:.3f}".format(image_loading_time_elapsed))
    images_loading_time += image_loading_time_elapsed
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) 
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    for box, label, score in zip(detections['detection_boxes'][0].numpy(), 
                             detections['detection_classes'][0].numpy(), 
                             detections['detection_scores'][0].numpy()):
        if score > 0.3 and label == 0:
            print("box: ", box, ", label: ", label, ", score: ", score)
            results.append(convert_tf_detection_to_yolo(image_id, box, score))
    return results

def create_solution():
    logger.debug("creating solution")
    results = []
    for f in Path(TEST_IMAGES_PATH).glob('*.JPG'):
        results += generate_solution(f)
    print("results: ", results)

    test_df = pd.DataFrame(results, columns=['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score'])
    test_df.to_csv(SAVE_PATH, index=False)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
print(tf.config.list_physical_devices('GPU'))
print("Beginning inference")
print("imports")
import cv2
import logging
import time
import numpy as np

print("logger")
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug("initializing the variables")
# Variables to calculate FPS
start_time = time.time()
frame_counter, fps = 0, 0
fps_avg_frame_count = 10
detection_time_avg, detection_time_total = 0, 0

logger.debug("starting frame capture loop")
# while True:
# Read frame from camera
# ret, image_np = cap.read()
# image_np = cv2.imread(r"man_cafe.jpg")

frame_counter += 1
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
# image_np_expanded = np.expand_dims(image_np, axis=0)

# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(
#     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
logger.debug("detecting")
detection_start_time = time.perf_counter()
# input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
# detections, predictions_dict, shapes = detect_fn(input_tensor)
create_solution()
# logger.debug("predictions:", predictions_dict)
# logger.debug("shapes:", shapes)
logger.debug("detecting done")
if frame_counter > 1:
    detection_time = time.perf_counter() - detection_start_time
    detection_time_total += detection_time
    detection_time_avg = detection_time_total / frame_counter
    logger.info("Last frame detection time: {:.3f}".format(detection_time))
    logger.info("Average frame detection time: {:.3f}".format(detection_time_avg))

# label_id_offset = 1
# image_np_with_detections = image_np.copy()

# logger.debug("visualizing")
# viz_utils.visualize_boxes_and_labels_on_image_array(
#         image_np_with_detections,
#         detections['detection_boxes'][0].numpy(),
#         (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
#         detections['detection_scores'][0].numpy(),
#         category_index,
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=200,
#         min_score_thresh=.30,
#         agnostic_mode=False)

if frame_counter % fps_avg_frame_count == 0:
    end_time = time.time()
    fps = fps_avg_frame_count / (end_time - start_time)
    start_time = time.time()
    # logger.debug("FPS calculated, time: {:.3f}".format(time.perf_counter() - perf_start_time))
logger.info("FPS = {:.1f}".format(fps))
logger.debug("Frames total = {:.1f}".format(frame_counter))
logger.debug("Image loading total time: {:.3f}".format(images_loading_time))
# Display output
# cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
# cv2.imwrite('detection_results.jpg', image_np_with_detections)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
    # break

# if frame_counter == 1000:
#     print("Exiting...")
#     break

# cap.release()
                                                                   
