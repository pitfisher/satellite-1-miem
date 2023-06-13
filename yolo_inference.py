from ultralytics import YOLO
import pandas as pd
from os import listdir
from os.path import isfile, join
import sys
from pathlib import Path

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

model = YOLO('data/models/yolo-best-1206.pt')
mypath = 'data/test_data/images'
onlyfiles = [join(TEST_IMAGES_PATH, f) for f in listdir(TEST_IMAGES_PATH)]

coordinates = pd.DataFrame({'image': [], 'output': [], 'score': []})

def convert_detection_to_yolo(image_id, yolov8_result):
    original_width, original_height = yolov8_result.orig_shape
    x, y, w, h = yolov8_result.boxes.xywhn[0]
    score = result.boxes.conf[0]
    result = {
        'image_id': image_id,
        'xc': round(x, 4), # center of bbox x coordinate
        'yc': round(y, 4), # center of bbox y coordinate
        'w': round(w, 4), # width of bbox
        'h': round(h, 4), # height of bbox
        'label': 0, # COCO class label
        'score': round(score, 4) # class probability score
    }
    return result

res = model.predict(
   source=onlyfiles,
   conf=0.25
)
# print("res: ", res)
for result in res:
    print("result.boxes: ", result.boxes)
    image_id = Path(result.path).stem
    output = []
    scores = []
    coords = result.boxes.xyxy
    sc = result.boxes.conf
    for i, coor in enumerate(coords):
        print("image_id", image_id)
        output.append(['person', int(coor[0]), int(coor[1]), int(coor[2]), int(coor[3])])
        scores.append(sc[i].item())
    
    coordinates.loc[len(coordinates)] = [result.path, output, scores]

print(coordinates)
