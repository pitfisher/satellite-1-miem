from pathlib import Path
import os
os.environ["TMPDIR"] = str(Path().absolute())
print(os.environ["TMPDIR"])

from os import listdir
from os.path import isfile, join
from ultralytics import YOLO
import pandas as pd
import sys

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

model = YOLO('data/models/yolo-best-1206.pt')
mypath = 'data/test_data/images'
onlyfiles = [join(TEST_IMAGES_PATH, f) for f in listdir(TEST_IMAGES_PATH)]

coordinates = pd.DataFrame({'image': [], 'output': [], 'score': []})

def convert_detection_to_yolo(image_id, xywhn, score):
    x, y, w, h = xywhn
    result = {
        'image_id': image_id,
        'xc': round(float(x), 4), # center of bbox x coordinate
        'yc': round(float(y), 4), # center of bbox y coordinate
        'w': round(float(w), 4), # width of bbox
        'h': round(float(h), 4), # height of bbox
        'label': 0, # COCO class label
        'score': round(float(score), 4) # class probability score
    }
    return result

detections = model.predict(
   source=onlyfiles,
   conf=0.25
)
def generate_solution(detections):
    # print("res: ", res)
    results = []
    for detection in detections:
        # print("result.boxes: ", result.boxes)
        image_id = Path(detection.path).stem
        image_results = []
        bboxes = detection.boxes.xywhn
        scores = detection.boxes.conf
        for bbox, score in zip(bboxes, scores):
            image_results.append(convert_detection_to_yolo(image_id, bbox, score))
        print("image results: ", image_results)
        results+=image_results
    return results
    
results = generate_solution(detections)
print(results)
test_df = pd.DataFrame(results, columns=['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score'])
test_df.to_csv(SAVE_PATH, index=False)
