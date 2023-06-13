from ultralytics import YOLO
import pandas as pd
from os import listdir
from os.path import isfile, join


model = YOLO('data/models/yolo-best-1206.pt')
mypath = 'data/test_data/images'
onlyfiles = [join(mypath, f) for f in listdir(mypath)]


coordinates = pd.DataFrame({'image': [], 'output': [], 'score': []})

res = model.predict(
   source=onlyfiles,
   conf=0.25
)
for result in res:
    output = []
    scores = []
    cords = result.boxes.xyxy
    sc = result.boxes.conf
    for i, cor in enumerate(cords):
        output.append(['person', int(cor[0]), int(cor[1]), int(cor[2]), int(cor[3])])
        scores.append(sc[i].item())
    
    coordinates.loc[len(coordinates)] = [result.path, output, scores]