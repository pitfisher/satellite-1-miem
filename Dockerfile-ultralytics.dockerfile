# https://towardsdatascience.com/tensorflow-object-detection-with-docker-from-scratch-5e015b639b0b
# https://dev.to/docker/machine-learning-with-tensorflow-object-detection-running-on-docker-5ek0
FROM "ultralytics/ultralytics"
# RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt update && apt install -y tzdata
# RUN apt update && yes | apt upgrade
# RUN apt update && apt install -y git python3-pip python-is-python3
RUN pip install --upgrade pip
RUN pip install dill
RUN mkdir -p /data/data
RUN mkdir -p /data/data/models
ENV PYTHONDONTWRITEBYTECODE 1
COPY ultralytics /data/ultralytics
COPY yolo_inference.py /data/yolo_inference.py
COPY data/man_cafe.jpg /data/data/man_cafe.JPG
COPY data/test_data /data/data/test_data
COPY data/models/yolo-best-1206.pt /data/data/models/yolo-best-1206.pt
WORKDIR /data/
CMD python yolo_inference.py ./data/test_data/images/ ./recognition_results.csv
