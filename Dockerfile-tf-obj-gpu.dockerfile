# https://towardsdatascience.com/tensorflow-object-detection-with-docker-from-scratch-5e015b639b0b
# https://dev.to/docker/machine-learning-with-tensorflow-object-detection-running-on-docker-5ek0
FROM "tensorflow/tensorflow:latest-gpu"
RUN apt update && yes | apt upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt update && apt install -y tzdata
RUN apt update && apt install -y git python3-pip python-is-python3 protobuf-compiler python-pil python-lxml libgl1 libglib2.0-0
RUN pip install --upgrade pip
RUN pip install tensorflow
# RUN pip install matplotlib
# RUN pip install jupyter
RUN mkdir -p /tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
# RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
RUN cp object_detection/packages/tf2/setup.py .
RUN python3 -m pip install .
RUN python3 -m pip install protobuf==3.20.1
# CMD python3 object_detection/builders/model_builder_tf2_test.py
ENV PYTHONDONTWRITEBYTECODE 1
RUN mkdir -p /data/data
RUN mkdir -p /data/data/models
COPY tf_inference.py /data/tf_inference.py
COPY data/man_cafe.jpg /data/man_cafe.JPG
COPY data/test_data /data/test_data
COPY data/models/frozen_drone_5 /data/data/models/frozen_drone_5
WORKDIR /data/
CMD python tf_inference.py ./data/test_data/images/ ./recognition_results.csv
# RUN jupyter notebook --generate-config --allow-root
# RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
# EXPOSE 8888
# CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tensorflow/models/research/object_detection", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
