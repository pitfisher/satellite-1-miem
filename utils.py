import csv
import cv2
import torch
import numpy as np
import pickle


def calculate_metric(beta):
    return


def recognise_objects(image, recognition_function):
    detections = recognition_function(image)
    return detections


def load_coco_labels(path):
    # opening the file in read mode
    my_file = open(path, "r")
    # reading the file
    data = my_file.read()
    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split(" ")
    # printing the data
    my_file.close()
    return data_into_list

def process_image(image_path, label_path, recognition_function):
    #temporary code
    # set the device we will be using to run the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the list of categories in the COCO dataset and then generate a
    # set of bounding box colors for each class
    CLASSES = pickle.loads(open(args["labels"], "rb").read())
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    #
    print("Image name: ", image_path)
    print("Annotation name: ", label_path)
    image = cv2.imread(image_path)
    # some potential image preprocessing
    detections = recognise_objects(image, recognition_function)
    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            if idx == 0:
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction to our terminal
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return


def serialize_row(image_id, label, xc, yc, w, h, score):
    return [image_id, label, xc, yc, w, h, score]


def append_results(results, filename):
    # open the file in the write mode
    f = open('./test.csv', 'w')
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    for row in results:
        writer.writerow(row)

    # close the file
    f.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    process_image(r"test_data\images\1_001537.JPG", r"test_data\labels\1_001537.txt")
    append_results([["test", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], './test.csv')
