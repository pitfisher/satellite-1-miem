import csv
import cv2


def open_image(name):

    return 0


def calculate_metric(beta):
    return 0


def serialize_row(image_id, xc, yc, w, h, label, score):
    return [image_id, xc, yc, w, h, label, score]


def save_results(row):
    # open the file in the write mode
    f = open('./test.csv', 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(row)

    # close the file
    f.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    save_results(["test", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
