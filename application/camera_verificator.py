import cv2
import sys
import os
from PIL import Image
import numpy as np

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path + '/YOLO/2_Training/src')
from keras_yolo3.yolo import YOLO


def run():
    # initialize yolo model
    model_path = root_path + '/YOLO/Data/Model_Weights/trained_weights_final.h5'
    anchors_path = root_path + '/YOLO/2_Training/src/keras_yolo3/model_data/yolo_anchors.txt'
    classes_path = root_path + '/YOLO/Data/Model_Weights/data_classes.txt'
    yolo = YOLO(
        **{
            "model_path": model_path,
            "anchors_path": anchors_path,
            "classes_path": classes_path,
            "score": 0.25,
            "gpu_num": 1,
            "model_image_size": (256, 256),
        }
    )

    # open camera feed
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Window")

    while True:
        ret, frame = video_capture.read()

        frame = Image.fromarray(frame)
        # detect faces
        predictions, image = yolo.detect_image(frame, show_stats=False)
        print(predictions)
        cv2.imshow("Window", np.asarray(frame))
        # This breaks on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()