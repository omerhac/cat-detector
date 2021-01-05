import cv2
import sys
import os
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path + '/YOLO/2_Training/src')
sys.path.append(root_path + '/cat_verificator')
sys.path.append(root_path)

# import YOLO dependencies
from keras_yolo3.yolo import YOLO
from YOLO.Utils import utils
import utilities

# import cat verificator dependencies
from cat_verificator import CatVerificator


def run():
    print("Loading model...")

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

    # initialize verificator model
    start_time = time.time()
    verificator = CatVerificator([64, 64, 3], threshold=1.4, data_dir=dir_path + '/data', load_data=True)
    print("Loaded Cat Verficator in {:.2f}sec.".format(time.time() - start_time))

    # open camera feed
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Window")

    # set time
    tag_time = time.time()

    while True:
        ret, frame = video_capture.read()

        frame = Image.fromarray(frame)

        # detect faces only every second
        if np.isclose(time.time() - tag_time, 1, rtol=0.1):
            # detect
            predictions, _ = yolo.detect_image(frame, show_stats=False)

            # check if theres more then one cat
            if len(predictions) == 1:
                # crop images
                x_min, y_min, x_max, y_max = predictions[0][:-2]  # get only coordinates
                cropped_face = utilities.crop_bounding_box(np.asarray(frame), x_min, x_max, y_min, y_max)

                # run verificator
                same_cat, distance = verificator.is_own_cat(cropped_face)

                if same_cat:
                    # draw green bbox
                    annotated_image = utils.draw_annotated_box(frame, [predictions], ['Own_Cat'], [(85, 255, 85)])
                else:
                    # draw red bbox
                    annotated_image = utils.draw_annotated_box(frame, [predictions], ['Own_Cat'], [(0, 0, 255)])

            elif len(predictions) > 1:
                # draw yellow bboxes
                annotated_image = utils.draw_annotated_box(frame, [predictions], ['To_Many_Cats'], [(85, 255, 255)])

            tag_time = time.time()  # set time
        else:
            annotated_image = frame

        # show image
        cv2.imshow("Window", np.asarray(annotated_image))
        # This breaks on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
