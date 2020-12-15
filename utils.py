import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


def cut_bounding_box(image, xmin, ymin, xmax, ymax):
    """Cut face from image according to detection results. Retain image size and put black borders."""
    width = xmax - xmin
    height = ymax - ymin

    # crop
    cropped_image = np.zeros(image.shape)
    cropped_image[:height, :width, :] = image[ymin:ymax, xmin:xmax, :]
    return cropped_image


def cut_dataset_bounding_boxes(images_dir):
    """Cut bounding boxes from cat_dir/raw according to cat_dir/detected/Detection_Results.csv"""
    pass


if __name__ == '__main__':
    image_num = '2.jpg'
    image_path = 'images/49403512/raw/' + image_num
    image = Image.open(image_path)
    image = np.asarray(image)
    detection_results = pd.read_csv('images/49403512/detected/Detection_Results.csv')

    xmin = detection_results.loc[detection_results['image'] == image_num, 'xmin'].values[0]
    xmax = detection_results.loc[detection_results['image'] == image_num, 'xmax'].values[0]
    ymin = detection_results.loc[detection_results['image'] == image_num, 'ymin'].values[0]
    ymax = detection_results.loc[detection_results['image'] == image_num, 'ymax'].values[0]

    cropped_image = cut_bounding_box(image, xmin, ymin, xmax, ymax)
    cropped_image = Image.fromarray(np.uint8(cropped_image))
    cropped_image.save('check.jpg')
