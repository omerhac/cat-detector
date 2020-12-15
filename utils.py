import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def crop_bounding_box(image, xmin, xmax, ymin, ymax):
    """Cut face from image according to detection results. Retain image size and put black borders."""
    width = xmax - xmin
    height = ymax - ymin

    # crop
    cropped_image = np.zeros(image.shape, dtype=np.uint8)
    cropped_image[:height, :width, :] = image[ymin:ymax, xmin:xmax, :]
    return cropped_image


def get_bbox(image_name, bbox_csv):
    """Return bbox coordinates as (x_min, x_max, y_min, y_max) as highest confidence bounding box.
    Also return bbox count
    Args:
        image_name: image name in directory, for example 0.jpg
        bbox_csv: csv file containing bbox coordinates
    """

    bboxes = bbox_csv.loc[bbox_csv['image'] == image_name]  # get all bboxes for the image
    bbox_count = bboxes.shape[0]  # how many bboxes

    if bbox_count > 0:
        # get highest confidence bbox
        best_bbox = bboxes.iloc[bboxes['confidence'].argmax()]
        xmin, xmax, ymin, ymax = best_bbox['xmin'], best_bbox['xmax'], best_bbox['ymin'], best_bbox['ymax']

        return (xmin, xmax, ymin, ymax), bbox_count
    else:
        return None, 0  # no bboxes found


def crop_directory_bounding_boxes(input_dir, output_dir, bbox_csv_path):
    """Crop all images in input_dir according to bbox_csv and save in output_dir"""
    image_paths = glob.glob(input_dir + '/*.jpg')
    bbox_csv = pd.read_csv(bbox_csv_path)

    # make output dir if it doesnt exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate over each image
    for image_path in image_paths:
        image = plt.imread(image_path)  # load image
        image_name = os.path.basename(image_path)

        # get bbox
        bbox, bbox_count = get_bbox(image_name, bbox_csv)
        if bbox_count > 0:
            xmin, xmax, ymin, ymax = bbox
            cropped_image = crop_bounding_box(image, xmin, xmax, ymin, ymax)  # crop best bbox
            plt.imsave(output_dir + '/' + image_name, cropped_image)  # save


def crop_dataset_bounding_boxes(images_dir):
    """Cut bounding boxes from cat_dir/raw according to cat_dir/detected/Detection_Results.csv"""
    cat_dirs = glob.glob(images_dir + '/*')  # get all cat directories

    # crop images for each cat
    for cat in cat_dirs:
        # paths to files of each cat
        raw_images = cat + '/raw'
        cropped_images = cat + '/cropped'
        csv_path = cat + '/detected/Detection_Results.csv'

        # crop
        crop_directory_bounding_boxes(raw_images, cropped_images, csv_path)


if __name__ == '__main__':
    crop_dataset_bounding_boxes('images')
