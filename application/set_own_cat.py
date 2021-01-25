import utilities
import cat_verificator
import detect_faces
import os
import matplotlib.pyplot as plt

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    image_size = [256, 256, 3]

    # define verificator
    cat_ver = cat_verificator.CatVerificator(image_size, data_dir=root_path + '/application/data', load_data=False)

    # set own cat
    cat = plt.imread('own.jpg')
    cat_ver.set_own_image(cat)

    # crop images
    detect_faces.detect_faces(root_path + '/application', root_path + '/application/data/cropped', save_images=False)
    utilities.crop_directory_bounding_boxes(root_path + '/application', root_path + '/application/data/cropped',
                                            root_path + '/application/data/cropped/Detection_Results.csv')