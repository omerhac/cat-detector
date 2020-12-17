import cat_verificator
import matplotlib.pyplot as plt
import os
import detect_faces
import utilities
"""Demo for showing cat verification capabilities"""
"""Replace images/own.jpg with own cat and images/diff.jpg with another cat and check results!"""

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    image_size = [64, 64, 3]

    # define verificator
    cat_ver = cat_verificator.CatVerificator(image_size, data_dir=root_path + '/demo/data', load_data=False)

    # set own cat
    cat = plt.imread('images/own.jpg')
    cat_ver.set_own_image(cat)

    # crop images
    detect_faces.detect_faces(root_path + '/demo/images', root_path + '/demo/cropped', save_images=False)
    utilities.crop_directory_bounding_boxes(root_path + '/demo/images', root_path + '/demo/cropped',
                                            root_path + '/demo/cropped/Detection_Results.csv')
    # set different cat
    diff_cat = plt.imread(root_path + '/demo/cropped/diff.jpg')

    # check verification
    print(cat_ver.is_own_cat(diff_cat))

