"""Demo for showing cat verification capabilities"""
import cat_verificator
import matplotlib.pyplot as plt
import os

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    image_size = [64, 64, 3]

    # define verificator
    cat_ver = cat_verificator.CatVerificator(image_size, data_dir=root_path + '/demo/data', load_data=False)

    # set own cat
    cat = plt.imread('own.jpg')
    cat_ver.set_own_image(cat)

    # set different cat
    diff_cat = plt.imread('diff.jpg')

    # check verification
    print(cat_ver.is_own_cat(diff_cat))

