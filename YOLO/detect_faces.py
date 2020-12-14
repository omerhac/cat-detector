import os
import subprocess
import time
import glob
import sys


def make_call_string(arglist):
    result_string = ""
    for arg in arglist:
        result_string += "".join(["--", arg[0], " ", arg[1], " "])
    return result_string


def detect_faces(input_dir, output_dir, multiple_inputs_flilepath=None):
    """Detect faces in input_dir and put on output_dir. If detecting for multiple input directories,
    A file with input directory paths shold be provided.
    """
    # create paths
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_folder, "Data")
    model_folder = os.path.join(data_folder, "Model_Weights")

    # Now run the cat face detector
    detector_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "3_Inference", "Detector.py"
    )
    result_file = os.path.join(output_dir, "Detection_Results.csv")
    model_weights = os.path.join(model_folder, "trained_weights_final.h5")
    classes_file = os.path.join(model_folder, "data_classes.txt")
    anchors = os.path.join(
        root_folder, "2_Training", "src", "keras_yolo3", "model_data", "yolo_anchors.txt"
    )
    arglist = [
        ["input_path", input_dir],
        ["classes", classes_file],
        ["output", output_dir],
        ["yolo_model", model_weights],
        ["box_file", result_file],
        ["anchors", anchors],
        ["file_types", ".jpg .jpeg .png"],
    ]

    # check for multiple inputs
    if multiple_inputs_flilepath:
        arglist.append(["multiple_inputs_filepath", multiple_inputs_flilepath])

    call_string = " ".join(["python", detector_script, make_call_string(arglist)])
    print("Detecting Cat Faces by calling: \n\n", call_string, "\n")
    start = time.time()
    subprocess.call(call_string, shell=True)
    end = time.time()
    print("Detected Cat Faces in {0:.1f} seconds".format(end - start))


def detect_dataset_faces():
    """Detect faces in all of the dataset images. Keep images in cat_dir/detected."""
    # get dirs
    images_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/images'
    cat_dirs = glob.glob(images_dir + '/*')

    # detect faces for all cats
    with open('Data/inputs_file.txt', 'w') as file:
        for cat in cat_dirs:
            file.write(cat + '\n')

    detect_faces('Data/Source_Images/Test_Images', 'Data/Source_Images/Test_Image_Detection_Results',
                 multiple_inputs_flilepath='Data/inputs_file.txt')


if __name__ == '__main__':
    detect_dataset_faces()
