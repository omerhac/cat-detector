# cat-detector
This is a cat detector and verifier based on this paper.
It uses YOLO v3 object detector fine tuned on cat images for cat face detection.
Cat face verfication is done by using FaceNet triplet loss as described here with orthogonal reglurization as described here.
The triplet loss term just pushes embeddings of different identities further then embeddings of the same identity.
The orthogonal loss term essentialy makes embeddings of different identities closer to orthogonal.

## Dataset
This model is trained on a novel dataset prodeced from PetFinder as described in the paper above.
An API crowler was used to generate groups of images of different cats with at least 5 Pics per cat.
The code for the crowler is at cat_downloader.


## Installation and prequisites
The dockerfile for building the Image for this project are called:
- Dockerfile.cpu: for running on CPU
- Dockerfile.gpu: for running on GPU
They are bare tensorflow 2.3.0 with python 3 and dependencies from requirements.txt

Initialize YOLO:

    !python YOLO/initialize_yolo.py
 
Set own cat:
- Put own cat image in application/
- Run:

      !python application/set_own_cat.py
      
Run application:

      !python application/camera_verificator.py
