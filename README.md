# cat-detector
This is a cat detector and verifier based on [this](http://cs230.stanford.edu/projects_fall_2019/reports/26251543.pdf) paper.  
It uses YOLO v3 object detector fine tuned on cat images for cat face detection.  
Cat face verfication is done by using [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) triplet loss here with orthogonal reglurization as described [here](https://arxiv.org/pdf/1708.06320.pdf).  
The triplet loss term just pushes embeddings of different identities further then embeddings of the same identity.  
The orthogonal loss term essentialy makes embeddings of different identities closer to orthogonal.  

## How it works
The user defines his cat, and then the CatEmbedder module creates a 64 dimensional embeddings which desribes that cat.  
On inference, images from camera feed are fed into to the YOLO object detector which in turn detects and crops cats faces.  
Each deteced cat-face is then embedded and then checked against own-cat embeddings to see if the euclidean distance between the two embeddings is smaller then the seperating THRESHOLD which is determined after training. 

## Dataset
This model is trained on a novel dataset prodeced from PetFinder as described in the paper above.
An API crowler was used to generate groups of images of different cats with at least 5 Pics per cat.  
There are about 80000 cat pics.  
The code for the crowler is at cat_downloader.


## Installation and prequisites
The dockerfile for building the Image for this project are called:
- Dockerfile.cpu: for running on CPU
- Dockerfile.gpu: for running on GPU
They are bare tensorflow 2.3.0 with python 3 and dependencies from requirements.txt

Initialize YOLO:
- To use YOLO with pretrained weights (slightly lesser performance) use:

      !python YOLO/initialize_yolo.py

- To train your own YOLO:

      !python YOLO/2_Training/Download_and_Convert_YOLO_weights.py
      
- Annotate images as described in [here](https://blog.insightdatascience.com/how-to-train-your-own-yolov3-detector-from-scratch-224d10e55de2)
- Put images in YOLO/data/Source_Images/Training_Images
- Run:

      !python YOLO/1_Image_Annotation/Covert_to_YOLO_format.py
      !python YOLO/2_Training/Train_YOLO.py
      
Set own cat:
- Put own cat image in application/
- Run:

      !python application/set_own_cat.py
      
Run application:

      !python application/camera_verificator.py


## Training
Training is done at multiple stages as desicribed in the paper mentioned above. Bot sure if all stages are actually necessary.  
You can train the cat verificator by running:

    !python cat_verificator/train.py

The metric used is AUC, because there is an inate bias towards different-cat in the dataset. Only 5 pics of same cat per cat...   
Triplet loss models should be trained with large batch sizes (larger then 500) because if the implementation details.  
Batch hard strategy is used, which means that the model is trained only on the hardest postive and hardest negatives.  
It thus trained on only a small portion of the valid triplets and if small batch size is used they could also be repetative and highly correlated.  
I trained the model on a 16GB GPU which was suffient for only 128 triplets per batch. This is not enough and I suspect that using higher batch size would yeild better results. 

## Model performance
The validation AUC score is 0.97, it sounds good but the FPR and TPR arent that impressive..
At 1.25 THRESHOLD:
- TPR = 0.9
- FPR = 0.1
I think this model can do better.
