#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[12]:


#get_ipython().system('pip install conda')
#get_ipython().system('conda activate tensorflowgpu')
#!pip install numpy
#!pip install scikit-image
#!pip install mrcnn
#!pip install tensorflow


# In[6]:

import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from deep_sort_pytorch.deep_sort import build_tracker
from deep_sort_pytorch.utils.draw import draw_boxes
from deep_sort_pytorch.utils.parser import get_config


# Root directory of the project
ROOT_DIR = os.path.abspath("/home/ric/Mask_RCNN_master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_uvas_0021.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG',"grapes"]


# ## Run Object Detection

# In[ ]:


# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR,"GH011754_f553.jpg"))

# Run detection


# Visualize results

#print("hola666")
#print(r["rois"].shape)
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
 #                           class_names, r['scores'])

cfg = get_config()
cfg.merge_from_file("./deep_sort_pytorch/configs/yolov3.yaml")
cfg.merge_from_file("./deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = build_tracker(cfg, use_cuda=True)


start_frame = 1000
end_frame = 1100



video_path = 'GH011754.MP4'
video = cv2.VideoCapture(video_path)  
#print(image.shape)
output = 'grape_tracks.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 10.0, (1366, 768))

totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

for frameCount in range(totalFrames):

    #print(i)
    #print(image.shape)
    success, image=video.read()
    if frameCount <= start_frame: continue
    if frameCount >= end_frame: break
    
    if not success:
        print("corrupto")
        continue
    image=cv2.resize(image,dsize=(1366,768))
    results = model.detect([image], verbose=1)
    r = results[0]



    out_image=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
    detections = r['rois']
    if(detections is not []):
        bbox_xywh = detections.copy()
        bbox_xywh[:,0] = (detections[:,1] + detections[:,3])/2 
        bbox_xywh[:,1] = (detections[:,0] + detections[:,2])/2 
        bbox_xywh[:,2] = (detections[:,3] - detections[:,1]) 
        bbox_xywh[:,3] = (detections[:,2] - detections[:,0]) 
        cls_conf = r['scores']
        outputs = deepsort.update(bbox_xywh, cls_conf, out_image)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:,:4]
            identities = outputs[:,-1]
            out_image = draw_boxes(out_image, bbox_xyxy, identities, cls_conf)

    
    out.write(out_image)
#plt.imsave("hola.jpg",masked_image.astype(np.uint8))
         
    
out.release()




