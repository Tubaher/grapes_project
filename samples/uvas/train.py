"""
Mask R-CNN
Train on the toy Grape dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 grape.py train --dataset=/path/to/grape/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 grape.py train --dataset=/path/to/grape/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 grape.py train --dataset=/path/to/grape/dataset --weights=imagenet

    # Apply color splash to an image
    python3 grape.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 grape.py splash --weights=last --video=<URL or path to file>
"""
import keras
import os
import sys
import json
import datetime
import numpy as np
import imgaug
import skimage.draw
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
import random
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "samples/uvas/stuff/logs")

############################################################
#  Configurations
############################################################


class TrainConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Give the configuration a recognizable name
    NAME = "uvas"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + grape

    # Number of epochs for keras train
    EPOCHS = 100

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class GrapeDataset(utils.Dataset):
    def load_grape(self, dataset_dir, subset):
        """Load a subset of the grape dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Add classes. We have only one class to add.
        self.add_class("grape", 1, "grape")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        files=os.listdir(dataset_dir)

        for file in files:
            # extract image names
            filename, filename_ext = os.path.splitext(file)

            if filename_ext==".jpg":

                image_path = os.path.join(dataset_dir,file)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                mask_path=os.path.join(dataset_dir,filename+".npz")

                #print(mask_path)

                self.add_image(
                    "grape",
                    image_id=file,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    mask_path=mask_path)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a grape dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "grape":
            return super(self.__class__, self).load_mask(image_id)

        # load the masks directly
        mask_path=image_info["mask_path"]
        mask = np.load(mask_path)['arr_0']

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "grape":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GrapeDataset()
    dataset_train.load_grape(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GrapeDataset()
    dataset_val.load_grape(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # seq = imgaug.augmenters.Sequential([imgaug.augmenters.Fliplr(0.5),imgaug.augmenters.imgcorruptlike.Contrast(severity=1),imgaug.augmenters.imgcorruptlike.Brightness(severity=random.randint(1,4))])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers='heads',
                augmentation=None)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect grapes.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/uvas/dataset/",
                        help='Directory of the Grape dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    
    config = TrainConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    train(model)
