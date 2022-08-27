# Main Grapes Repository (R-CNN)

**Project Page:** [zosov.github.io/portfolio/mask_rcnn/](https://zosov.github.io/portfolio/mask_rcnn/)

This is an additional implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) for grapes mask detection, grapes bunches counting, and heat maps generation. We based this work on the main architecture implementation from [main architecture](https://github.com/matterport/Mask_RCNN), and the grape detection sample from [grape sample detection](https://github.com/johncuicui/grapeMRCNN.git). Also, you can read [the original README](original_readme.md) for understanding our uptades.

## Contributions

The main contribution of this work is the addition of a tracker model (programmed with Pytorch). This implementation helps us to count the number of bunches without repetitions detected in a specific video. Another contribution is the heatmaps image generation from satellite images of one particular field. These images comprise the visual interpretability of the grapes bunches in an area.

## Repository composition 

This repository is composed of two main folders `mrcnn` and `samples`. The `mrcnn` folder comprises the main architecture implementation in Keras, and the `samples` comprises two samples of use. The use of `coco` sample and demos can be found in [README](original_readme.md). To this work, we focus on the [uvas](samples/uvas) sample. Review the README of that folder.


## Installation Requirements

To execute this project you need:

1. Python version: 3.6.9
2. Create a python environment (using pip env or conda) with requirment.txt installed.
3. File .h5 with the pretrained model weights of the mrcnn.
4. File .t7 with the weights of the DeepSort model.
5. A video for testing (in the work context of course).

### Detailed Steps

1. Create the environment with conda and python 3.6.9

`conda create -n grapes python=3.6.9`

2. Install the requirements

`pip install -r requirements.txt`

3. Install cuda toolkit 10.0 and cudnn 7.+. (You must have the drivers already installed)

	conda install -c anaconda cudatoolkit=10.0
	conda install -c anaconda cudnn=7

4. Download pretrained weights .h5 from [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing):
	- Or request the file .h5 to a team colleague.

5. Download tracker pytorch checkpoint ckpt.t7 from [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing) in the folder pretrained weights
    - Copy the file to the directory `maskrcnngrape/samples/uvas/deep_sort_pytorch/deep_sort/deep/checkpoint/`

6. Download a test video from [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing):	
	- Or request the video to a team colleague.

### Grapes Video Prediction

Execution one splash grape prediction in a video

  	cd samples/uvas
	python splash_uvas.py \
		--weights=stuff/pretrained_weights/mask_rcnn_uvas_0029.h5 \
		--video=stuff/input_videos/processed/DJI_0211.MOV 

---

### Notes and Problems

Here, you can find some problems with its respective solution that we found until now.
Feel free to add a new problem and its respective solution.

1. (issues) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script

	- (solution)(shell): (https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented)
    - `sudo apt-get install libgtk2.0-dev pkg-config`

2. (issues) The cuda toolkit must be 10.0 and the cudnn > 7.5

3. To retrain the model, you can use all the available GPUs, changind the value `IMAGES_PER_GPU`. However, if you run the inference, we recomend to test first with only one GPU `IMAGES_PER_GPU=1`, in the InferenceConfig of the file [splash_uvas.py](samples/uvas/splash_uvas.py).
