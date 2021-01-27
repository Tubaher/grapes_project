## Grapes Detection, Bunches Counting and HeatMaps Generation

This folder encloses the main scripts to train and evaluate a neural network (based on MR-CNN main module), which detects grapes from videos. Also, we added some configuration for taking metadata information in the form of pickles files. Then, these pickles are used to generate heatmaps over satellite images.

The full workflow to execute and get results is the following:

1. Grapes Detection using [`splash_uvas.py`](splash_uvas.py).

	Input: `video.mp4` Output: location and prediction pickles files
	
2. Create Mega Pickle with the prediction metadata using [`create_mega_pickle.py`](utils/create_mega_pickle.py)

	Input: prediction pickles Output: `mega_pickle.pkl` with info of all videos
	
3. Group by cuartel using [`group_by_cuartel.py`](utils/group_by_cuartel.py)

	Input: location pickles Output: location pickles order by cuartel in folders
	
4. Heatmaps generation with a satellite image and pickles information using [`process_loc_pickles.py`](heat_map_generation/process_loc_pickles.py)

	Input: `mega_pickle.pkl` `satellite_image` `dir_location_pickles` Output: `heat_map.jpg`

We provide a bash script [`run_full_execution.sh`](run_full_execution.sh) to perform the full execution of the workflow. To run this script you need an appropiate stuff folder with the information of weights, input_videos, satellite image, etc. You can find an example of this folder in [`driver`](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing). The path of this folder is `grapes_project/samples/uvas/stuff/`.

Additionally, we provided in this document a detail explanation of each step.

## Detection and count of grapes bunches per row

The main file is [`splash_uvas.py`](splash_uvas.py) which parameters for execution are the paths of `pretrained_weights` and the path of the videos to process. 
The `splash_uvas.py` file is the inference implementation of a` maskrcnn` that identifies bunches of grapes and that, together with a tracker implemented in `PyTorch`, keeps their count per row.


### Steps

1. Pretrained weights must be saved in the `stuff/pretrained_weights/` folder and the file must have the extension `.h5`. 

2. The videos to be processed must be stored in the folder `"stuff/input_videos/processed/`. Examples of processed videos are provided in the following link [drive] (https://drive.google.com/drive/folders/1HqjbKdNxMhJTVpt2gLP7_xK0kvxRU2wG? usp = sharing). However, to configure the proper name of each video the script [process_video.py] (utils/process_video.py) is used. Thus, its final name must take the following format: `"<name>_<client_id>_<field_id>_<row_id>_<ampm>.MOV "`. 

3. Another essential file for the code to work is the file of the network weights `deep_sort` these must be saved with the following address `deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7`

4. To execute the predictions of a video we use the following command, and return metadata:

		cd samples/uvas
		python splash_uvas.py \
			--weights=stuff/pretrained_weights/mask_rcnn_uvas_0029.h5 \
			--video=stuff/input_videos/processed/TesCampo/DJI_0211.MOV \
			--pickles_dir=stuff/pickles/TestCampo/ \
			--output_dir=stuff/output_videos/TestCampo/


5. The videos resulting from the detection and inference are saved in the `stuff` folder inside` output_videos/TestCampo/`. The metadata data is saved in the `.pkl` pickle format in the` stuff/pickles/TestCampo` folder. The pickles that contain the location and coordinates are stored in `'location_pickles'` while the pickles that store the number of bunches per row are stored in`' prediction_pickles'`. The content of the meta data within each `prediction_pickles` has the following format:` ["row_id", "cluster_id", "cluster_area", "cluster_lng", "cluster_lat", "cluster_sepa", "timestamp"] `

### Run multiple videos

To run `splash_uvas.py` on a video folder, use the file`splash_multi_videos.sh` where the variables `FILES` and` WEIGHTS_FILE` must be specified. The `FILE` variable is a string with the address of the videos to be processed, while the` WEIGHTS_FILE` variable is a string with the address of the weights to be used. Its execution is with the command `./splash_multi_videos.sh`.

### View pickles files.
To visualize the data that is saved in the pickles files, a Jupyter Notebook [ReadPickles.ipynb] (utils/notebooks/ReadPickles.ipynb) is provided in which the data from pickles is passed to pandas.DataFrame and saved in csv. if necessary.

## Mega Pickle Generation

The script [`create_mega_pickle.py`](utils/create_mega_pickle.py) stores all detections to the prediction pickles in a full dataframe. The repeated predictions are filtered.

The columns of the DataFrame follow this structure:

	COLUMNAS = CAMPO_ID | CUARTEL_ID | HILERA_ID | AM/PM | RACIMO_ID | AREA_I

## Order by Cuartel

The script [`group_by_cuartel.py`](utils/group_by_cuartel.py) orders the location pickles by cuartel. The folders must follow this format: `cuartel_<campo_id>_<cuartel_id>`. In this way, each pickle is moved to the corresponding folder given the `campo_id` and `cuartel_id`.

## HeatMaps Generation

To know how to generate the the HeatMaps, please go to [`README.md`](heat_map_generation/README.md).

---

## Train the MR-CNN model

In short, to train the model with a new dataset, we use the script [train.py](train.py). In the code, we need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

### Steps

1. Prepare the dataset, we need to provide image samples together with a .json file that stores the annotation per image. To do that, we can label the images using the following online tool [VGG annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). Go to the web site an use region shape option: `Polyline`. Then, click annotations and save them in .json format. Both images and annotations must be stores in the same folder.

2. Additionally, we need some pre-trained weights to train the new model. For that, we could use the pre-trained weights found in [drive](https://drive.google.com/drive/folders/1BVnFb5XKCctHdzKL2XMRAoYWUNlufd8o?usp=sharing).

3. We can configure the training according our needs. To do that, change the configurations of the class `TrainConfig` in the file [train.py](train.py)

4. Finally, to run a training we use the following command:

		python train.py \
			--dataset=stuff/datasets/datasetCurico \
			--weights=stuff/pretrained_weights/mask_rcnn_uvas_0050.h5

5. The logs and model are stored in the folder `stuff\logs\`.

---

## Notes and Problems

1. A common problem that we detected in systems with multiple GPUs. The model `deep_sort` does not load in the same GPU that the model `mrcnn`. This means that the inputs are loaded in a GPU, whereas the weights in other. To solve it, we change the file [feature_extraction.py](deep_sort_pytorch/deep_sort/deep/feature_extraction.py). In the initilization of Extractor class, we change `cuda` for `cuda:0`, with this all the information is loaded in the same GPU.
