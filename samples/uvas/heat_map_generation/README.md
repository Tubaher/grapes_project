# Heat Maps from Pickles files

In this folder [heat_maps_generation] we can find everything realated to creation of pickles files for a vineyard. This pickles are created by fields and quarters.

First, we need the pickle files inside the folder [location_pickles], where every pickle has to be inside a subfolder of its respective field and quarter. The structure is the following:


```bash
Curico/
├── location_pickles/
│   ├── vinaSP_2_2
│   ├── vinaSP_2_3
│   └── vinaSP_2_4
|── prediction_pickles/
|── mega_pickle.pkl    
```

Here, vinaSP_2_2 represent a quarter in the Curico's field. The folder's name would be:
vinaSp_#field_#quarter

Inside the folder of each quarter, we need to place the pickle files that represent each rows of the quarter.
Then, the format will be:
locvinaSP_2_2_2_1.pkl
locvinaSP_#field_#quarter_#row_#time(am o pm).pkl

# Running
To generate the heat maps of a certain field, we need to use the following script with the following parameters:
[process_loc_pickles.py] --field path to the field's folder --img path to the satelital image of the field --megapk path to the mega_pickles.pkl of the field.

example: 
python process_loc_pickles.py --campo $LOCATION_PKL_DIR --img $SAT_IMG_DIR --megappk $MEGAPKL_DIR

To optimize the creation of heatmaps, we implement a bash script where the parameters above are variables of the script. The name of the bash script is [get_heatmaps.sh] and its excecution will be:

./get_heatmaps.sh
