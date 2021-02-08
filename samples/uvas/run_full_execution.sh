CAMPO_NAME="ConchayToro_test"
VIDEO_FILES="stuff/input_videos/processed/$CAMPO_NAME/*"
WEIGHTS_FILE="stuff/logs/dcomplete_pgrapes_aug/mask_rcnn_uvas_0020 .h5"
PICKLES_DIR="stuff/pickles/$CAMPO_NAME"
OUTPUT_VIDEO_DIR="stuff/output_videos/$CAMPO_NAME"

# -------INFERENCE AND PICKLES------
echo "Processing Splash Videos from Campo: "$CAMPO_NAME 
for f in $VIDEO_FILES
do  
    echo "Processing video:"$f
	python splash_line_cross.py --weights $WEIGHTS_FILE --video $f --pickles_dir $PICKLES_DIR/ --output_dir $OUTPUT_VIDEO_DIR
    echo "Metadata and predictio video collected."
done
# # -------MEGA PICKLE-----
# echo "Generating Mega Pickle"
# python utils/create_mega_pickle.py --pickles_dir $PICKLES_DIR/

# # -------GROUP BY CUARTEL ------
# echo "Grouping by cuartel"
# python utils/group_by_cuartel.py --locations_dir="$PICKLES_DIR/location_pickles/"

# -------HEATMAPS---------
# Heat map generation
# echo "Creating HeatMaps"
# LOCATION_PKL_DIR="$PICKLES_DIR/location_pickles"
# SAT_IMG_DIR="stuff/satellite_images/$CAMPO_NAME.jpg"
# MEGAPK_DIR="$PICKLES_DIR/mega_pickle.pkl"

# echo $SAT_IMG_DIR
# echo $LOCATION_PKL_DIR
# echo $MEGAPK_DIR
# python heat_map_generation/process_loc_pickles.py --campo $LOCATION_PKL_DIR --img $SAT_IMG_DIR  --megapk $MEGAPK_DIR