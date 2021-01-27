CAMPO_NAME="TestCampo"
VIDEO_FILES="stuff/input_videos/processed/$CAMPO_NAME/*"
WEIGHTS_FILE="stuff/pretrained_weights/mask_rcnn_uvas_0050.h5"
PICKLES_DIR="stuff/pickles/$CAMPO_NAME"
OUTPUT_VIDEO_DIR="stuff/output_videos/$CAMPO_NAME"

# -------INFERENCE AND PICKLES------
echo "Processing Splash Videos from Campo: "$CAMPO_NAME 
for f in $VIDEO_FILES
do  
    echo "Processing video:"$f
	python splash_uvas.py --weights $WEIGHTS_FILE --video $f --pickles_dir $PICKLES_DIR/ --output_dir $OUTPUT_VIDEO_DIR
    echo "Metadata and predictio video collected."
done
