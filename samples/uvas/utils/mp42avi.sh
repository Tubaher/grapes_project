echo "changing from MP4 to avi"  
VIDEO_FILES="stuff/input_videos/processed/ConchayToro_test/*"

for f in $VIDEO_FILES
do  
    echo "Processing video:"$f
    ffmpeg -i $f -vcodec copy -acodec copy ${f%.MP4}.avi
done